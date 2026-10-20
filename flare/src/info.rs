use std::{fmt::Write as _, io::Write as _};

use flare_helios::runtime_package::package_metadata;
use serde_json::Value;

use crate::cli::Info;

pub fn run(args: Info) -> Result<(), Box<dyn std::error::Error>> {
    let metadata = package_metadata(&args.package)
        .map_err(|e| format!("cannot inspect package {}: {e}", args.package.display()))?;
    let output = if args.json {
        serde_json::to_string_pretty(&metadata)? + "\n"
    } else {
        render(&args, &metadata)
    };
    std::io::stdout().lock().write_all(output.as_bytes())?;
    Ok(())
}

fn display(value: &Value) -> String {
    match value {
        Value::Null => "not recorded".into(),
        Value::String(s) => s.chars().flat_map(char::escape_debug).collect(),
        _ => value.to_string(),
    }
}

fn bytes(n: u128) -> String {
    for (shift, unit) in [(40, "TiB"), (30, "GiB"), (20, "MiB"), (10, "KiB")] {
        if n >= 1u128 << shift {
            return format!("{:.2} {unit}", n as f64 / (1u128 << shift) as f64);
        }
    }
    format!("{n} B")
}

fn size(value: &Value) -> String {
    value
        .as_u64()
        .map_or_else(|| display(value), |n| bytes(n.into()))
}

fn address(value: &Value) -> String {
    value
        .as_u64()
        .map_or_else(|| "relocatable".into(), |n| format!("0x{n:016x}"))
}

fn row(output: &mut String, label: &str, value: impl std::fmt::Display) {
    writeln!(output, "  {label:<24} {value}").unwrap();
}

fn table(output: &mut String, headings: &[&str], rows: Vec<Vec<String>>) {
    let widths = (0..headings.len())
        .map(|i| {
            rows.iter()
                .map(|r| r[i].chars().count())
                .max()
                .unwrap_or(0)
                .max(headings[i].len())
        })
        .collect::<Vec<_>>();
    for columns in std::iter::once(headings.iter().map(|s| s.to_string()).collect()).chain(rows) {
        output.push_str("  ");
        for (i, column) in columns.iter().enumerate() {
            output.push_str(column);
            if i + 1 != columns.len() {
                output.extend(std::iter::repeat_n(
                    ' ',
                    widths[i] - column.chars().count() + 2,
                ));
            }
        }
        output.push('\n');
    }
}

fn render(args: &Info, metadata: &Value) -> String {
    let mut out = String::new();
    let ranks = metadata["ranks"].as_array().unwrap();
    let contract = &metadata["contract"];
    let model = &contract["model_contract"];
    writeln!(
        out,
        "Package: {}",
        display(&args.package.display().to_string().into())
    )
    .unwrap();
    row(
        &mut out,
        "Metadata",
        "unverified (publisher signatures not checked)",
    );
    row(
        &mut out,
        "Format",
        format!(
            "envelope {}, index {}",
            metadata["envelope_version"], metadata["index_version"]
        ),
    );
    row(&mut out, "Primary file", size(&metadata["file_bytes"]));
    let publisher = &metadata["publisher"];
    row(&mut out, "Publisher", display(&publisher["namespace"]));
    row(&mut out, "Source", display(&publisher["official_source"]));
    if !publisher["license"].is_null() {
        row(&mut out, "License", display(&publisher["license"]["name"]));
        if publisher["license"]["link"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
        {
            row(
                &mut out,
                "License URL",
                display(&publisher["license"]["link"]),
            );
        }
    }
    row(&mut out, "Parallel ranks", ranks.len());
    row(
        &mut out,
        "Placement",
        if metadata["cluster"].is_null() {
            "single host"
        } else {
            "multiple hosts"
        },
    );
    out.push_str("\nModel\n");
    for (label, key) in [
        ("Model ID", "model_id"),
        ("Identity", "identity"),
        ("Served names", "served_names"),
        ("Context tokens", "max_sequence_tokens"),
        ("Prefill chunk tokens", "prefill_chunk_size"),
        ("Recorded concurrency", "max_active"),
        ("Default output tokens", "max_new_tokens"),
        ("Page tokens", "page_tokens"),
        ("Prepared pages", "pages"),
        ("Prepared state rows", "state_rows"),
        ("Snapshot slots", "snapshots"),
        ("Speculative widths", "speculative_widths"),
        ("Fixed speculative width", "speculative_fixed_width"),
        ("Proposal source", "speculative_proposal_source"),
        ("Speculative token budget", "speculative_token_capacity"),
    ] {
        row(&mut out, label, display(&contract[key]));
    }
    row(&mut out, "Vocabulary", display(&model["vocabulary"]));
    row(
        &mut out,
        "Cache histories",
        display(&metadata["cache_kinds"]),
    );
    if model["vision"]["max_images"].as_u64().unwrap_or(0) > 0 {
        row(&mut out, "Vision capacity", display(&model["vision"]));
    }
    row(
        &mut out,
        "Serving reserve",
        size(&contract["reserve_bytes"]),
    );
    out.push_str("\nSampling defaults\n");
    if let Some(sampling) = metadata["sampling_defaults"].as_object() {
        for (key, value) in sampling {
            let text = if value.is_f64() {
                (value.as_f64().unwrap() as f32).to_string()
            } else {
                display(value)
            };
            row(&mut out, key, text);
        }
    } else {
        out.push_str("  not recorded\n");
    }
    out.push_str("\nRank resources (recorded sizes; cache backing is selected at load time)\n");
    for rank in ranks {
        let allocations = rank["allocations"].as_array().unwrap();
        let total = |memory: &str, dynamic: bool| -> String {
            bytes(
                allocations
                    .iter()
                    .filter(|a| a["memory"] == memory && a["sizing"].is_null() != dynamic)
                    .map(|a| {
                        a[if dynamic { "virtual_bytes" } else { "nbytes" }]
                            .as_u64()
                            .unwrap_or(0) as u128
                    })
                    .sum(),
            )
        };
        writeln!(
            out,
            "  Rank {}: SM {}, prepared concurrency {}, {} graphs, {} modules, {} functions",
            rank["rank"],
            display(&rank["architecture"]),
            display(&rank["max_active"]),
            rank["graphs"].as_array().unwrap().len(),
            rank["module_count"],
            rank["function_count"]
        )
        .unwrap();
        row(&mut out, "Fixed device allocations", total("DEVICE", false));
        row(
            &mut out,
            "Mapped host allocations",
            total("MAPPED_HOST", false),
        );
        row(&mut out, "Peer allocation views", total("PEER", false));
        row(&mut out, "Runtime cache VA maximum", total("DEVICE", true));
        if args.memory_map || args.all {
            memory_map(&mut out, rank);
        }
        if args.graphs || args.all {
            graphs(&mut out, rank);
        }
    }
    if args.all {
        for (label, key) in [
            ("Serving contract", "contract"),
            ("Frontend settings", "frontend"),
            ("Frontend assets", "assets"),
            ("Auxiliary tables", "auxiliary_tables"),
            ("Peer memory", "peer_memory"),
            ("Cluster", "cluster"),
        ] {
            writeln!(
                out,
                "\n{label}\n{}",
                serde_json::to_string_pretty(&metadata[key]).unwrap()
            )
            .unwrap();
        }
    }
    out
}

fn memory_map(out: &mut String, rank: &Value) {
    out.push_str("\n  Memory map (sizes in bytes; addresses are recorded virtual addresses)\n");
    let mut allocations = rank["allocations"]
        .as_array()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    allocations.sort_by_key(|a| a["address"].as_u64());
    table(
        out,
        &[
            "Address",
            "Memory",
            "Fixed",
            "VA extent",
            "Stored",
            "Allocation",
        ],
        allocations
            .iter()
            .map(|a| {
                vec![
                    address(&a["address"]),
                    display(&a["memory"]),
                    display(&a["nbytes"]),
                    display(&a["virtual_bytes"]),
                    display(&a["stored_bytes"]),
                    display(&a["name"]),
                ]
            })
            .collect(),
    );
    for a in allocations {
        if !a["sizing"].is_null() {
            writeln!(
                out,
                "  {}: {} × {} bytes, maximum {} units, alignment {} bytes",
                display(&a["name"]),
                display(&a["sizing"]["capacity"]),
                a["sizing"]["unit_bytes"],
                a["sizing"]["maximum"],
                a["alignment"]
            )
            .unwrap();
        }
        if let Some(payload) = a["payload"].as_object() {
            writeln!(
                out,
                "  {} payload: {} + {} ({} bytes)",
                display(&a["name"]),
                if payload["filename"] == "" {
                    "primary".into()
                } else {
                    display(&payload["filename"])
                },
                payload["offset"],
                payload["nbytes"]
            )
            .unwrap();
        }
    }
    out.push_str("\n  Address reservations (virtual address space; sizes in bytes)\n");
    table(
        out,
        &["Address", "Bytes"],
        rank["reservations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| vec![address(&r["address"]), display(&r["nbytes"])])
            .collect(),
    );
    out.push_str("\n  Execution buffer views (offsets and sizes in bytes; strides in elements)\n");
    table(
        out,
        &[
            "Buffer",
            "Allocation",
            "Offset",
            "Bytes",
            "Dtype",
            "Shape",
            "Strides",
        ],
        rank["buffers"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|b| {
                vec![
                    display(&b["name"]),
                    display(&b["reference"]["allocation"]),
                    display(&b["reference"]["offset"]),
                    display(&b["nbytes"]),
                    display(&b["dtype"]),
                    display(&b["shape"]),
                    display(&b["strides"]),
                ]
            })
            .collect(),
    );
}

fn graphs(out: &mut String, rank: &Value) {
    out.push_str("\n  Captured graphs\n");
    table(
        out,
        &[
            "Graph",
            "Kind",
            "Batch",
            "Tokens",
            "Packed rows",
            "Prefill",
            "Sampling",
            "Nodes",
            "Edges",
        ],
        rank["graphs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| {
                let e = &g["execution"];
                vec![
                    display(&g["name"]),
                    display(&e["kind"]),
                    display(&e["batch_capacity"]),
                    display(&e["token_capacity"]),
                    display(&g["packed_tokens"]),
                    display(&e["prefill_capacity"]),
                    display(&e["sampling_mode"]),
                    display(&g["node_count"]),
                    display(&g["edge_count"]),
                ]
            })
            .collect(),
    );
}
