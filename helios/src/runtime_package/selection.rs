//! Select serving graph variants without changing captured storage layouts.

use std::collections::{BTreeSet, HashMap, HashSet};

use serde_json::Value;

use super::format::{Package, invalid};
use crate::{Result, execution::BATCH_CAPACITIES};

#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// Absent, [`crate::engram::Storage::choose`] picks RAM or disk at load.
    pub embedding_table_storage: Option<crate::engram::Storage>,
    pub max_concurrency: Option<usize>,
    pub mtp_draft_tokens: Option<usize>,
    pub adaptive_verification: Option<bool>,
    /// User-visible ordinals when the launcher renumbers CUDA devices.
    pub device_labels: Option<Vec<i32>>,
    pub prefix_cache: crate::offload::PrefixCacheOffload,
    /// For a package whose ranks run on separate hosts: the one rank this host loads.
    pub cluster_rank: Option<usize>,
}

impl LoadOptions {
    pub(super) fn select(&self, packages: &mut [Package], config: &mut Value) -> Result<usize> {
        let execution = packages
            .first()
            .and_then(|p| p.execution.as_ref())
            .ok_or_else(|| invalid("model package has no serving contract"))?;
        let prepared = usize::try_from(execution.max_active)
            .map_err(|_| invalid("prepared concurrency exceeds usize"))?;
        let active = self.max_concurrency.unwrap_or(prepared);
        if !(1..=64).contains(&prepared) || active == 0 || active > prepared {
            return Err(invalid(format!(
                "runtime concurrency must be 1 through the captured maximum {prepared}"
            )));
        }
        let widths = config
            .get("speculative_widths")
            .map(|value| {
                value
                    .as_array()
                    .ok_or_else(|| invalid("speculative_widths must be an array"))?
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|n| usize::try_from(n).ok())
                            .ok_or_else(|| invalid("invalid speculative width"))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        crate::scheduler::speculation::validate_widths(&widths)?;
        let maximum = execution
            .graphs
            .iter()
            .filter_map(|g| g.draft_tokens)
            .max()
            .unwrap_or(0) as usize;
        let drafts = self.mtp_draft_tokens.unwrap_or(maximum);
        if drafts > maximum {
            return Err(invalid(format!(
                "runtime MTP draft tokens must be 0 through the captured maximum {maximum}"
            )));
        }
        let fixed = config
            .get("speculative_fixed_width")
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| invalid("speculative_fixed_width must be an integer or null"))
            })
            .transpose()?;
        if fixed.is_some_and(|width| !widths.contains(&width)) {
            return Err(invalid(
                "fixed verification width is outside the captured widths",
            ));
        }
        if widths.is_empty() && self.adaptive_verification.is_some() {
            return Err(invalid(
                "verification policy requires a package captured with speculation",
            ));
        }
        let adaptive = self.adaptive_verification.unwrap_or(fixed.is_none());
        let fixed = if widths.is_empty() || adaptive {
            None
        } else {
            Some(if self.mtp_draft_tokens.is_some() {
                widths
                    .iter()
                    .copied()
                    .filter(|w| *w <= drafts + 1)
                    .max()
                    .unwrap_or(1)
            } else {
                fixed.unwrap_or_else(|| widths.last().copied().unwrap_or(1).min(drafts + 1))
            })
        };
        let enabled = drafts > 0 && fixed != Some(1) && !widths.is_empty();
        let fixed = enabled.then_some(fixed).flatten();
        let widths: Vec<_> = widths
            .into_iter()
            .filter(|width| enabled && *width <= drafts + 1)
            .collect();
        let source = match config
            .get("speculative_proposal_source")
            .and_then(Value::as_str)
            .unwrap_or("mtp")
        {
            "mtp" => 1,
            "copy" => 2,
            _ => return Err(invalid("unknown speculative proposal source")),
        };
        let mut rank_catalog = None;
        for (rank, package) in packages.iter_mut().enumerate() {
            let execution = package
                .execution
                .as_mut()
                .ok_or_else(|| invalid("rank lacks serving contract"))?;
            if execution.max_active != prepared as u64 {
                return Err(invalid("rank concurrency capacities differ"));
            }
            let mut names = HashSet::new();
            let mut variants = HashSet::new();
            for graph in &execution.graphs {
                let kind = crate::execution::GraphKind::try_from(graph.kind.as_str())?;
                crate::execution::SamplingMode::try_from(graph.sampling_mode.as_str())?;
                let speculative = kind == crate::execution::GraphKind::Speculative
                    || (kind == crate::execution::GraphKind::Mixed
                        && graph.speculation_enabled == Some(true));
                let valid = match kind {
                    crate::execution::GraphKind::Mixed => {
                        BATCH_CAPACITIES.contains(&(graph.batch_capacity as usize))
                            && graph.batch_capacity >= 2
                            && graph.batch_capacity <= prepared.next_power_of_two() as u64
                            && graph.prefill_capacity >= 64
                            && graph.prefill_capacity.is_multiple_of(64)
                            && graph.packed_tokens() <= i32::MAX as u64
                            && graph.prefill_capacity
                                <= config["prefill_chunk_size"].as_u64().unwrap_or(0)
                            && graph.speculation_enabled.is_some()
                            && if speculative {
                                (1..=8).contains(&graph.token_capacity)
                            } else {
                                graph.token_capacity == 1
                            }
                    }
                    crate::execution::GraphKind::Prefill | crate::execution::GraphKind::Vision => {
                        graph.batch_capacity == 1 && graph.token_capacity > 0
                    }
                    _ => {
                        BATCH_CAPACITIES.contains(&(graph.batch_capacity as usize))
                            && graph.batch_capacity <= prepared.next_power_of_two() as u64
                            && if kind == crate::execution::GraphKind::Speculative {
                                (1..=8).contains(&graph.token_capacity)
                            } else {
                                graph.token_capacity == 1
                            }
                    }
                };
                if !valid
                    || (kind != crate::execution::GraphKind::Mixed && graph.prefill_capacity != 0)
                    || (kind == crate::execution::GraphKind::Speculative
                        && graph.speculation_enabled == Some(false))
                    || (kind == crate::execution::GraphKind::Decode
                        && graph.speculation_enabled == Some(true))
                    || if speculative {
                        graph.verify_tokens != Some(graph.token_capacity as u32)
                            || graph
                                .draft_tokens
                                .is_none_or(|n| n > 7 || n + 1 < graph.token_capacity as u32)
                            || ![1, 2].contains(&graph.draft_source)
                    } else {
                        graph.draft_tokens.is_some()
                            || graph.verify_tokens.is_some()
                            || graph.draft_source != 0
                    }
                    || !variants.insert((
                        graph.kind.clone(),
                        graph.sampling_mode.clone(),
                        graph.batch_capacity,
                        graph.token_capacity,
                        graph.prefill_capacity,
                        graph.draft_tokens,
                        graph.draft_source,
                        graph.speculation_enabled,
                    ))
                {
                    return Err(invalid("invalid or duplicate serving graph variant"));
                }
                if !names.insert(graph.name.clone()) {
                    return Err(invalid("duplicate execution graph name"));
                }
            }
            let resources: HashSet<_> = package
                .graphs
                .iter()
                .map(|graph| graph.name.clone())
                .collect();
            if names != resources {
                return Err(invalid(
                    "execution catalog must describe every captured graph exactly once",
                ));
            }
            execution.graphs.retain(|graph| {
                graph.batch_capacity <= active.next_power_of_two() as u64
                    && graph
                        .speculation_enabled
                        .is_none_or(|value| value == enabled)
                    && (graph.kind != "decode" || !enabled)
                    && (!(graph.kind == "speculative" || (graph.kind == "mixed" && enabled))
                        || (graph.draft_source == source
                            && graph.draft_tokens.is_some_and(|n| n as usize <= drafts)
                            && widths.contains(&(graph.verify_tokens.unwrap() as usize))
                            && fixed.is_none_or(|width| graph.token_capacity == width as u64)))
            });
            let mut draft_counts = HashMap::new();
            for graph in &execution.graphs {
                if let Some(drafts) = graph.draft_tokens {
                    let key = (
                        graph.kind.clone(),
                        graph.batch_capacity,
                        graph.token_capacity,
                        graph.prefill_capacity,
                        graph.sampling_mode.clone(),
                    );
                    let count = draft_counts.entry(key).or_insert(drafts);
                    *count = (*count).max(drafts);
                }
            }
            execution.graphs.retain(|g| {
                g.draft_tokens.is_none_or(|drafts| {
                    draft_counts[&(
                        g.kind.clone(),
                        g.batch_capacity,
                        g.token_capacity,
                        g.prefill_capacity,
                        g.sampling_mode.clone(),
                    )] == drafts
                })
            });
            let mut selected_variants = HashSet::new();
            for graph in &execution.graphs {
                if !selected_variants.insert((
                    graph.kind.as_str(),
                    graph.sampling_mode.as_str(),
                    graph.batch_capacity,
                    graph.token_capacity,
                    graph.prefill_capacity,
                )) {
                    return Err(invalid(
                        "ambiguous graph variants for the selected speculation mode",
                    ));
                }
            }
            for batch in BATCH_CAPACITIES
                .into_iter()
                .filter(|batch| *batch <= active.next_power_of_two())
            {
                for kind in [
                    "init",
                    "restore",
                    "snapshot",
                    "resume",
                    if widths.is_empty() {
                        "decode"
                    } else {
                        "speculative"
                    },
                ] {
                    if !execution
                        .graphs
                        .iter()
                        .any(|g| g.kind == kind && g.batch_capacity == batch as u64)
                    {
                        return Err(invalid(format!(
                            "rank {rank} has no captured {kind} graph for batch {batch} within the requested verification limits; export the required capacity range"
                        )));
                    }
                }
            }
            let catalog: BTreeSet<_> = execution
                .graphs
                .iter()
                .map(|g| {
                    (
                        g.name.clone(),
                        g.kind.clone(),
                        g.sampling_mode.clone(),
                        g.batch_capacity,
                        g.token_capacity,
                        g.prefill_capacity,
                        g.draft_tokens,
                        g.verify_tokens,
                        g.draft_source,
                    )
                })
                .collect();
            if rank_catalog
                .as_ref()
                .is_some_and(|expected| *expected != catalog)
            {
                return Err(invalid("selected graph catalogs differ between ranks"));
            }
            rank_catalog = Some(catalog);
            let selected: HashSet<_> = execution.graphs.iter().map(|g| g.name.as_str()).collect();
            package
                .graphs
                .retain(|graph| selected.contains(graph.name.as_str()));
            prune_kernels(package)?;
        }
        let object = config
            .as_object_mut()
            .ok_or_else(|| invalid("server configuration must be an object"))?;
        object.insert("max_active".into(), active.into());
        object.insert("speculative_widths".into(), serde_json::json!(widths));
        object.insert("speculative_fixed_width".into(), serde_json::json!(fixed));
        if !enabled {
            object.insert("history_drafts".into(), false.into());
        }
        Ok(active)
    }
}

fn prune_kernels(package: &mut Package) -> Result<()> {
    let used: HashSet<_> = package
        .graphs
        .iter()
        .flat_map(|g| g.functions.iter().copied())
        .collect();
    package.functions.retain(|f| used.contains(&f.id));
    let modules: HashSet<_> = package
        .functions
        .iter()
        .map(|f| f.module.as_str())
        .collect();
    // Module globals can own persistent data referenced through opaque kernel arguments.
    package
        .modules
        .retain(|module| modules.contains(module.name.as_str()) || !module.globals.is_empty());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_package::{format::*, protobuf::wire};

    fn package() -> Package {
        let mut execution = wire::Execution {
            max_active: 4,
            input_frame_bytes: 8192,
            ..Default::default()
        };
        for batch in [1, 2, 4] {
            for enabled in [false, true] {
                for kind in ["init", "restore", "snapshot", "resume", "decode"] {
                    if enabled && kind == "decode" {
                        continue;
                    }
                    execution.graphs.push(wire::ExecutionGraph {
                        name: format!("{kind}{batch}-{}", if enabled { "mtp" } else { "no-mtp" }),
                        kind: kind.into(),
                        sampling_mode: "general".into(),
                        estimated_duration_ns: 1000,
                        batch_capacity: batch,
                        token_capacity: 1,
                        speculation_enabled: Some(enabled),
                        ..Default::default()
                    });
                }
            }
            for width in 1..=6 {
                execution.graphs.push(wire::ExecutionGraph {
                    name: format!("speculative{batch}x{width}"),
                    kind: "speculative".into(),
                    sampling_mode: "general".into(),
                    estimated_duration_ns: 1000,
                    batch_capacity: batch,
                    token_capacity: width,
                    draft_tokens: Some(width as u32 - 1),
                    verify_tokens: Some(width as u32),
                    draft_source: wire::DraftSource::Mtp as i32,
                    ..Default::default()
                });
            }
        }
        let graphs = execution
            .graphs
            .iter()
            .enumerate()
            .map(|(function, g)| Graph {
                name: g.name.clone(),
                flags: 0,
                resource: g.name.clone(),
                node_count: 1,
                edge_count: 0,
                functions: vec![function],
            })
            .collect();
        let functions = execution
            .graphs
            .iter()
            .enumerate()
            .map(|(id, g)| Function {
                id,
                module: g.name.clone(),
                symbol: "kernel".into(),
                parameter_sizes: vec![],
                attributes: vec![],
            })
            .collect();
        let modules = execution
            .graphs
            .iter()
            .map(|g| Module {
                name: g.name.clone(),
                file: format!("{}.cubin", g.name),
                globals: vec![],
            })
            .collect();
        Package {
            resources: std::collections::HashMap::new(),
            events: vec![],
            architecture: [12, 0],
            graphs,
            functions,
            modules,
            execution: Some(execution),
            context: Context::default(),
            reservations: vec![],
            allocations: ["weights", "mtp-weights", "scratch"]
                .map(|name| Allocation {
                    name: name.into(),
                    nbytes: 8192,
                    alignment: 256,
                    file: None,
                    data_spans: vec![],
                    relocations: vec![],
                    address: None,
                    memory: Memory::Device,
                    sizing: None,
                })
                .into(),
        }
    }

    fn config() -> Value {
        serde_json::json!({"speculative_widths": [1,2,3,4,5,6], "speculative_fixed_width": null})
    }

    #[test]
    fn indeterminate_graph_durations_are_loadable() {
        let mut package = package();
        package.execution.as_mut().unwrap().graphs[0].estimated_duration_ns = 0;
        LoadOptions::default()
            .select(&mut [package], &mut config())
            .unwrap();
    }

    #[test]
    fn offload_identity_separates_selected_state_without_fragmenting_compatible_limits() {
        let mut identities = HashMap::new();
        for source in ["mtp", "copy"] {
            for concurrency in 1..=4 {
                for drafts in 0..=5 {
                    for adaptive in [false, true] {
                        let mut package = package();
                        for graph in &mut package.execution.as_mut().unwrap().graphs {
                            if graph.kind == "speculative" {
                                graph.draft_source = if source == "mtp" { 1 } else { 2 };
                            }
                        }
                        let mut config = config();
                        config["identity"] = "01".repeat(32).into();
                        config["speculative_proposal_source"] = source.into();
                        LoadOptions {
                            max_concurrency: Some(concurrency),
                            mtp_draft_tokens: Some(drafts),
                            adaptive_verification: Some(adaptive),
                            ..Default::default()
                        }
                        .select(std::slice::from_mut(&mut package), &mut config)
                        .unwrap();
                        let identity = super::super::model::offload_identity(&config).unwrap();
                        assert_ne!(identity, [1; 32]);
                        let state = if drafts == 0 { "ordinary" } else { source };
                        assert_eq!(*identities.entry(state).or_insert(identity), identity);
                    }
                }
            }
        }
        assert_eq!(identities.values().collect::<HashSet<_>>().len(), 3);

        let mut config = config();
        config["identity"] = "01".repeat(32).into();
        config["speculative_fixed_width"] = 1.into();
        LoadOptions::default()
            .select(&mut [package()], &mut config)
            .unwrap();
        assert_eq!(
            super::super::model::offload_identity(&config).unwrap(),
            identities["ordinary"]
        );
        config["identity"] = "02".repeat(32).into();
        assert_ne!(
            super::super::model::offload_identity(&config).unwrap(),
            identities["ordinary"]
        );
    }

    #[test]
    fn offload_identity_respects_default_proposals_and_overridden_fixed_policy() {
        let mut adaptive_config = config();
        adaptive_config["identity"] = "ab".repeat(32).into();
        LoadOptions::default()
            .select(&mut [package()], &mut adaptive_config)
            .unwrap();
        let speculative = super::super::model::offload_identity(&adaptive_config).unwrap();
        adaptive_config["speculative_proposal_source"] = "mtp".into();
        adaptive_config["identity"] = "AB".repeat(32).into();
        assert_eq!(
            super::super::model::offload_identity(&adaptive_config).unwrap(),
            speculative
        );
        for adaptive in [None, Some(false), Some(true)] {
            let mut config = config();
            config["identity"] = "ab".repeat(32).into();
            config["speculative_fixed_width"] = 1.into();
            LoadOptions {
                adaptive_verification: adaptive,
                ..Default::default()
            }
            .select(&mut [package()], &mut config)
            .unwrap();
            assert_eq!(
                super::super::model::offload_identity(&config).unwrap() == speculative,
                adaptive == Some(true)
            );
        }
    }

    #[test]
    fn draft_count_and_verification_width_are_independent_selection_axes() {
        for limit in [3, 5] {
            let mut p = package();
            for drafts in [3, 5] {
                let name = format!("speculative2-mtp{drafts}-verify2");
                p.execution
                    .as_mut()
                    .unwrap()
                    .graphs
                    .push(wire::ExecutionGraph {
                        name: name.clone(),
                        kind: "speculative".into(),
                        sampling_mode: "general".into(),
                        estimated_duration_ns: 1000,
                        batch_capacity: 2,
                        token_capacity: 2,
                        draft_tokens: Some(drafts),
                        verify_tokens: Some(2),
                        draft_source: 1,
                        ..Default::default()
                    });
                p.graphs.push(Graph {
                    name,
                    resource: "graph".into(),
                    node_count: 1,
                    edge_count: 0,
                    flags: 0,
                    functions: vec![0],
                });
            }
            LoadOptions {
                max_concurrency: Some(2),
                mtp_draft_tokens: Some(limit),
                adaptive_verification: Some(true),
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut config())
            .unwrap();
            let variants: Vec<_> = p
                .execution
                .unwrap()
                .graphs
                .into_iter()
                .filter(|g| {
                    g.kind == "speculative" && g.batch_capacity == 2 && g.verify_tokens == Some(2)
                })
                .collect();
            assert_eq!(variants.len(), 1);
            assert_eq!(variants[0].draft_tokens, Some(limit as u32));
        }
    }

    #[test]
    fn draft_limit_comes_from_draft_declarations_not_verification_capacity() {
        let mut p = package();
        p.execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| g.token_capacity <= 2);
        for graph in &mut p.execution.as_mut().unwrap().graphs {
            if graph.kind == "speculative" && graph.token_capacity == 2 {
                graph.draft_tokens = Some(5);
            }
        }
        let names: HashSet<_> = p
            .execution
            .as_ref()
            .unwrap()
            .graphs
            .iter()
            .map(|g| g.name.clone())
            .collect();
        p.graphs.retain(|g| names.contains(&g.name));
        let mut config = serde_json::json!({"speculative_widths": [1,2]});
        LoadOptions {
            mtp_draft_tokens: Some(5),
            adaptive_verification: Some(false),
            ..Default::default()
        }
        .select(std::slice::from_mut(&mut p), &mut config)
        .unwrap();
        assert_eq!(config["speculative_fixed_width"], 2);
        assert!(
            p.execution
                .unwrap()
                .graphs
                .iter()
                .filter(|g| g.kind == "speculative")
                .all(|g| g.draft_tokens == Some(5) && g.verify_tokens == Some(2))
        );
    }

    #[test]
    fn every_concurrency_and_width_reduction_selects_only_required_graphs() {
        for concurrency in 1usize..=4 {
            for drafts in 0..=5 {
                for adaptive in [false, true] {
                    let mut packages = vec![package(), package()];
                    let mut config = config();
                    let options = LoadOptions {
                        max_concurrency: Some(concurrency),
                        mtp_draft_tokens: Some(drafts),
                        adaptive_verification: Some(adaptive),
                        ..Default::default()
                    };
                    options.select(&mut packages, &mut config).unwrap();
                    let batches = concurrency.next_power_of_two().ilog2() as usize + 1;
                    let expected = batches * (4 + if adaptive { drafts + 1 } else { 1 });
                    for package in &packages {
                        let execution = package.execution.as_ref().unwrap();
                        assert_eq!(execution.max_active, 4);
                        assert_eq!(execution.input_frame_bytes, 8192);
                        assert_eq!(package.graphs.len(), expected);
                        assert_eq!(package.functions.len(), expected);
                        assert_eq!(package.modules.len(), expected);
                        assert_eq!(package.allocations.len(), 3);
                        assert_eq!(package.allocations[1].name, "mtp-weights");
                        for graph in &package.graphs {
                            assert_eq!(
                                package
                                    .functions
                                    .iter()
                                    .find(|f| f.id == graph.functions[0])
                                    .unwrap()
                                    .module,
                                graph.name
                            );
                        }
                    }
                    assert_eq!(config["max_active"], concurrency);
                    assert_eq!(
                        config["speculative_widths"].as_array().unwrap().len(),
                        if drafts == 0 { 0 } else { drafts + 1 }
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_limits_and_missing_graphs_fail_before_resource_loading() {
        for options in [
            LoadOptions {
                max_concurrency: Some(0),
                ..Default::default()
            },
            LoadOptions {
                max_concurrency: Some(5),
                ..Default::default()
            },
            LoadOptions {
                mtp_draft_tokens: Some(6),
                ..Default::default()
            },
        ] {
            assert!(options.select(&mut [package()], &mut config()).is_err());
        }
        let mut p = package();
        p.graphs.retain(|g| g.name != "speculative2x3");
        p.execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| g.name != "speculative2x3");
        let options = LoadOptions {
            mtp_draft_tokens: Some(2),
            adaptive_verification: Some(false),
            ..Default::default()
        };
        assert!(
            options
                .select(&mut [p], &mut config())
                .unwrap_err()
                .to_string()
                .contains("batch 2")
        );
    }

    #[test]
    fn tensor_parallel_catalogs_must_match() {
        let mut packages = vec![package(), package()];
        packages[1].graphs.retain(|g| g.name != "speculative2x3");
        packages[1]
            .execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| g.name != "speculative2x3");
        assert!(
            LoadOptions::default()
                .select(&mut packages, &mut config())
                .unwrap_err()
                .to_string()
                .contains("differ between ranks")
        );
    }

    #[test]
    fn policy_defaults_and_overrides_select_the_recorded_catalog() {
        let mut fixed_config = config();
        fixed_config["speculative_fixed_width"] = 6.into();
        let mut packages = [package()];
        LoadOptions::default()
            .select(&mut packages, &mut fixed_config)
            .unwrap();
        assert_eq!(packages[0].graphs.len(), 15);
        let mut packages = [package()];
        LoadOptions {
            mtp_draft_tokens: Some(2),
            adaptive_verification: Some(true),
            ..Default::default()
        }
        .select(&mut packages, &mut fixed_config)
        .unwrap();
        assert!(fixed_config["speculative_fixed_width"].is_null());
        assert_eq!(packages[0].graphs.len(), 21);
    }

    #[test]
    fn globals_remain_even_when_their_module_has_no_selected_kernel() {
        let mut package = package();
        package.modules.push(Module {
            name: "data".into(),
            file: "data.cubin".into(),
            globals: vec![Global {
                symbol: "weights".into(),
                file: "weights.bin".into(),
                nbytes: 8,
                relocations: vec![],
            }],
        });
        let mut packages = [package];
        LoadOptions {
            max_concurrency: Some(1),
            mtp_draft_tokens: Some(0),
            ..Default::default()
        }
        .select(&mut packages, &mut config())
        .unwrap();
        assert_eq!(packages[0].graphs.len(), 5);
        assert_eq!(packages[0].modules.len(), 6);
        assert_eq!(packages[0].modules.last().unwrap().name, "data");
    }

    #[test]
    fn pinned_zero_selects_ordinary_operations_and_disables_draft_scheduling() {
        for adaptive in [None, Some(false), Some(true)] {
            let mut p = package();
            let mut config = config();
            config["history_drafts"] = true.into();
            LoadOptions {
                mtp_draft_tokens: Some(0),
                adaptive_verification: adaptive,
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut config)
            .unwrap();
            assert_eq!(config["speculative_widths"], serde_json::json!([]));
            assert!(config["speculative_fixed_width"].is_null());
            assert_eq!(config["history_drafts"], false);
            assert!(
                p.execution
                    .as_ref()
                    .unwrap()
                    .graphs
                    .iter()
                    .all(|g| g.speculation_enabled == Some(false))
            );
            assert_eq!(
                p.execution
                    .as_ref()
                    .unwrap()
                    .graphs
                    .iter()
                    .filter(|g| g.kind == "decode")
                    .count(),
                3
            );
        }
        let mut config = config();
        config["speculative_fixed_width"] = 1.into();
        LoadOptions::default()
            .select(&mut [package()], &mut config)
            .unwrap();
        assert_eq!(config["speculative_widths"], serde_json::json!([]));
    }

    #[test]
    fn pinned_zero_requires_ordinary_decode_instead_of_using_speculative_width_one() {
        let mut p = package();
        let names: HashSet<_> = p
            .execution
            .as_ref()
            .unwrap()
            .graphs
            .iter()
            .filter(|g| g.kind == "decode")
            .map(|g| g.name.clone())
            .collect();
        p.execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| !names.contains(&g.name));
        p.graphs.retain(|g| !names.contains(&g.name));
        let error = LoadOptions {
            mtp_draft_tokens: Some(0),
            ..Default::default()
        }
        .select(&mut [p], &mut config())
        .unwrap_err();
        assert!(error.to_string().contains("no captured decode graph"));
    }

    #[test]
    fn prefill_selection_matches_the_decode_mode() {
        for drafts in [0, 5] {
            let mut p = package();
            for enabled in [false, true] {
                let name = format!("prefill64-{enabled}");
                p.execution
                    .as_mut()
                    .unwrap()
                    .graphs
                    .push(wire::ExecutionGraph {
                        name: name.clone(),
                        kind: "prefill".into(),
                        sampling_mode: "general".into(),
                        estimated_duration_ns: 1000,
                        batch_capacity: 1,
                        token_capacity: 64,
                        speculation_enabled: Some(enabled),
                        ..Default::default()
                    });
                p.graphs.push(Graph {
                    name,
                    resource: "prefill".into(),
                    node_count: 1,
                    edge_count: 0,
                    flags: 0,
                    functions: vec![0],
                });
            }
            LoadOptions {
                mtp_draft_tokens: Some(drafts),
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut config())
            .unwrap();
            let selected: Vec<_> = p
                .execution
                .as_ref()
                .unwrap()
                .graphs
                .iter()
                .filter(|g| g.kind == "prefill")
                .collect();
            assert_eq!(selected.len(), 1);
            assert_eq!(selected[0].speculation_enabled, Some(drafts > 0));
        }
    }
    fn mixed_package() -> Package {
        let mut p = package();
        for enabled in [false, true] {
            for prefill in [64, 512] {
                for width in if enabled { vec![1, 3, 6] } else { vec![1] } {
                    let name = format!("mixed-4-{prefill}-{width}-{enabled}");
                    p.execution
                        .as_mut()
                        .unwrap()
                        .graphs
                        .push(wire::ExecutionGraph {
                            name: name.clone(),
                            kind: "mixed".into(),
                            sampling_mode: "general".into(),
                            batch_capacity: 4,
                            token_capacity: width,
                            prefill_capacity: prefill,
                            speculation_enabled: Some(enabled),
                            draft_tokens: enabled.then_some(width as u32 - 1),
                            verify_tokens: enabled.then_some(width as u32),
                            draft_source: if enabled { 1 } else { 0 },
                            ..Default::default()
                        });
                    p.graphs.push(Graph {
                        name,
                        resource: "mixed".into(),
                        node_count: 1,
                        edge_count: 0,
                        flags: 0,
                        functions: vec![0],
                    });
                }
            }
        }
        p
    }

    #[test]
    fn mixed_selection_preserves_prompt_variants_and_applies_decode_limits() {
        for (drafts, adaptive, widths) in [
            (0, true, vec![1]),
            (2, true, vec![1, 3]),
            (2, false, vec![3]),
        ] {
            let mut p = mixed_package();
            let mut c = config();
            c["prefill_chunk_size"] = 512.into();
            LoadOptions {
                mtp_draft_tokens: Some(drafts),
                adaptive_verification: Some(adaptive),
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut c)
            .unwrap();
            let mixed: Vec<_> = p
                .execution
                .as_ref()
                .unwrap()
                .graphs
                .iter()
                .filter(|g| g.kind == "mixed")
                .collect();
            assert_eq!(mixed.len(), 2 * widths.len());
            for prefill in [64, 512] {
                for &width in &widths {
                    let graph = mixed
                        .iter()
                        .find(|g| g.prefill_capacity == prefill && g.token_capacity == width)
                        .unwrap();
                    assert_eq!(graph.speculation_enabled, Some(drafts > 0));
                    assert_eq!(graph.packed_tokens(), prefill + 3 * width);
                }
            }
        }
    }

    #[test]
    fn mixed_selection_rejects_rank_shape_mismatch_and_invalid_mode() {
        let mut ranks = [mixed_package(), mixed_package()];
        let mut c = config();
        c["prefill_chunk_size"] = 512.into();
        ranks[1]
            .execution
            .as_mut()
            .unwrap()
            .graphs
            .iter_mut()
            .find(|g| g.kind == "mixed" && g.speculation_enabled == Some(true))
            .unwrap()
            .prefill_capacity = 128;
        assert!(
            LoadOptions::default()
                .select(&mut ranks, &mut c)
                .unwrap_err()
                .to_string()
                .contains("catalogs differ")
        );
        for mutation in 0..4 {
            let mut p = mixed_package();
            let g = p
                .execution
                .as_mut()
                .unwrap()
                .graphs
                .iter_mut()
                .find(|g| g.kind == "mixed")
                .unwrap();
            match mutation {
                0 => g.speculation_enabled = None,
                1 => g.token_capacity = 3,
                2 => g.prefill_capacity = 65,
                _ => g.prefill_capacity = 1024,
            }
            let mut c = config();
            c["prefill_chunk_size"] = 512.into();
            assert!(LoadOptions::default().select(&mut [p], &mut c).is_err());
        }
    }
}
