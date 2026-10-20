//! Metadata-only inspection of a package's primary file.

use super::{archive::Archive, format::invalid, protection::hex, protobuf::wire};
use crate::Result;
use prost::Message;
use serde_json::{Value, json};
use std::path::Path;

/// Read bounded package metadata without authenticating signatures, opening companion files,
/// reading payloads, or initializing CUDA. The returned object describes recorded preparation
/// limits and address layouts, not the physical memory selected when serving.
pub fn package_metadata(path: &Path) -> Result<Value> {
    let (file, _, payload_bytes, bytes) = Archive::envelope(path)?;
    let envelope = wire::ProtectedIndex::decode(bytes.as_slice())
        .map_err(|e| invalid(format!("invalid signed envelope: {e}")))?;
    if envelope.version != 4
        || envelope.groups.len() != 3
        || envelope
            .groups
            .iter()
            .enumerate()
            .any(|(i, g)| g.kind != i as u32 + 1)
        || envelope.ranks.is_empty()
    {
        return Err(invalid("invalid protected package envelope"));
    }
    let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())
        .map_err(|e| invalid(format!("invalid publisher metadata: {e}")))?;
    let index = metadata
        .index
        .ok_or_else(|| invalid("missing model index"))?;
    if !matches!(index.version, 4 | 5) {
        return Err(invalid("unsupported model index"));
    }
    if !index.ranks.is_empty() || metadata.ranks as usize != envelope.ranks.len() {
        return Err(invalid("inconsistent package rank count"));
    }
    let mut ranks = Vec::new();
    for (position, group) in envelope.ranks.iter().enumerate() {
        if group.kind != 4 {
            return Err(invalid("invalid rank signature group"));
        }
        let rank = wire::RankProtection::decode(group.body.as_slice())
            .map_err(|e| invalid(format!("invalid rank metadata: {e}")))?;
        if rank.rank as usize != position {
            return Err(invalid("unordered package ranks"));
        }
        let bundle = rank.bundle.ok_or_else(|| invalid("missing rank bundle"))?;
        let mut allocations = Vec::new();
        for a in &bundle.allocations {
            let memory = wire::Memory::try_from(a.memory)
                .map_err(|_| invalid("unknown allocation memory kind"))?;
            let virtual_bytes = match &a.sizing {
                Some(s) => {
                    if !a.alignment.is_power_of_two() {
                        return Err(invalid("invalid allocation alignment"));
                    }
                    s.unit_bytes
                        .checked_mul(s.maximum)
                        .and_then(|n| n.checked_add(a.alignment - 1))
                        .map(|n| n & !(a.alignment - 1))
                        .ok_or_else(|| invalid("allocation extent overflow"))?
                }
                None => a.nbytes,
            };
            let stored_bytes = a.data_spans.iter().try_fold(0u64, |n, s| {
                n.checked_add(s.nbytes)
                    .ok_or_else(|| invalid("stored allocation size overflow"))
            })?;
            let sizing = a
                .sizing
                .as_ref()
                .map(|s| -> Result<Value> {
                    let capacity = wire::Capacity::try_from(s.capacity)
                        .map_err(|_| invalid("unknown allocation capacity"))?;
                    Ok(
                        json!({"capacity": capacity.as_str_name(), "unit_bytes": s.unit_bytes,
                    "maximum": s.maximum}),
                    )
                })
                .transpose()?;
            allocations.push(json!({
                "name": a.name, "memory": memory.as_str_name(), "address": a.address,
                "nbytes": a.nbytes, "virtual_bytes": virtual_bytes, "alignment": a.alignment,
                "stored_bytes": stored_bytes, "sizing": sizing, "data_spans": a.data_spans,
                "payload": a.file.as_ref().and_then(|name| bundle.resources.get(name)),
            }));
        }
        let execution = bundle.execution.as_ref();
        let graphs = bundle
            .graphs
            .iter()
            .map(|g| {
                let shape = execution.and_then(|e| e.graphs.iter().find(|s| s.name == g.name));
                json!({"name": g.name, "node_count": g.node_count, "edge_count": g.edge_count,
                "execution": shape, "packed_tokens": shape.map(|s| s.packed_tokens())})
            })
            .collect::<Vec<_>>();
        ranks.push(json!({
            "rank": rank.rank, "architecture": bundle.architecture,
            "max_active": execution.map(|e| e.max_active),
            "allocations": allocations, "reservations": bundle.reservations,
            "buffers": execution.map(|e| &e.buffers), "graphs": graphs,
            "module_count": bundle.modules.len(), "function_count": bundle.functions.len(),
        }));
    }
    let cache_kinds = index
        .contract
        .as_ref()
        .and_then(|c| c.model_contract.as_ref())
        .and_then(|m| m.cache.as_ref())
        .map(|c| {
            c.regions
                .iter()
                .map(|r| {
                    wire::HistoryKind::try_from(r.kind)
                        .map(|k| k.as_str_name().to_owned())
                        .unwrap_or_else(|_| format!("UNKNOWN({})", r.kind))
                })
                .collect::<std::collections::BTreeSet<_>>()
        });
    let tables = index
        .engram
        .as_ref()
        .map(|e| {
            e.tables.iter().map(|t| json!({
        "layer": t.layer, "rows": t.rows, "embedding_width": t.embedding_width,
        "encoding": t.encoding, "weights": t.weights, "scales": t.scales, "shards": t.shards,
    })).collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(json!({
        "signatures_verified": false, "envelope_version": envelope.version,
        "index_version": index.version, "file_bytes": file.metadata().map_err(|e| e.to_string())?.len(),
        "primary_payload_bytes": payload_bytes,
        "publisher": {"namespace": metadata.namespace, "official_source": metadata.official_source,
            "development": metadata.development, "signer": hex(&envelope.signer), "license": metadata.license},
        "contract": index.contract, "cache_kinds": cache_kinds, "sampling_defaults": index.sampling_defaults,
        "frontend": index.frontend, "cluster": index.cluster, "peer_memory": index.peer_memory,
        "assets": index.assets, "auxiliary_tables": tables, "ranks": ranks,
    }))
}
