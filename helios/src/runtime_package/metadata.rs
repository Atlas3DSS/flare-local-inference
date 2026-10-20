//! Metadata-only inspection of a package's primary file.

use super::{
    InspectionOptions, archive::Archive, format::invalid, protection::hex, protobuf::wire,
};
use crate::Result;
use prost::Message;
use serde_json::{Value, json};
use std::path::Path;

/// Read bounded package metadata without authenticating signatures, opening companion files,
/// reading payloads, or initializing CUDA. The returned object describes recorded preparation
/// limits and address layouts, not the physical memory selected when serving.
pub fn unverified_metadata(
    path: &Path,
    options: InspectionOptions,
    rank: Option<usize>,
) -> Result<Value> {
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
    let mut index = metadata
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
        ranks.push(bundle);
    }
    if let Some(rank) = rank {
        if index.cluster.is_none() || rank >= ranks.len() {
            return Err(invalid(
                "rank is not a separately served rank of this package",
            ));
        }
        ranks = vec![ranks.remove(rank)];
    }
    index.ranks = ranks;
    let mut value = super::describe::describe_index(&index, options, rank)?;
    value["package"] = json!({
        "signatures_verified": false,
        "format_version": index.version,
        "bytes": file.metadata().map_err(|e| e.to_string())?.len(),
        "primary_payload_bytes": payload_bytes,
        "companions": null,
        "cluster": index.cluster,
        "publisher": {"name": null, "namespace": metadata.namespace,
            "development": metadata.development, "fingerprint": hex(&envelope.signer)},
        "official_source": metadata.official_source,
        "license": metadata.license,
    });
    Ok(value)
}
