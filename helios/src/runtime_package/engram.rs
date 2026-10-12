use super::{
    format::{Memory, Package, invalid},
    protobuf::wire,
};
use crate::{
    Result,
    engram::{Hash, Index, Plane, Table},
};
use std::collections::HashSet;

pub(super) fn decode(value: wire::Engram, start: u64, packages: &[Package]) -> Result<Index> {
    if value.compressed_vocabulary == 0
        || value
            .token_map
            .iter()
            .any(|n| *n >= value.compressed_vocabulary)
        || value.pad_token_id
            >= if value.token_map.is_empty() {
                value.compressed_vocabulary
            } else {
                value.token_map.len() as u32
            }
        || value.tables.is_empty()
    {
        return Err(invalid("invalid Engram vocabulary or table catalog"));
    }
    let mut layers = HashSet::new();
    let mut buffers = HashSet::new();
    let mut events = HashSet::new();
    let mut tables = Vec::new();
    for table in value.tables {
        let gpu = !table.request_buffer.is_empty();
        let encoding = wire::engram_table::Encoding::try_from(table.encoding)
            .map_err(|_| invalid("unknown n-gram encoding"))?;
        let (width, block, row_bytes, scale_bytes) = match encoding {
            wire::engram_table::Encoding::Fp8E4m3E8m0 => (256, 32, 256, 8),
            wire::engram_table::Encoding::Nvfp4E4m3Global
                if table.global_scale.is_finite() && table.global_scale > 0.0 =>
            {
                (160, 16, 80, 10)
            }
            _ => return Err(invalid("invalid n-gram encoding or global scale")),
        };
        let geometry = table
            .hash
            .as_ref()
            .ok_or_else(|| invalid("n-gram table lacks hash geometry"))?;
        let order = match wire::engram_hash::Method::try_from(geometry.method) {
            Ok(wire::engram_hash::Method::XorModulo) if !value.token_map.is_empty() => 4,
            Ok(wire::engram_hash::Method::EosXorModulo) if gpu && value.token_map.is_empty() => 3,
            _ => return Err(invalid("unsupported n-gram hash method or request source")),
        };
        let heads = (order - 1) * 8;
        if !layers.insert(table.layer)
            || table.rows == 0
            || table.embedding_width != width
            || table.scale_block != block
            || !events.insert(table.ready_event.clone())
            || table.ready_event.is_empty()
            || [
                &table.weight_buffer,
                &table.scale_buffer,
                &table.hash_buffer,
            ]
            .iter()
            .any(|name| name.is_empty() || !buffers.insert((*name).clone()))
        {
            return Err(invalid("invalid Engram table encoding or bindings"));
        }
        if gpu && !buffers.insert(table.request_buffer.clone()) {
            return Err(invalid("n-gram mailbox aliases another binding"));
        }
        let geometry = table
            .hash
            .ok_or_else(|| invalid("Engram table lacks hash geometry"))?;
        if geometry.orders != (2..=order as u32).flat_map(|n| [n; 8]).collect::<Vec<_>>()
            || geometry.multipliers.len() != order
            || geometry.primes.len() != heads
            || geometry.offsets.len() != heads
            || geometry.multipliers.iter().any(|m| {
                *m == 0
                    || m.checked_mul(value.compressed_vocabulary as u64)
                        .is_none_or(|n| n > i64::MAX as u64)
            })
            || geometry.primes.iter().zip(&geometry.offsets).any(|(p, o)| {
                *p < 2
                    || o.checked_add(*p)
                        .is_none_or(|n| n > table.rows || n > i64::MAX as u64)
            })
        {
            return Err(invalid("invalid Engram hash geometry"));
        }
        let plane = |span: Option<wire::Span>, row_bytes: usize| -> Result<Plane> {
            let span = span.ok_or_else(|| invalid("Engram table lacks a payload plane"))?;
            if table.rows.checked_mul(row_bytes as u64) != Some(span.nbytes) {
                return Err(invalid("Engram payload geometry mismatch"));
            }
            Ok(Plane {
                offset: start
                    .checked_add(span.offset)
                    .ok_or_else(|| invalid("Engram file offset overflow"))?,
                nbytes: usize::try_from(span.nbytes)
                    .map_err(|_| invalid("Engram plane exceeds host range"))?,
                row_bytes,
            })
        };
        let planes = [
            plane(table.weights, row_bytes)?,
            plane(table.scales, scale_bytes)?,
        ];
        for package in packages {
            if !package.events.contains(&table.ready_event) {
                return Err(invalid("rank lacks Engram readiness event"));
            }
            let execution = package
                .execution
                .as_ref()
                .ok_or_else(|| invalid("Engram rank lacks execution bindings"))?;
            let tokens = execution
                .graphs
                .iter()
                .filter(|g| matches!(g.kind.as_str(), "prefill" | "decode" | "speculative"))
                .map(|g| g.batch_capacity * g.token_capacity)
                .max()
                .ok_or_else(|| invalid("Engram catalog lacks consumers"))?;
            for (name, dtype, columns) in [
                (&table.weight_buffer, "uint8", row_bytes as u64),
                (&table.scale_buffer, "uint8", scale_bytes as u64),
                (&table.hash_buffer, "int64", heads as u64),
            ] {
                let buffer = execution
                    .buffers
                    .iter()
                    .find(|b| &b.name == name)
                    .ok_or_else(|| invalid("rank lacks Engram input buffer"))?;
                let rows = if dtype == "uint8" {
                    tokens * heads as u64
                } else {
                    tokens
                };
                if buffer.dtype != dtype
                    || buffer.shape.len() != 2
                    || buffer.shape[0] < rows
                    || buffer.shape[1] != columns
                    || buffer.strides != [columns, 1]
                    || buffer.nbytes
                        != buffer.shape[0]
                            .checked_mul(columns * if dtype == "int64" { 8 } else { 1 })
                            .unwrap_or(0)
                {
                    return Err(invalid("invalid compact Engram buffer layout"));
                }
            }
            if gpu {
                let request = execution
                    .buffers
                    .iter()
                    .find(|b| b.name == table.request_buffer)
                    .ok_or_else(|| invalid("rank lacks n-gram request mailbox"))?;
                let reference = request
                    .reference
                    .as_ref()
                    .ok_or_else(|| invalid("mailbox lacks storage"))?;
                if request.dtype != "int64"
                    || request.shape.len() != 1
                    || request.strides != [1]
                    || request.shape[0] < tokens * heads as u64 + 1
                    || request.nbytes != request.shape[0] * 8
                    || !reference.offset.is_multiple_of(8)
                    || !package.allocations.iter().any(|a| {
                        a.name == reference.allocation && matches!(a.memory, Memory::MappedHost)
                    })
                {
                    return Err(invalid("invalid mapped n-gram mailbox"));
                }
            }
            for graph in execution
                .graphs
                .iter()
                .filter(|g| matches!(g.kind.as_str(), "prefill" | "decode" | "speculative"))
            {
                if !gpu
                    && !graph.input_bindings.iter().any(|b| {
                        b.buffer_name == table.hash_buffer
                            && b.nbytes
                                >= graph.batch_capacity * graph.token_capacity * heads as u64 * 8
                    })
                {
                    return Err(invalid("Engram graph lacks hash input binding"));
                }
            }
        }
        tables.push(Table {
            planes,
            hash: Hash {
                multipliers: geometry.multipliers,
                primes: geometry.primes,
                offsets: geometry.offsets,
            },
            weights: table.weight_buffer,
            scales: table.scale_buffer,
            hashes: table.hash_buffer,
            event: table.ready_event,
            request: gpu.then_some(table.request_buffer),
        });
    }
    Ok(Index {
        pad: if value.token_map.is_empty() {
            value.pad_token_id
        } else {
            value.token_map[value.pad_token_id as usize]
        },
        token_map: value.token_map,
        tables,
    })
}
