//! Decode generated Protobuf messages into the validated runtime description.

use super::format::*;
use crate::Result;

pub(super) mod wire {
    include!(concat!(env!("OUT_DIR"), "/flare.runtime.rs"));
}

pub(super) fn size(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid("resource size exceeds host address space"))
}
fn required<T>(value: Option<T>, name: &str) -> Result<T> {
    value.ok_or_else(|| invalid(format!("missing protobuf {name}")))
}
fn convert<T, U>(values: Vec<T>, f: impl Fn(T) -> Result<U>) -> Result<Vec<U>> {
    values.into_iter().map(f).collect()
}
fn array<T, const N: usize>(values: Vec<T>, name: &str) -> Result<[T; N]> {
    values
        .try_into()
        .map_err(|_| invalid(format!("invalid protobuf {name} width")))
}
fn reference(value: wire::Reference) -> Result<Reference> {
    Ok(Reference {
        allocation: value.allocation,
        offset: size(value.offset)?,
    })
}
fn relocation(value: wire::Relocation) -> Result<Relocation> {
    Ok(Relocation {
        offset: size(value.offset)?,
        target: reference(required(value.target, "relocation target")?)?,
        kind: match wire::RelocationKind::try_from(value.kind) {
            Ok(wire::RelocationKind::Pointer) => RelocationKind::Pointer,
            Ok(wire::RelocationKind::TensorMap) => RelocationKind::TensorMap,
            Err(_) => return Err(invalid("unknown protobuf relocation kind")),
        },
    })
}
fn blob(value: wire::Blob) -> Result<Blob> {
    Ok(Blob {
        data: value.data,
        relocations: convert(value.relocations, relocation)?,
    })
}
pub(super) fn node(value: wire::Node) -> Result<Node> {
    use wire::node::Operation;
    Ok(match required(value.operation, "node operation")? {
        Operation::Kernel(value) => Node::Kernel {
            function: size(value.function)?,
            grid: array(value.grid, "kernel grid")?,
            block: array(value.block, "kernel block")?,
            shared: value.shared,
            arguments: convert(value.arguments, blob)?,
            attributes: convert(value.attributes, |value| {
                Ok(Attribute {
                    id: value.id,
                    value: blob(required(value.value, "node attribute")?)?,
                })
            })?,
        },
        Operation::Memset(value) => Node::Memset {
            destination: reference(required(value.destination, "memset destination")?)?,
            pitch: size(value.pitch)?,
            value: value.value,
            element_size: value.element_size,
            width: size(value.width)?,
            height: size(value.height)?,
        },
        Operation::Copy(value) => Node::Copy {
            source: reference(required(value.source, "copy source")?)?,
            destination: reference(required(value.destination, "copy destination")?)?,
            nbytes: size(value.nbytes)?,
        },
        Operation::Empty(_) => Node::Empty,
        Operation::EventWait(value) => Node::EventWait { event: value.event },
    })
}

pub(super) fn decode(value: wire::BundleIndex) -> Result<Package> {
    let context = required(value.context, "CUDA context")?;
    let package = Package {
        resources: value.resources,
        events: value.events,
        architecture: array(value.architecture, "architecture")?,
        context: Context {
            flags: context.flags,
            cache: context.cache,
            limits: convert(context.limits, |v| Ok([size(v.id)?, size(v.value)?]))?,
        },
        reservations: convert(value.reservations, |v| {
            Ok(Reservation {
                address: v.address,
                nbytes: size(v.nbytes)?,
            })
        })?,
        allocations: convert(value.allocations, |v| {
            Ok(Allocation {
                name: v.name,
                sizing: v.sizing,
                nbytes: size(v.nbytes)?,
                alignment: size(v.alignment)?,
                file: v.file,
                data_spans: convert(v.data_spans, |v| Ok([size(v.offset)?, size(v.nbytes)?]))?,
                relocations: convert(v.relocations, relocation)?,
                address: v.address,
                memory: match wire::Memory::try_from(v.memory) {
                    Ok(wire::Memory::Device) => Memory::Device,
                    Ok(wire::Memory::MappedHost) => Memory::MappedHost,
                    Ok(wire::Memory::Peer) => Memory::Peer,
                    Err(_) => return Err(invalid("unknown protobuf memory kind")),
                },
            })
        })?,
        modules: convert(value.modules, |v| {
            Ok(Module {
                name: v.name,
                file: v.file,
                globals: convert(v.globals, |v| {
                    Ok(Global {
                        symbol: v.symbol,
                        file: v.file,
                        nbytes: size(v.nbytes)?,
                        relocations: convert(v.relocations, relocation)?,
                    })
                })?,
            })
        })?,
        functions: convert(
            value.functions.into_iter().enumerate().collect(),
            |(id, v)| {
                Ok(Function {
                    id,
                    module: v.module,
                    symbol: v.symbol,
                    parameter_sizes: convert(v.parameter_sizes, size)?,
                    attributes: v.attributes.into_iter().map(|v| [v.id, v.value]).collect(),
                })
            },
        )?,
        graphs: convert(value.graphs, |v| {
            Ok(Graph {
                name: v.name,
                flags: v.flags,
                resource: v.resource,
                node_count: size(v.node_count)?,
                edge_count: size(v.edge_count)?,
                functions: convert(v.functions, size)?,
            })
        })?,
        execution: value.execution,
    };
    Ok(package)
}

pub(super) fn edge(value: wire::Edge) -> Result<Edge> {
    Ok(Edge {
        source: size(value.source)?,
        destination: size(value.destination)?,
        data: array(value.data, "edge data")?,
    })
}

pub(super) fn json(value: prost_types::Value) -> Result<serde_json::Value> {
    use prost_types::value::Kind;
    Ok(match required(value.kind, "configuration value")? {
        Kind::NullValue(_) => serde_json::Value::Null,
        Kind::BoolValue(v) => v.into(),
        Kind::StringValue(v) => v.into(),
        Kind::NumberValue(v) => {
            if !v.is_finite() {
                return Err(invalid("nonfinite configuration number"));
            }
            if v >= 0.0 && v.fract() == 0.0 && v < u64::MAX as f64 {
                (v as u64).into()
            } else if v.fract() == 0.0 && v >= i64::MIN as f64 && v < 0.0 {
                (v as i64).into()
            } else {
                v.into()
            }
        }
        Kind::ListValue(v) => {
            serde_json::Value::Array(v.values.into_iter().map(json).collect::<Result<_>>()?)
        }
        Kind::StructValue(v) => serde_json::Value::Object(
            v.fields
                .into_iter()
                .map(|(k, v)| Ok((k, json(v)?)))
                .collect::<Result<_>>()?,
        ),
    })
}
