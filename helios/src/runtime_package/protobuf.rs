//! Decode generated Protobuf messages into the validated runtime description.

use super::format::*;
use crate::Result;
use prost::Message;

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
fn node(value: wire::Node) -> Result<Node> {
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
    })
}

pub(super) fn decode(bytes: &[u8]) -> Result<Package> {
    let value = wire::Package::decode(bytes)
        .map_err(|e| invalid(format!("invalid protobuf runtime package: {e}")))?;
    if value.version != 3 {
        return Err(invalid("unsupported protobuf runtime package version"));
    }
    let context = required(value.context, "CUDA context")?;
    let package = Package {
        version: value.version,
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
                nbytes: size(v.nbytes)?,
                alignment: size(v.alignment)?,
                file: v.file,
                data_spans: convert(v.data_spans, |v| Ok([size(v.offset)?, size(v.nbytes)?]))?,
                relocations: convert(v.relocations, relocation)?,
                address: v.address,
                memory: match wire::Memory::try_from(v.memory) {
                    Ok(wire::Memory::Device) => Memory::Device,
                    Ok(wire::Memory::MappedHost) => Memory::MappedHost,
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
        functions: convert(value.functions, |v| {
            Ok(Function {
                module: v.module,
                symbol: v.symbol,
                parameter_sizes: convert(v.parameter_sizes, size)?,
                attributes: v.attributes.into_iter().map(|v| [v.id, v.value]).collect(),
            })
        })?,
        graphs: convert(value.graphs, |v| {
            Ok(Graph {
                name: v.name,
                flags: v.flags,
                nodes: convert(v.nodes, node)?,
                edges: convert(v.edges, |v| {
                    Ok(Edge {
                        source: size(v.source)?,
                        destination: size(v.destination)?,
                        data: array(v.data, "edge data")?,
                    })
                })?,
            })
        })?,
        execution: value.execution,
    };
    Ok(package)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message() -> wire::Package {
        wire::Package {
            version: 3,
            architecture: vec![12, 0],
            context: Some(wire::Context::default()),
            allocations: vec![wire::Allocation {
                name: "scratch".into(),
                nbytes: 256,
                alignment: 256,
                ..Default::default()
            }],
            graphs: vec![wire::Graph {
                name: "empty".into(),
                nodes: vec![wire::Node {
                    operation: Some(wire::node::Operation::Empty(wire::Empty {})),
                }],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn protobuf_decodes_zero_storage_and_validates_required_fields() {
        let value = message();
        decode(&value.encode_to_vec()).unwrap().validate().unwrap();
        let mut invalid_message = value.clone();
        invalid_message.version = 4;
        assert!(decode(&invalid_message.encode_to_vec()).is_err());
        invalid_message = value.clone();
        invalid_message.allocations[0].memory = 5;
        assert!(decode(&invalid_message.encode_to_vec()).is_err());
        invalid_message = value.clone();
        invalid_message.graphs[0].nodes[0].operation = None;
        assert!(decode(&invalid_message.encode_to_vec()).is_err());
        invalid_message = value;
        invalid_message.context = None;
        assert!(decode(&invalid_message.encode_to_vec()).is_err());
        assert!(decode(&[0xff]).is_err());
    }
}
