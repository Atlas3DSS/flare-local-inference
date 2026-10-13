//! Decode generated Protobuf messages into the validated runtime description.

use super::format::*;
use crate::Result;

pub(super) mod wire {
    include!(concat!(env!("OUT_DIR"), "/flare.runtime.rs"));
}

impl wire::ExecutionGraph {
    pub(super) fn packed_tokens(&self) -> u64 {
        if self.kind == "mixed" {
            self.prefill_capacity.saturating_add(
                self.batch_capacity
                    .saturating_sub(1)
                    .saturating_mul(self.token_capacity),
            )
        } else {
            self.batch_capacity.saturating_mul(self.token_capacity)
        }
    }
}

pub(super) fn size(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid("resource size exceeds host address space"))
}

pub(super) fn sampling_defaults(
    value: wire::DefaultSamplingParameters,
) -> Result<serde_json::Value> {
    let sampling = crate::model::SamplingParameters {
        seed: required(value.seed, "sampling seed")?,
        temperature: required(value.temperature, "sampling temperature")?,
        top_k: required(value.top_k, "sampling top_k")?,
        top_p: required(value.top_p, "sampling top_p")?,
        min_p: required(value.min_p, "sampling min_p")?,
        repetition_penalty: required(value.repetition_penalty, "sampling repetition_penalty")?,
        presence_penalty: required(value.presence_penalty, "sampling presence_penalty")?,
        frequency_penalty: required(value.frequency_penalty, "sampling frequency_penalty")?,
    };
    sampling
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(serde_json::json!({
        "seed": sampling.seed, "temperature": sampling.temperature, "top_k": sampling.top_k,
        "top_p": sampling.top_p, "min_p": sampling.min_p,
        "repetition_penalty": sampling.repetition_penalty, "presence_penalty": sampling.presence_penalty,
        "frequency_penalty": sampling.frequency_penalty,
    }))
}

pub(super) fn frontend_settings(value: wire::FrontendSettings) -> Result<serde_json::Value> {
    let tokens = required(value.tokenizer, "tokenizer settings")?;
    if value.eos_token_ids.is_empty()
        || value.eos_token_ids.len() > 8
        || value
            .eos_token_ids
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != value.eos_token_ids.len()
    {
        return Err(invalid(
            "EOS token IDs must be one through eight distinct values",
        ));
    }
    let tokenizer = serde_json::json!({
        "bos_token": tokens.bos_token, "eos_token": tokens.eos_token, "unk_token": tokens.unk_token,
        "sep_token": tokens.sep_token, "pad_token": tokens.pad_token, "cls_token": tokens.cls_token,
        "mask_token": tokens.mask_token,
        "additional_special_tokens": tokens.additional_special_tokens.map(|tokens| serde_json::json!({"values": tokens.values})),
    });
    let image = value
        .image
        .map(|image| -> Result<serde_json::Value> {
            use wire::image_settings::Profile;
            Ok(match required(image.profile, "image profile")? {
                Profile::Qwen(config) => serde_json::json!({"qwen": {
                    "min_pixels": config.min_pixels, "max_pixels": config.max_pixels,
                    "position_side": config.position_side, "image_token_id": config.image_token_id,
                    "video_token_id": config.video_token_id,
                }}),
                Profile::Mimo(config) => serde_json::json!({"mimo": {
                    "min_pixels": config.min_pixels, "max_pixels": config.max_pixels,
                    "image_mean": config.image_mean, "image_std": config.image_std,
                }}),
                Profile::Deepseek4(_) => serde_json::json!({"deepseek4": {}}),
            })
        })
        .transpose()?;
    Ok(
        serde_json::json!({"tokenizer": tokenizer, "image": image, "eos_token_ids": value.eos_token_ids}),
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn defaults() -> wire::DefaultSamplingParameters {
        wire::DefaultSamplingParameters {
            seed: Some(u64::MAX),
            temperature: Some(1.),
            top_k: Some(-1),
            top_p: Some(0.95),
            min_p: Some(0.),
            repetition_penalty: Some(1.),
            presence_penalty: Some(0.),
            frequency_penalty: Some(0.),
        }
    }

    #[test]
    fn sampling_round_trip_preserves_explicit_zeroes_and_uint64_seed() {
        let value = defaults();
        let decoded =
            wire::DefaultSamplingParameters::decode(value.encode_to_vec().as_slice()).unwrap();
        let json = sampling_defaults(decoded).unwrap();
        assert_eq!(json["seed"].as_u64(), Some(u64::MAX));
        assert_eq!(json["top_k"], -1);
        assert_eq!(json["frequency_penalty"], 0.);
        #[cfg(feature = "server")]
        {
            let sampling: crate::model::SamplingParameters = serde_json::from_value(json).unwrap();
            assert_eq!(sampling.seed, u64::MAX);
            assert_eq!(sampling.top_k, -1);
        }
    }

    #[test]
    fn incomplete_or_invalid_defaults_never_inherit_model_policy() {
        let mut value = defaults();
        value.top_k = None;
        assert!(
            sampling_defaults(value)
                .unwrap_err()
                .to_string()
                .contains("sampling top_k")
        );
        let mut value = defaults();
        value.top_p = Some(0.);
        assert!(sampling_defaults(value).is_err());
        let mut value = defaults();
        value.temperature = Some(f32::NAN);
        assert!(sampling_defaults(value).is_err());
    }

    #[test]
    fn frontend_round_trip_keeps_eos_and_empty_special_token_list_distinct_from_absence() {
        let value = wire::FrontendSettings {
            tokenizer: Some(wire::TokenizerSettings {
                bos_token: Some("<bos>".into()),
                eos_token: Some("<eos>".into()),
                additional_special_tokens: Some(wire::TokenList { values: Vec::new() }),
                ..Default::default()
            }),
            eos_token_ids: vec![17, 23],
            image: None,
        };
        let decoded = wire::FrontendSettings::decode(value.encode_to_vec().as_slice()).unwrap();
        let json = frontend_settings(decoded).unwrap();
        assert_eq!(json["eos_token_ids"], serde_json::json!([17, 23]));
        assert_eq!(json["tokenizer"]["bos_token"], "<bos>");
        assert_eq!(
            json["tokenizer"]["additional_special_tokens"]["values"],
            serde_json::json!([])
        );
        assert!(json["tokenizer"]["pad_token"].is_null());
        #[cfg(feature = "server")]
        {
            let frontend: crate::frontend::FrontendSettings = serde_json::from_value(json).unwrap();
            assert!(
                frontend
                    .tokenizer
                    .additional_special_tokens
                    .unwrap()
                    .values
                    .is_empty()
            );
        }
    }

    #[test]
    fn image_contract_contains_only_the_selected_profile() {
        let value = wire::FrontendSettings {
            tokenizer: Some(wire::TokenizerSettings::default()),
            eos_token_ids: vec![1],
            image: Some(wire::ImageSettings {
                profile: Some(wire::image_settings::Profile::Deepseek4(wire::Empty {})),
            }),
        };
        let json = frontend_settings(value).unwrap();
        assert_eq!(json["image"], serde_json::json!({"deepseek4": {}}));
        #[cfg(feature = "server")]
        {
            let frontend: crate::frontend::FrontendSettings = serde_json::from_value(json).unwrap();
            let _ = crate::frontend::images::ImageProcessor::load(
                frontend.image.unwrap(),
                Default::default(),
            )
            .unwrap();
        }
    }
}
