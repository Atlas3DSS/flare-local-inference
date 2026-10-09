use super::ServerConfig;
use crate::{
    Result,
    execution::ExecutionManifest,
    frontend::{
        Asset, Frontend, FrontendAssets,
        images::{ImageLimits, ImageProcessor},
    },
    model::ModelConfig,
    scheduler::SchedulerLimits,
};
use serde_json::Value;

pub struct Bootstrap {
    pub model: ModelConfig,
    pub frontend: Frontend,
    pub images: ImageProcessor,
    pub server: ServerConfig,
}
impl Bootstrap {
    pub fn load(json: &str, manifest: &ExecutionManifest) -> Result<Self> {
        let value: Value = serde_json::from_str(json)
            .map_err(|error| format!("invalid server configuration JSON: {error}"))?;
        let object = value
            .as_object()
            .ok_or("server configuration must be a JSON object")?;
        let text = |name: &str| {
            object
                .get(name)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("server configuration requires string {name}"))
        };
        let integer = |name: &str, default: Option<u64>| {
            object
                .get(name)
                .map(|value| {
                    value.as_u64().ok_or_else(|| {
                        format!("server configuration {name} must be a nonnegative integer")
                    })
                })
                .unwrap_or_else(|| {
                    default.ok_or_else(|| format!("server configuration requires {name}"))
                })
        };
        let identity = text("identity")?;
        if identity.len() != 64 || !identity.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("model identity must be 64 hexadecimal characters".into());
        }
        let mut digest = [0u8; 32];
        for (index, byte) in digest.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&identity[index * 2..index * 2 + 2], 16)
                .map_err(|error| error.to_string())?;
        }
        let mut model = ModelConfig {
            model_id: text("model_id")?.into(),
            identity: digest,
            limits: SchedulerLimits {
                max_active: manifest.max_active,
                max_sequence_tokens: usize::try_from(integer("max_sequence_tokens", None)?)
                    .map_err(|_| "context extent overflow")?,
                prefill_chunk_size: usize::try_from(integer("prefill_chunk_size", Some(4096))?)
                    .map_err(|_| "prefill chunk extent overflow")?,
                pages: u32::try_from(integer("pages", None)?)
                    .map_err(|_| "page capacity overflow")?,
                snapshots: u32::try_from(integer("snapshots", None)?)
                    .map_err(|_| "snapshot capacity overflow")?,
            },
            eos_token_ids: object
                .get("eos_token_ids")
                .and_then(Value::as_array)
                .ok_or("eos_token_ids must be an array")?
                .iter()
                .map(|id| {
                    id.as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or_else(|| "invalid EOS token ID".to_owned())
                })
                .collect::<Result<Vec<_>>>()?,
            vision_features_per_request: usize::try_from(integer(
                "vision_features_per_request",
                Some(16384),
            )?)
            .map_err(|_| "vision feature capacity overflow")?,
            sampling_defaults: crate::model::SamplingParameters::default(),
            speculative_fixed_width: object
                .get("speculative_fixed_width")
                .filter(|v| !v.is_null())
                .map(|v| {
                    v.as_u64()
                        .and_then(|v| usize::try_from(v).ok())
                        .ok_or("speculative_fixed_width must be an integer or null")
                })
                .transpose()?,
            speculative_copy: match object.get("speculative_copy") {
                None => true,
                Some(Value::Bool(value)) => *value,
                _ => return Err("speculative_copy must be a boolean".into()),
            },
            speculative_proposal_source: crate::scheduler::speculation::ProposalSource::parse(
                match object.get("speculative_proposal_source") {
                    None => "mtp",
                    Some(Value::String(source)) => source,
                    _ => return Err("speculative_proposal_source must be mtp or copy".into()),
                },
            )?,
            speculative_widths: object
                .get("speculative_widths")
                .map(|v| {
                    v.as_array()
                        .ok_or("speculative_widths must be an array")?
                        .iter()
                        .map(|w| {
                            w.as_u64()
                                .and_then(|v| usize::try_from(v).ok())
                                .ok_or("invalid speculative width")
                        })
                        .collect::<std::result::Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default(),
            max_new_tokens: u32::try_from(integer("max_new_tokens", Some(256))?)
                .map_err(|_| "default output limit overflow")?,
        };
        model.validate(manifest)?;
        let assets = object
            .get("assets")
            .and_then(Value::as_object)
            .ok_or("assets must contain verified frontend descriptors")?;
        let asset = |name: &str| -> Result<Asset> {
            let descriptor = assets
                .get(name)
                .ok_or_else(|| format!("missing asset {name}"))?;
            Ok(Asset {
                path: descriptor
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or("asset path must be a string")?
                    .into(),
                sha256: descriptor
                    .get("sha256")
                    .and_then(Value::as_str)
                    .ok_or("asset sha256 must be a string")?
                    .into(),
            })
        };
        let generation: Value = serde_json::from_slice(&asset("generation_config")?.read()?)
            .map_err(|error| format!("invalid generation config: {error}"))?;
        let mut generation = generation
            .as_object()
            .ok_or("generation config must be an object")?
            .clone();
        if let Some(overrides) = object.get("generation_defaults") {
            generation.extend(
                overrides
                    .as_object()
                    .ok_or("generation_defaults must be an object")?
                    .clone(),
            );
        }
        if generation.get("do_sample") == Some(&Value::Bool(false)) {
            generation.insert("temperature".into(), Value::from(0));
        }
        model.sampling_defaults =
            super::request::sampling_parameters(&generation, &model.sampling_defaults)?;
        let mut defaults = match object.get("template_defaults") {
            None => serde_json::Map::new(),
            Some(Value::Object(defaults)) => defaults.clone(),
            _ => return Err("template_defaults must be an object".into()),
        };
        defaults
            .entry("add_generation_prompt")
            .or_insert(Value::Bool(true));
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset("tokenizer")?,
                tokenizer_config: asset("tokenizer_config")?,
                chat_template: asset("chat_template")?,
            },
            defaults,
        )?;
        let images = ImageProcessor::load(
            asset("preprocessor_config")?,
            asset("model_config")?,
            ImageLimits {
                max_total_patches: 65536,
                ..ImageLimits::default()
            },
        )?;
        let server = ServerConfig {
            bind: object
                .get("bind")
                .and_then(Value::as_str)
                .unwrap_or("127.0.0.1:8000")
                .parse()
                .map_err(|error| format!("invalid bind socket address: {error}"))?,
            queue_capacity: usize::try_from(integer("queue_capacity", Some(16))?)
                .map_err(|_| "request queue capacity overflow")?,
            event_capacity: usize::try_from(integer("event_capacity", Some(128))?)
                .map_err(|_| "event queue capacity overflow")?,
            frontend_workers: usize::try_from(integer("frontend_workers", Some(2))?)
                .map_err(|_| "frontend capacity overflow")?,
            max_body_bytes: usize::try_from(integer("max_body_bytes", Some(32 * 1024 * 1024))?)
                .map_err(|_| "HTTP body capacity overflow")?,
            qualification_token_trace: match object.get("qualification_token_trace") {
                None => false,
                Some(Value::Bool(enabled)) => *enabled,
                _ => return Err("qualification_token_trace must be a boolean".into()),
            },
        };
        Ok(Self {
            model,
            frontend,
            images,
            server,
        })
    }
}
