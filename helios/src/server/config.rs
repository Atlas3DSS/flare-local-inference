use super::ServerConfig;
use crate::{
    Result,
    execution::ExecutionManifest,
    frontend::{
        Asset, Frontend, FrontendAssets, FrontendSettings,
        images::{ImageLimits, ImageProcessor},
    },
    model::{ModelConfig, ModelContract},
    scheduler::SchedulerLimits,
};
use serde_json::Value;

pub struct Bootstrap {
    pub model: ModelConfig,
    pub frontend: Frontend,
    pub images: Option<ImageProcessor>,
    pub server: ServerConfig,
}
impl Bootstrap {
    pub fn load<M: ModelContract>(json: &str, manifest: &ExecutionManifest) -> Result<Self> {
        Self::load_inner::<M>(json, manifest).map_err(|error| match error {
            crate::Error::Execution(crate::ExecutionError::State(_)) => {
                crate::ExecutionError::Manifest(crate::Failure::caused_by(error.to_string(), error))
                    .into()
            }
            _ => error,
        })
    }

    fn load_inner<M: ModelContract>(json: &str, manifest: &ExecutionManifest) -> Result<Self> {
        let value: Value = serde_json::from_str(json).map_err(|error| {
            crate::ExecutionError::Manifest(crate::Failure::caused_by(
                format!("invalid server configuration JSON: {error}"),
                error,
            ))
        })?;
        let object = value
            .as_object()
            .ok_or("server configuration must be a JSON object")?;
        if object.contains_key("generation_defaults") || object.contains_key("eos_token_ids") {
            return Err("sampling and token defaults require typed package fields".into());
        }
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
        let frontend_settings: FrontendSettings = serde_json::from_value(
            object
                .get("frontend")
                .ok_or("prepared frontend settings are required")?
                .clone(),
        )
        .map_err(|error| {
            crate::ExecutionError::Manifest(crate::Failure::caused_by(
                "invalid prepared frontend settings",
                error,
            ))
        })?;
        let sampling_defaults: crate::model::SamplingParameters = serde_json::from_value(
            object
                .get("sampling_defaults")
                .ok_or("prepared sampling defaults are required")?
                .clone(),
        )
        .map_err(|error| {
            crate::ExecutionError::Manifest(crate::Failure::caused_by(
                "invalid prepared sampling defaults",
                error,
            ))
        })?;
        sampling_defaults.validate()?;
        if identity.len() != 64 || !identity.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("model identity must be 64 hexadecimal characters".into());
        }
        let mut digest = [0u8; 32];
        for (index, byte) in digest.iter_mut().enumerate() {
            *byte =
                u8::from_str_radix(&identity[index * 2..index * 2 + 2], 16).map_err(|error| {
                    crate::ExecutionError::Manifest(crate::Failure::caused_by(
                        error.to_string(),
                        error,
                    ))
                })?;
        }
        let contract = serde_json::from_value(
            object
                .get("model_contract")
                .ok_or("model_contract metadata is required")?
                .clone(),
        )
        .map_err(|error| {
            crate::ExecutionError::Manifest(crate::Failure::caused_by(
                format!("invalid model_contract metadata: {error}"),
                error,
            ))
        })?;
        let prepared_sequence_tokens = usize::try_from(integer("max_sequence_tokens", None)?)
            .map_err(|_| "prepared context extent overflow")?;
        let mut model = ModelConfig {
            prepared_sequence_tokens,
            contract,
            model_id: text("model_id")?.into(),
            identity: digest,
            limits: SchedulerLimits {
                request_slots: manifest.max_active.next_power_of_two(),
                max_active: usize::try_from(integer(
                    "max_active",
                    Some(manifest.max_active as u64),
                )?)
                .map_err(|_| "runtime concurrency exceeds usize")?,
                max_sequence_tokens: prepared_sequence_tokens,
                prefill_chunk_size: usize::try_from(integer("prefill_chunk_size", Some(4096))?)
                    .map_err(|error| {
                        crate::ExecutionError::Manifest(crate::Failure::caused_by(
                            "prefill chunk extent overflow",
                            error,
                        ))
                    })?,
                pages: u32::try_from(integer("pages", None)?).map_err(|error| {
                    crate::ExecutionError::Manifest(crate::Failure::caused_by(
                        "page capacity overflow",
                        error,
                    ))
                })?,
                snapshots: u32::try_from(integer("snapshots", None)?).map_err(|error| {
                    crate::ExecutionError::Manifest(crate::Failure::caused_by(
                        "state row capacity overflow",
                        error,
                    ))
                })?,
            },
            eos_token_ids: frontend_settings.eos_token_ids,
            sampling_defaults,
            speculative_fixed_width: object
                .get("speculative_fixed_width")
                .filter(|v| !v.is_null())
                .map(|v| {
                    v.as_u64()
                        .and_then(|v| usize::try_from(v).ok())
                        .ok_or("speculative_fixed_width must be an integer or null")
                })
                .transpose()?,
            history_drafts: match object.get("history_drafts") {
                None => false,
                Some(Value::Bool(value)) => *value,
                _ => return Err("history_drafts must be a boolean".into()),
            },
            speculative_proposal_source: crate::scheduler::speculation::ProposalSource::parse(
                match object.get("speculative_proposal_source") {
                    None => "mtp",
                    Some(Value::String(source)) => source,
                    _ => return Err("speculative_proposal_source must be mtp or copy".into()),
                },
            )?,
            state_rows_per_request: object
                .get("state_rows_per_request")
                .map(|v| {
                    v.as_u64()
                        .and_then(|v| usize::try_from(v).ok())
                        .filter(|v| (1..=8).contains(v))
                        .ok_or("state_rows_per_request must be an integer from 1 through 8")
                })
                .transpose()?,
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
            max_new_tokens: u32::try_from(integer(
                "max_new_tokens",
                Some(integer("max_sequence_tokens", None)?),
            )?)
            .map_err(|error| {
                crate::ExecutionError::Manifest(crate::Failure::caused_by(
                    "default output limit overflow",
                    error,
                ))
            })?,
        };
        model.fit_context::<M>();
        crate::model::catalog::validate::<M>(&model, manifest)?;
        let assets = object
            .get("assets")
            .and_then(Value::as_object)
            .ok_or("assets must contain verified frontend descriptors")?;
        if assets.len() != 2
            || assets
                .keys()
                .any(|key| !matches!(key.as_str(), "tokenizer" | "chat_template"))
        {
            return Err("runtime assets must contain only tokenizer and chat_template".into());
        }
        let asset = |name: &str| -> Result<Asset> {
            let descriptor = assets
                .get(name)
                .ok_or_else(|| format!("missing asset {name}"))?;
            Ok(Asset {
                range: match (descriptor.get("offset"), descriptor.get("nbytes")) {
                    (None, None) => None,
                    (Some(offset), Some(size)) => Some((
                        offset.as_u64().ok_or("invalid asset offset")?,
                        usize::try_from(size.as_u64().ok_or("invalid asset size")?)
                            .map_err(|_| "asset size exceeds usize")?,
                    )),
                    _ => return Err("asset range requires offset and nbytes".into()),
                },
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
                tokens: frontend_settings.tokenizer,
                chat_template: asset("chat_template")?,
            },
            defaults,
        )?;
        let images = if model.contract.vision.enabled() {
            Some(ImageProcessor::load(
                frontend_settings
                    .image
                    .ok_or("vision requires prepared image settings")?,
                ImageLimits {
                    max_total_patches: 65536,
                    ..ImageLimits::default()
                },
            )?)
        } else {
            if frontend_settings.image.is_some() {
                return Err("text-only model must not declare image settings".into());
            }
            None
        };
        let optional_text = |name: &str| match object.get(name) {
            None => Ok(String::new()),
            Some(Value::String(value)) => Ok(value.clone()),
            Some(_) => Err(format!("server configuration {name} must be a string")),
        };
        let served_names = match object.get("served_names") {
            None => Vec::new(),
            Some(Value::Array(names)) => names
                .iter()
                .map(|name| name.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()
                .ok_or("served_names must contain strings")?,
            Some(_) => return Err("served_names must be an array".into()),
        };
        let server = ServerConfig {
            api_key: Some(optional_text("api_key")?).filter(|key| !key.is_empty()),
            served_names,
            context_hint: optional_text("context_hint")?,
            capacity_hint: optional_text("capacity_hint")?,
            bind: object
                .get("bind")
                .and_then(Value::as_str)
                .unwrap_or("127.0.0.1:8000")
                .parse()
                .map_err(|error| {
                    crate::ExecutionError::Manifest(crate::Failure::caused_by(
                        format!("invalid bind socket address: {error}"),
                        error,
                    ))
                })?,
            queue_capacity: usize::try_from(integer("queue_capacity", Some(16))?).map_err(
                |error| {
                    crate::ExecutionError::Manifest(crate::Failure::caused_by(
                        "request queue capacity overflow",
                        error,
                    ))
                },
            )?,
            event_capacity: usize::try_from(integer("event_capacity", Some(128))?).map_err(
                |error| {
                    crate::ExecutionError::Manifest(crate::Failure::caused_by(
                        "event queue capacity overflow",
                        error,
                    ))
                },
            )?,
            frontend_workers: usize::try_from(integer("frontend_workers", Some(2))?).map_err(
                |error| {
                    crate::ExecutionError::Manifest(crate::Failure::caused_by(
                        "frontend capacity overflow",
                        error,
                    ))
                },
            )?,
            max_body_bytes: usize::try_from(integer("max_body_bytes", Some(32 * 1024 * 1024))?)
                .map_err(|error| {
                    crate::ExecutionError::Manifest(crate::Failure::caused_by(
                        "HTTP body capacity overflow",
                        error,
                    ))
                })?,
        };
        Ok(Self {
            model,
            frontend,
            images,
            server,
        })
    }
}
