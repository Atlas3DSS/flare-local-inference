use super::ServerConfig;
use crate::{
    Result,
    execution::ExecutionManifest,
    frontend::{
        Asset, Frontend, FrontendAssets, FrontendSettings,
        images::{ImageLimits, ImageProcessor},
    },
    model::ModelConfig,
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
    pub fn load(json: &str, manifest: &ExecutionManifest) -> Result<Self> {
        Self::load_inner(json, manifest).map_err(|error| match error {
            crate::Error::Execution(crate::ExecutionError::State(_)) => {
                crate::ExecutionError::Manifest(crate::Failure::caused_by(error.to_string(), error))
                    .into()
            }
            _ => error,
        })
    }

    fn load_inner(json: &str, manifest: &ExecutionManifest) -> Result<Self> {
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
            eos_token_ids: frontend_settings.eos_token_ids.clone(),
            sampling_defaults,
            speculative_token_capacity: usize::try_from(integer("speculative_token_capacity", Some(0))?).map_err(|_| "speculative token capacity exceeds usize")?,
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
        model
            .contract
            .graphs
            .retain(|g| manifest.graphs.iter().any(|m| m.name == g.graph));
        model.fit_context();
        crate::model::catalog::validate(&model, manifest)?;
        let assets = object
            .get("assets")
            .and_then(Value::as_object)
            .ok_or("assets must contain verified frontend descriptors")?;
        let jinja = matches!(frontend_settings.prompt_encoder, crate::frontend::PromptEncoder::Jinja);
        let expected = if jinja { &["tokenizer", "chat_template"][..] } else { &["tokenizer"][..] };
        if assets.len() != expected.len() || expected.iter().any(|name| !assets.contains_key(*name)) {
            return Err(format!("runtime assets must contain exactly {}", expected.join(" and ")).into());
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
        let frontend = Frontend::load_with_settings(
            FrontendAssets {
                tokenizer: asset("tokenizer")?,
                tokens: frontend_settings.tokenizer.clone(),
                chat_template: jinja.then(|| asset("chat_template")).transpose()?,
            },
            defaults,
            &frontend_settings,
        )?;
        if frontend_settings
            .image
            .as_ref()
            .map(|settings| settings.preparation_kind())
            != model.contract.frontend.image_preparation
        {
            return Err("image preparation identifier differs from its settings".into());
        }
        let images = if model.contract.vision.enabled() {
            Some(ImageProcessor::load(
                frontend_settings
                    .image
                    .ok_or("vision requires prepared image settings")?,
                image_limits(&model.contract.vision)?,
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
            scheduling_policy: match object.get("scheduler_policy") {
                None => crate::scheduler::PolicyConfig::default(),
                Some(Value::String(value)) if value == "balanced" => crate::scheduler::PolicyConfig::default(),
                Some(Value::String(value)) if value == "throughput" => crate::scheduler::PolicyConfig::throughput(),
                Some(_) => return Err("scheduler_policy must be balanced or throughput".into()),
            },
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

fn image_limits(vision: &crate::model::VisionCapabilities) -> Result<ImageLimits> {
    let max_total_patches = vision
        .features_per_request
        .checked_mul(vision.patches_per_feature)
        .ok_or("prepared image patch capacity overflow")?;
    let max_patches_per_image = vision
        .patch_capacities
        .last()
        .copied()
        .ok_or("vision requires a captured patch capacity")?
        .min(max_total_patches);
    Ok(ImageLimits {
        max_images: vision.image_count_capacity(),
        max_patches_per_image,
        max_total_patches,
        ..ImageLimits::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{frontend::images::ImageSettings, model::VisionCapabilities};

    #[test]
    fn image_limits_follow_graph_and_feature_storage_capacities() {
        for (features, merge, graph_patches, total) in [
            (16384, 4, 65536, 65536),
            (16384, 4, 16384, 65536),
            (1536, 9, 3456, 13824),
            (1024, 4, 16384, 4096),
        ] {
            let vision = VisionCapabilities {
                feature_slots: 4,
                features_per_request: features,
                max_images: 4,
                patches_per_feature: merge,
                patch_capacities: vec![graph_patches],
            };
            vision.validate().unwrap();
            let limits = image_limits(&vision).unwrap();
            assert_eq!(limits.max_images, features);
            assert_eq!(limits.max_total_patches, total);
            assert_eq!(limits.max_patches_per_image, graph_patches.min(total));
            let processor = ImageProcessor::load(ImageSettings::Moonvit {}, limits).unwrap();
            let geometry = processor.geometry(3584, 3584);
            if graph_patches == 65536 {
                let geometry = geometry.unwrap();
                assert_eq!(geometry.patches, 65536);
                assert_eq!(geometry.tokens, 16384);
            } else {
                assert!(geometry.is_err());
            }
        }
    }

    #[test]
    fn bootstrap_frontend_assets_match_the_prompt_encoder_contract() {
        use crate::execution::ExecutionManifest;
        use crate::frontend::PromptEncoder;
        use crate::model::contract::GraphOperation;
        use serde_json::json;
        use sha2::{Digest, Sha256};
        use std::path::PathBuf;

        struct FixtureDirectory(PathBuf);
        impl Drop for FixtureDirectory {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let directory = FixtureDirectory(std::env::temp_dir().join(format!(
            "flare-bootstrap-assets-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        )));
        std::fs::create_dir(&directory.0).unwrap();
        let vocab: tokenizers::models::bpe::Vocab = (0..128)
            .map(|id| ((id as u8 as char).to_string(), id))
            .collect();
        let tokenizer = tokenizers::Tokenizer::new(
            tokenizers::models::bpe::BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .build()
                .unwrap(),
        );
        let asset = |name: &str, bytes: &[u8]| {
            let path = directory.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            json!({"path":path,"sha256":format!("{:x}", Sha256::digest(bytes))})
        };
        let tokenizer_asset = asset(
            "tokenizer.json",
            tokenizer.to_string(false).unwrap().as_bytes(),
        );
        let template_asset = asset("chat_template.jinja", b"{{ messages[0].content }}");
        let mut model =
            crate::frontend::test_support::model_config(crate::model::testing::metadata());
        model.prepared_sequence_tokens = 2048;
        model.limits = SchedulerLimits {
            request_slots: 1,
            max_active: 1,
            max_sequence_tokens: 2048,
            prefill_chunk_size: 512,
            pages: 64,
            snapshots: 40,
        };
        model.eos_token_ids = vec![0];
        model.contract.graphs.retain(|graph| {
            graph.kind != GraphOperation::Speculative && graph.graph.split('-').nth(1) == Some("1")
        });
        let mut manifest: ExecutionManifest = crate::model::testing::manifest(&model);
        for (index, graph) in manifest.graphs.iter_mut().enumerate() {
            graph.handle = index + 1;
        }
        crate::model::catalog::validate(&model, &manifest).unwrap();
        let config = json!({
            "identity":"11".repeat(32), "model_id":"bootstrap-assets",
            "max_sequence_tokens":model.prepared_sequence_tokens,
            "prefill_chunk_size":model.limits.prefill_chunk_size,
            "pages":model.limits.pages, "snapshots":model.limits.snapshots,
            "model_contract":model.contract, "sampling_defaults":model.sampling_defaults,
            "frontend":{"tokenizer":{},"eos_token_ids":[0]},
            "assets":{"tokenizer":tokenizer_asset}
        });
        for (selector, expected) in [
            (1, PromptEncoder::DeepseekV4),
            (2, PromptEncoder::DeepseekV41),
        ] {
            let mut native = config.clone();
            native["frontend"]["prompt_encoder"] = selector.into();
            let loaded = Bootstrap::load(&native.to_string(), &manifest).unwrap();
            assert_eq!(loaded.frontend.prompt_encoder(), expected);
            for assets in [
                json!({}),
                json!({"chat_template":template_asset}),
                json!({"tokenizer":tokenizer_asset,"chat_template":template_asset}),
                json!({"tokenizer":tokenizer_asset,"extra":template_asset}),
            ] {
                native["assets"] = assets;
                let error = Bootstrap::load(&native.to_string(), &manifest)
                    .err()
                    .unwrap();
                assert!(
                    error
                        .to_string()
                        .contains("runtime assets must contain exactly tokenizer"),
                    "{error}"
                );
            }
        }
        let mut legacy = config.clone();
        let error = Bootstrap::load(&legacy.to_string(), &manifest)
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("runtime assets must contain exactly tokenizer and chat_template"),
            "{error}"
        );
        legacy["assets"]["chat_template"] = template_asset.clone();
        let loaded = Bootstrap::load(&legacy.to_string(), &manifest).unwrap();
        assert_eq!(loaded.frontend.prompt_encoder(), PromptEncoder::Jinja);
        for assets in [
            json!({"chat_template":template_asset}),
            json!({"tokenizer":tokenizer_asset,"chat_template":template_asset,"extra":tokenizer_asset}),
        ] {
            legacy["assets"] = assets;
            let error = Bootstrap::load(&legacy.to_string(), &manifest)
                .err()
                .unwrap();
            assert!(
                error
                    .to_string()
                    .contains("runtime assets must contain exactly tokenizer and chat_template"),
                "{error}"
            );
        }
        let mut unknown = config;
        unknown["frontend"]["prompt_encoder"] = 99.into();
        let error = Bootstrap::load(&unknown.to_string(), &manifest)
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("invalid prepared frontend settings"),
            "{error}"
        );
    }
}
