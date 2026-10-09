//! Native tokenization and rendering over verified Hugging Face assets.

#[cfg(feature = "images")]
pub mod images;
mod json;

use std::fs;
use std::path::PathBuf;

use minijinja::{Environment, Error as TemplateError, ErrorKind, Value};
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::{Error, Failure, FrontendError, Result};

pub struct Asset {
    pub path: PathBuf,
    pub sha256: String,
}

impl Asset {
    pub(crate) fn read(&self) -> Result<Vec<u8>> {
        if !self.path.is_absolute() {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!(
                    "frontend asset path must be absolute: {}",
                    self.path.display()
                ),
            ))));
        }
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!("invalid expected SHA256 for {}", self.path.display()),
            ))));
        }
        let bytes = fs::read(&self.path).map_err(|error| {
            Error::frontend(FrontendError::Asset(Failure::caused_by(
                format!("cannot read {}: {error}", self.path.display()),
                error,
            )))
        })?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!(
                    "frontend asset SHA256 mismatch for {}: expected {}, received {actual}",
                    self.path.display(),
                    self.sha256
                ),
            ))));
        }
        Ok(bytes)
    }
}

pub struct FrontendAssets {
    pub tokenizer: Asset,
    pub tokenizer_config: Asset,
    pub chat_template: Asset,
}

pub struct RenderedPrompt {
    pub text: String,
    pub token_ids: Vec<u32>,
}

pub struct Frontend {
    tokenizer: Tokenizer,
    environment: Environment<'static>,
    template_source: String,
    defaults: Map<String, Json>,
}

impl Frontend {
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn template_default(&self, name: &str) -> Option<&Json> {
        self.defaults.get(name)
    }

    pub fn load(assets: FrontendAssets, template_defaults: Map<String, Json>) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_bytes(assets.tokenizer.read()?).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                format!("invalid tokenizer: {error}"),
                error,
            )))
        })?;
        tokenizer.with_truncation(None).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })?;
        tokenizer.with_padding(None);
        let config: Json =
            serde_json::from_slice(&assets.tokenizer_config.read()?).map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::caused_by(
                    format!("invalid tokenizer config: {error}"),
                    error,
                )))
            })?;
        if !config.is_object() {
            return Err(Error::frontend(FrontendError::Tokenizer(Failure::new(
                "tokenizer config must be a JSON object",
            ))));
        }
        if config
            .get("clean_up_tokenization_spaces")
            .and_then(Json::as_bool)
            == Some(true)
        {
            return Err(Error::frontend(FrontendError::Tokenizer(Failure::new(
                "frontend requires tokenizer clean_up_tokenization_spaces=false",
            ))));
        }
        let template_source = String::from_utf8(assets.chat_template.read()?).map_err(|error| {
            Error::frontend(FrontendError::Template(Failure::caused_by(
                format!("chat template is not UTF-8: {error}"),
                error,
            )))
        })?;
        let mut environment = Environment::new();
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_filter("tojson", json::tojson);
        environment.add_function(
            "raise_exception",
            |message: Value| -> std::result::Result<Value, TemplateError> {
                Err(TemplateError::new(
                    ErrorKind::InvalidOperation,
                    message.to_string(),
                ))
            },
        );
        environment
            .add_template_owned("chat", template_source.clone())
            .map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(
                    error.to_string(),
                    error,
                )))
            })?;
        let mut defaults = Map::new();
        for name in [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "additional_special_tokens",
        ] {
            if let Some(value) = config.get(name).filter(|value| !value.is_null()) {
                defaults.insert(name.into(), special_token_contents(value));
            }
        }
        defaults.insert("add_generation_prompt".into(), Json::Bool(false));
        defaults.insert("tools".into(), Json::Null);
        defaults.insert("documents".into(), Json::Null);
        defaults.extend(template_defaults);
        Ok(Self {
            tokenizer,
            environment,
            template_source,
            defaults,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })
    }

    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })
    }

    pub fn render_chat(
        &self,
        messages: &[Json],
        options: &Map<String, Json>,
    ) -> Result<RenderedPrompt> {
        if messages.is_empty() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "Cannot apply chat template to an empty conversation. Provide at least one message.",
            ))));
        }
        let mut context = self.defaults.clone();
        context.extend(options.clone());
        let continuation = match context.remove("continue_final_message") {
            None | Some(Json::Null | Json::Bool(false)) => None,
            Some(Json::Bool(true)) => Some("content".to_owned()),
            Some(Json::String(field)) if field.is_empty() => None,
            Some(Json::String(field)) => Some(field),
            _ => {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "continue_final_message must be a boolean or field name",
                ))));
            }
        };
        if continuation.is_some() && context.get("add_generation_prompt") == Some(&Json::Bool(true))
        {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead.",
            ))));
        }
        if let Some(tools) = context.get("tools").filter(|value| !value.is_null()) {
            if !tools
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "Tools should either be a JSON schema, or a callable function with type hints and a docstring suitable for auto-conversion to a schema.",
                ))));
            }
        }
        if let Some(documents) = context.get("documents").filter(|value| !value.is_null()) {
            if !documents
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "Documents should be a list of dicts with 'title' and 'text' keys!",
                ))));
            }
        }
        let mut messages = messages.to_vec();
        let final_text = continuation
            .map(|field| self.mark_continuation(&mut messages, &field))
            .transpose()?;
        context.insert("messages".into(), Json::Array(messages));
        let mut text = self
            .environment
            .get_template("chat")
            .map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(
                    error.to_string(),
                    error,
                )))
            })?
            .render(context)
            .map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(
                    error
                        .detail()
                        .unwrap_or("chat template rendering failed")
                        .to_owned(),
                    error,
                )))
            })?;
        if let Some(final_text) = final_text {
            let marker = "CONTINUE_FINAL_MESSAGE_TAG";
            let index = text.rfind(marker).filter(|_| text.contains(final_text.trim())).ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new("continue_final_message is set but the final message does not appear in the chat after applying the chat template!"))))?;
            let preserved_space = text[index + marker.len()..].starts_with(' ');
            text.truncate(index);
            if !preserved_space {
                text.truncate(text.trim_end().len());
            }
        }
        let token_ids = self.encode(&text, false)?;
        Ok(RenderedPrompt { text, token_ids })
    }

    fn mark_continuation(&self, messages: &mut [Json], field: &str) -> Result<String> {
        let message = messages.last_mut().ok_or_else(|| {
            Error::frontend(FrontendError::Input(Failure::new(
                "list index out of range",
            )))
        })?;
        let value = message.get_mut(field).filter(|value| !value.is_null()).ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new(format!("continue_final_message is set but the final message has no \"{field}\" to continue!")))))?;
        if !self.template_source.contains(field) {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                format!(
                    "continue_final_message is set to \"{field}\" but this is not an accepted field in the chat_template"
                ),
            ))));
        }
        let text = if let Some(blocks) = value.as_array_mut() {
            blocks
                .iter_mut()
                .rev()
                .find_map(|block| block.get_mut("text"))
                .and_then(|value| value.as_str())
                .map(str::to_owned)
        } else {
            value.as_str().map(str::to_owned)
        };
        let original = text.ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new("continue_final_message is set but we could not find any text to continue in the final message!"))))?;
        let target = if let Some(blocks) = value.as_array_mut() {
            blocks
                .iter_mut()
                .rev()
                .find_map(|block| block.get_mut("text"))
                .unwrap()
        } else {
            value
        };
        *target = Json::String(format!("{original}CONTINUE_FINAL_MESSAGE_TAG "));
        Ok(original)
    }

    pub fn incremental_decoder(&self, skip_special_tokens: bool) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: &self.tokenizer,
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            finished: false,
        }
    }
}

fn special_token_contents(value: &Json) -> Json {
    match value {
        Json::Object(value) => value.get("content").cloned().unwrap_or(Json::Null),
        Json::Array(values) => Json::Array(values.iter().map(special_token_contents).collect()),
        _ => value.clone(),
    }
}

pub struct IncrementalDecoder<'a> {
    tokenizer: &'a Tokenizer,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    finished: bool,
}

impl IncrementalDecoder<'_> {
    pub fn push(&mut self, token_ids: &[u32]) -> Result<Option<String>> {
        if self.finished {
            return Err(Error::frontend(FrontendError::Tokenizer(Failure::new(
                "incremental decoder is finished",
            ))));
        }
        if token_ids.is_empty() {
            return Ok(None);
        }
        tokenizers::tokenizer::step_decode_stream(
            self.tokenizer,
            token_ids.to_vec(),
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })
    }

    pub fn finish(&mut self) -> Result<String> {
        if self.finished {
            return Ok(String::new());
        }
        let remainder = self
            .tokenizer
            .decode(&self.ids, self.skip_special_tokens)
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })?;
        let suffix = remainder
            .strip_prefix(&self.prefix)
            .ok_or_else(|| {
                Error::frontend(FrontendError::Tokenizer(Failure::new(
                    "incremental decoder final text changed its emitted prefix",
                )))
            })?
            .to_owned();
        self.finished = true;
        Ok(suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_is_shareable_between_native_request_workers() {
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<Frontend>();
    }

    #[test]
    fn missing_asset_retains_io_failure() {
        use std::error::Error as _;

        let asset = Asset {
            path: std::env::temp_dir()
                .join(format!("sparkgap-missing-asset-{}", std::process::id()))
                .join("tokenizer.json"),
            sha256: "0".repeat(64),
        };
        let error = asset.read().unwrap_err();
        assert!(matches!(error, Error::Frontend(FrontendError::Asset(_))));
        let cause = error
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap();
        assert_eq!(cause.kind(), std::io::ErrorKind::NotFound);
        let cloned = error.clone();
        assert_eq!(cloned.to_string(), error.to_string());
        assert!(std::ptr::eq(
            error.source().unwrap(),
            cloned.source().unwrap()
        ));
    }

    #[test]
    fn render_failure_retains_template_cause_and_detail() {
        use std::error::Error as _;

        let mut environment = Environment::new();
        environment.add_function(
            "raise_exception",
            || -> std::result::Result<(), TemplateError> {
                Err(TemplateError::new(
                    ErrorKind::InvalidOperation,
                    "fixture rejection",
                ))
            },
        );
        environment
            .add_template_owned("chat", "{{ raise_exception() }}".to_owned())
            .unwrap();
        let frontend = Frontend {
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            environment,
            template_source: "{{ raise_exception() }}".to_owned(),
            defaults: Map::new(),
        };
        let error = frontend
            .render_chat(
                &[serde_json::json!({"role":"user","content":"x"})],
                &Map::new(),
            )
            .err()
            .unwrap();
        assert!(matches!(error, Error::Frontend(FrontendError::Template(_))));
        assert_eq!(error.to_string(), "fixture rejection");
        let cause = error
            .source()
            .unwrap()
            .downcast_ref::<TemplateError>()
            .unwrap();
        assert_eq!(cause.kind(), ErrorKind::InvalidOperation);
        let input = frontend.render_chat(&[], &Map::new()).err().unwrap();
        assert!(matches!(input, Error::Frontend(FrontendError::Input(_))));
        assert!(input.source().is_none());
    }

    #[test]
    fn verified_asset_rejects_content_changed_after_fingerprinting() {
        let directory = std::env::temp_dir().join(format!("sparkgap-asset-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("tokenizer_config.json");
        let original = b"{\"eos_token\":\"<eos>\"}";
        fs::write(&path, original).unwrap();
        let asset = Asset {
            path: path.clone(),
            sha256: format!("{:x}", Sha256::digest(original)),
        };
        assert_eq!(asset.read().unwrap(), original);
        fs::write(&path, b"{\"eos_token\":\"changed\"}").unwrap();
        assert!(matches!(
            asset.read().unwrap_err(),
            Error::Frontend(FrontendError::Asset(_))
        ));
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
