//! Native tokenization and rendering over verified Hugging Face assets.

#[cfg(feature = "images")]
pub mod images;
mod json;

use std::fs;
use std::path::PathBuf;

use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::Result;

pub struct Asset {
    pub path: PathBuf,
    pub sha256: String,
}

impl Asset {
    pub(crate) fn read(&self) -> Result<Vec<u8>> {
        if !self.path.is_absolute() {
            return Err(format!(
                "frontend asset path must be absolute: {}",
                self.path.display()
            ));
        }
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "invalid expected SHA256 for {}",
                self.path.display()
            ));
        }
        let bytes = fs::read(&self.path)
            .map_err(|error| format!("cannot read {}: {error}", self.path.display()))?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(format!(
                "frontend asset SHA256 mismatch for {}: expected {}, received {actual}",
                self.path.display(),
                self.sha256
            ));
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
        let mut tokenizer = Tokenizer::from_bytes(assets.tokenizer.read()?)
            .map_err(|error| format!("invalid tokenizer: {error}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| error.to_string())?;
        tokenizer.with_padding(None);
        let config: Json = serde_json::from_slice(&assets.tokenizer_config.read()?)
            .map_err(|error| format!("invalid tokenizer config: {error}"))?;
        if !config.is_object() {
            return Err("tokenizer config must be a JSON object".into());
        }
        if config
            .get("clean_up_tokenization_spaces")
            .and_then(Json::as_bool)
            == Some(true)
        {
            return Err("frontend requires tokenizer clean_up_tokenization_spaces=false".into());
        }
        let template_source = String::from_utf8(assets.chat_template.read()?)
            .map_err(|error| format!("chat template is not UTF-8: {error}"))?;
        let mut environment = Environment::new();
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_filter("tojson", json::tojson);
        environment.add_function(
            "raise_exception",
            |message: Value| -> std::result::Result<Value, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message.to_string()))
            },
        );
        environment
            .add_template_owned("chat", template_source.clone())
            .map_err(|error| error.to_string())?;
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
            .map_err(|error| error.to_string())
    }

    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| error.to_string())
    }

    pub fn render_chat(
        &self,
        messages: &[Json],
        options: &Map<String, Json>,
    ) -> Result<RenderedPrompt> {
        if messages.is_empty() {
            return Err("Cannot apply chat template to an empty conversation. Provide at least one message.".into());
        }
        let mut context = self.defaults.clone();
        context.extend(options.clone());
        let continuation = match context.remove("continue_final_message") {
            None | Some(Json::Null | Json::Bool(false)) => None,
            Some(Json::Bool(true)) => Some("content".to_owned()),
            Some(Json::String(field)) if field.is_empty() => None,
            Some(Json::String(field)) => Some(field),
            _ => return Err("continue_final_message must be a boolean or field name".into()),
        };
        if continuation.is_some() && context.get("add_generation_prompt") == Some(&Json::Bool(true))
        {
            return Err("continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead.".into());
        }
        if let Some(tools) = context.get("tools").filter(|value| !value.is_null()) {
            if !tools
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err("Tools should either be a JSON schema, or a callable function with type hints and a docstring suitable for auto-conversion to a schema.".into());
            }
        }
        if let Some(documents) = context.get("documents").filter(|value| !value.is_null()) {
            if !documents
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err(
                    "Documents should be a list of dicts with 'title' and 'text' keys!".into(),
                );
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
            .map_err(|error| error.to_string())?
            .render(context)
            .map_err(|error| {
                error
                    .detail()
                    .unwrap_or("chat template rendering failed")
                    .to_owned()
            })?;
        if let Some(final_text) = final_text {
            let marker = "CONTINUE_FINAL_MESSAGE_TAG";
            let index = text.rfind(marker).filter(|_| text.contains(final_text.trim())).ok_or("continue_final_message is set but the final message does not appear in the chat after applying the chat template!")?;
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
        let message = messages.last_mut().ok_or("list index out of range")?;
        let value = message.get_mut(field).filter(|value| !value.is_null()).ok_or_else(|| format!("continue_final_message is set but the final message has no \"{field}\" to continue!"))?;
        if !self.template_source.contains(field) {
            return Err(format!(
                "continue_final_message is set to \"{field}\" but this is not an accepted field in the chat_template"
            ));
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
        let original = text.ok_or("continue_final_message is set but we could not find any text to continue in the final message!")?;
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
            return Err("incremental decoder is finished".into());
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
        .map_err(|error| error.to_string())
    }

    pub fn finish(&mut self) -> Result<String> {
        if self.finished {
            return Ok(String::new());
        }
        let remainder = self
            .tokenizer
            .decode(&self.ids, self.skip_special_tokens)
            .map_err(|error| error.to_string())?;
        let suffix = remainder
            .strip_prefix(&self.prefix)
            .ok_or("incremental decoder final text changed its emitted prefix")?
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
        assert!(asset.read().unwrap_err().contains("SHA256 mismatch"));
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
