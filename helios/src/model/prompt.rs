//! Prepared frontend syntax and image layouts construct shared request data.

use crate::{
    Result,
    frontend::{
        Frontend, RenderedPrompt,
        images::{ImageProcessor, RgbImage},
    },
    model::{ModelConfig, PreparedInput, PromptData, contract::ImageLayoutKind},
};
use serde_json::{Map, Value};

pub fn prepare_input(
    config: &ModelConfig,
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: Vec<RgbImage>,
) -> Result<PreparedInput> {
    if rendered
        .token_ids
        .iter()
        .any(|token| config.contract.frontend.rejected_tokens.contains(token))
    {
        return Err("prompt contains an input token outside the prepared frontend contract".into());
    }
    let prepared = match config.contract.frontend.image_layout {
        Some(ImageLayoutKind::Qwen) => crate::frontend::images::layouts::qwen::prepare_input(
            frontend, processor, rendered, images,
        )?,
        Some(ImageLayoutKind::Moonvit) => crate::frontend::images::layouts::moonvit::prepare_input(
            frontend, processor, rendered, images,
        )?,
        Some(ImageLayoutKind::Radio) => crate::frontend::images::layouts::radio::prepare_input(
            frontend, processor, rendered, images,
        )?,
        Some(ImageLayoutKind::Step) => crate::frontend::images::layouts::step::prepare_input(
            frontend, processor, rendered, images,
        )?,
        Some(ImageLayoutKind::Mimo) => crate::frontend::images::layouts::mimo::prepare_input(
            frontend, processor, rendered, images,
        )?,
        Some(ImageLayoutKind::DeepseekRowMajor) => {
            crate::frontend::images::layouts::deepseek::prepare_row_major_input(
                config.contract.vocabulary,
                frontend,
                processor,
                rendered,
                images,
            )?
        }
        Some(ImageLayoutKind::Deepseek) => {
            crate::frontend::images::layouts::deepseek::prepare_input(
                config.contract.vocabulary,
                frontend,
                processor,
                rendered,
                images,
            )?
        }
        None => {
            if !images.is_empty() {
                return Err("prepared input contract has no image layout".into());
            }
            PreparedInput {
                tokens: rendered.token_ids.clone(),
                data: PromptData {
                    positions: if config.contract.frontend.position_dimensions == 3 {
                        (0..rendered.token_ids.len())
                            .map(|p| [p as i64; 3])
                            .collect()
                    } else {
                        vec![]
                    },
                    ..Default::default()
                },
                conditioning: vec![],
                images: None,
            }
        }
    };
    super::validate_prompt_data(config, &prepared.tokens, &prepared.data)?;
    Ok(prepared)
}

pub fn system_boundary(
    config: &ModelConfig,
    frontend: &Frontend,
    tokens: &[u32],
    messages: &[Value],
    options: &Map<String, Value>,
) -> Option<usize> {
    let rule = config.contract.frontend.prefix_boundary.as_ref()?;
    let count = messages
        .iter()
        .take_while(|message| {
            message
                .get("role")
                .and_then(Value::as_str)
                .is_some_and(|role| rule.roles.iter().any(|r| r == role))
        })
        .count();
    if count < rule.minimum_messages {
        return None;
    }
    let mut messages = messages[..count].to_vec();
    if let Some(role) = &rule.append_role {
        messages.push(serde_json::json!({"role": role, "content": ""}));
    }
    let mut options = options.clone();
    options.insert("add_generation_prompt".into(), false.into());
    options.insert(
        "continue_final_message".into(),
        rule.continue_final_message.into(),
    );
    let prefix = frontend.render_boundary(&messages, &options).ok()?;
    let length = prefix.token_ids.len();
    (length > 0
        && length < tokens.len()
        && tokens.starts_with(&prefix.token_ids)
        && prefix.text.starts_with(&rule.required_start)
        && prefix.text.ends_with(&rule.required_end)
        && prefix.text.contains(&rule.required_marker))
    .then_some(length)
}

pub fn prepare_output(
    config: &ModelConfig,
    frontend: &Frontend,
    rendered: &RenderedPrompt,
    messages: &[Value],
    options: &Map<String, Value>,
    structured: bool,
) -> Result<crate::output::OutputConfig> {
    let syntax = &config.contract.frontend;
    let effective = |name: &str| {
        options
            .get(name)
            .or_else(|| frontend.template_default(name))
    };
    let continuation = effective("continue_final_message")
        .is_some_and(|v| v == &Value::Bool(true) || v.as_str().is_some_and(|s| !s.is_empty()));
    let open_assistant = effective("add_generation_prompt") == Some(&Value::Bool(true))
        || (continuation && messages.last().is_some_and(|m| m["role"] == "assistant"));
    let prefix = open_assistant
        .then(|| rendered.text.rsplit_once(&syntax.assistant_marker))
        .flatten()
        .map(|(_, suffix)| suffix)
        .filter(|suffix| {
            !syntax
                .message_end_markers
                .iter()
                .any(|end| suffix.contains(end))
        });
    let mut reasoning = syntax.reasoning.as_ref().is_some_and(|r| {
        prefix.is_some_and(|prefix| {
            prefix
                .rfind(&r.open)
                .is_some_and(|open| prefix.rfind(&r.close).is_none_or(|close| open > close))
        })
    });
    if let Some(r) = syntax
        .reasoning
        .as_ref()
        .filter(|r| r.kind == super::contract::ReasoningParserKind::XtmlThink)
    {
        // Ordinary XTML text can spell a delimiter without containing its control tokens.
        let last_marker = |marker: &str| -> Result<Option<usize>> {
            let ids = frontend.encode(marker, false)?;
            if ids.is_empty() {
                return Err("reasoning delimiters must encode to nonempty token sequences".into());
            }
            Ok(rendered
                .token_ids
                .windows(ids.len())
                .rposition(|tokens| tokens == ids))
        };
        let opened = last_marker(&r.open)?;
        let closed = last_marker(&r.close)?;
        reasoning =
            open_assistant && opened.is_some_and(|open| closed.is_none_or(|close| open > close));
    }
    let tools = options
        .get("tools")
        .or_else(|| frontend.template_default("tools"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !tools.is_empty() && syntax.tools.is_none() {
        return Err("tool parsing is not prepared".into());
    }
    let control_tokens = if syntax
        .tools
        .as_ref()
        .is_some_and(|tool| tool.kind == super::contract::ToolParserKind::Xtml)
    {
        ["<|open|>", "<|close|>", "<|sep|>"]
            .into_iter()
            .map(|marker| {
                let ids = frontend.encode(marker, false)?;
                if ids.len() != 1 {
                    return Err("XTML delimiters must each occupy one tokenizer token".into());
                }
                Ok((ids[0], marker.to_owned()))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![]
    };
    let start = if !tools.is_empty()
        && syntax.tools.as_ref().is_some_and(|tool| {
            tool.kind == super::contract::ToolParserKind::Dsml
                && prefix.is_some_and(|suffix| {
                    let open = if tool.tag_separator == " " {
                        "<｜DSML｜ calls>"
                    } else {
                        "<｜DSML｜tool_calls>"
                    };
                    suffix.trim_end().ends_with(open)
                })
        }) {
        crate::output::tagged::OutputStart::ToolBody
    } else if reasoning {
        crate::output::tagged::OutputStart::Reasoning
    } else {
        crate::output::tagged::OutputStart::Content
    };
    Ok(crate::output::OutputConfig {
        syntax: syntax.clone(),
        control_tokens,
        content: crate::output::tagged::OutputConfig {
            structured,
            start,
            tools,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn xtml_continuation_distinguishes_literal_channel_markers() {
        use crate::frontend::{Asset, FrontendAssets};
        use crate::model::contract::{ReasoningParserKind, ToolParserKind};
        use sha2::{Digest, Sha256};

        let root = std::env::temp_dir().join(format!("flare-xtml-prefix-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let vocab: tokenizers::models::bpe::Vocab = (0..128)
            .map(|id| ((id as u8 as char).to_string(), id))
            .collect();
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer
            .add_special_tokens(
                ["<|open|>", "<|close|>", "<|sep|>"]
                    .map(|text| tokenizers::AddedToken::from(text, true)),
            )
            .unwrap();
        let asset = |name: &str, bytes: &[u8]| {
            let path = root.join(name);
            std::fs::write(&path, bytes).unwrap();
            Asset {
                range: None,
                path,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            }
        };
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset(
                    "tokenizer.json",
                    tokenizer.to_string(false).unwrap().as_bytes(),
                ),
                tokens: Default::default(),
                chat_template: Some(asset("chat_template.jinja", b"{{ messages[0].content }}")),
            },
            Map::new(),
        )
        .unwrap();
        std::fs::remove_dir_all(root).unwrap();
        let mut config = super::super::tests::config();
        let mut syntax = serde_json::from_str::<crate::model::PreparedContract>(include_str!(
            "../../tests/support/contracts/qwen38.json"
        ))
        .unwrap()
        .frontend;
        syntax.assistant_marker = "<|open|>message role=\"assistant\"<|sep|>".into();
        syntax.message_end_markers = vec!["<|close|>message<|sep|>".into()];
        syntax.tools.as_mut().unwrap().kind = ToolParserKind::Xtml;
        let reason = syntax.reasoning.as_mut().unwrap();
        reason.kind = ReasoningParserKind::XtmlThink;
        reason.open = "<|open|>think<|sep|>".into();
        reason.close = "<|close|>think<|sep|>".into();
        config.contract.frontend = syntax;
        for (protocol, literal, expected) in [
            (
                "<|open|>think<|sep|>",
                "literal <|close|>think<|sep|>",
                true,
            ),
            (
                "<|open|>think<|sep|>",
                "literal <|close|>message<|sep|>",
                true,
            ),
            (
                "<|open|>think<|sep|><|close|>think<|sep|><|open|>response<|sep|>",
                "literal <|open|>think<|sep|>",
                false,
            ),
        ] {
            let prefix = config.contract.frontend.assistant_marker.clone() + protocol;
            let mut token_ids = frontend.encode(&prefix, false).unwrap();
            token_ids.extend(literal.bytes().map(u32::from));
            let rendered = RenderedPrompt {
                image_sources: vec![],
                text: prefix + literal,
                token_ids,
            };
            let output = prepare_output(
                &config,
                &frontend,
                &rendered,
                &[json!({"role":"assistant", "content":literal})],
                json!({"continue_final_message":true}).as_object().unwrap(),
                false,
            )
            .unwrap();
            assert_eq!(
                output.content.start == crate::output::tagged::OutputStart::Reasoning,
                expected,
                "{literal}"
            );
        }
    }

    #[test]
    fn initial_reasoning_uses_only_an_open_assistant_prefix() {
        let mut config = super::super::tests::config();
        let prepared: crate::model::PreparedContract =
            serde_json::from_str(include_str!("../../tests/support/contracts/qwen38.json"))
                .unwrap();
        config.contract.frontend = prepared.frontend;
        let frontend = crate::server::output::tests::frontend(&["x"]);
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/frontend/qwen38.json"
        )))
        .unwrap();
        for (name, reasoning) in [
            ("generation_prompt", true),
            ("thinking_disabled", false),
            ("assistant_continuation", false),
            ("reasoning_continuation", true),
            ("continuation_trailing_spaces", false),
            ("continuation_text_blocks", false),
        ] {
            let case = fixture["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|case| case["name"] == name)
                .unwrap();
            let rendered = RenderedPrompt {
                image_sources: vec![],
                text: case["rendered"].as_str().unwrap().into(),
                token_ids: vec![],
            };
            let output = prepare_output(
                &config,
                &frontend,
                &rendered,
                case["messages"].as_array().unwrap(),
                case["options"].as_object().unwrap(),
                false,
            )
            .unwrap();
            assert_eq!(
                output.content.start == crate::output::tagged::OutputStart::Reasoning,
                reasoning,
                "{name}"
            );
        }
        for role in ["user", "tool"] {
            let messages = vec![json!({"role":role,"content":"literal <think>"})];
            for (options, suffix, expected) in [
                (json!({}), "", false),
                (
                    json!({"add_generation_prompt":true}),
                    "<|im_start|>assistant\n",
                    false,
                ),
                (
                    json!({"add_generation_prompt":true}),
                    "<|im_start|>assistant\n<think>\n",
                    true,
                ),
            ] {
                let rendered = RenderedPrompt {
                    image_sources: vec![],
                    text: format!("<|im_start|>{role}\nliteral <think><|im_end|>\n{suffix}"),
                    token_ids: vec![],
                };
                assert_eq!(
                    prepare_output(
                        &config,
                        &frontend,
                        &rendered,
                        &messages,
                        options.as_object().unwrap(),
                        false
                    )
                    .unwrap()
                    .content
                    .start
                        == crate::output::tagged::OutputStart::Reasoning,
                    expected
                );
            }
        }
        let messages = vec![json!({"role":"user","content":"<think>"})];
        let rendered = RenderedPrompt {
            image_sources: vec![],
            text: "<|im_start|>user\n<think>".into(),
            token_ids: vec![],
        };
        assert!(
            prepare_output(
                &config,
                &frontend,
                &rendered,
                &messages,
                json!({"continue_final_message":true}).as_object().unwrap(),
                false
            )
            .unwrap()
            .content
            .start
                != crate::output::tagged::OutputStart::Reasoning
        );
    }

    #[test]
    fn output_start_follows_the_open_dsml_assistant_suffix() {
        use crate::output::tagged::OutputStart;
        let frontend = crate::server::output::tests::frontend(&["x"]);
        let mut config = super::super::tests::config();
        let prepared: crate::model::PreparedContract = serde_json::from_str(include_str!(
            "../../tests/support/contracts/deepseek41.json"
        ))
        .unwrap();
        config.contract.frontend = prepared.frontend;
        let tools = json!([{"type":"function","function":{"name":"lookup"}}]);
        for separator in ["", " "] {
            config
                .contract
                .frontend
                .tools
                .as_mut()
                .unwrap()
                .tag_separator = separator.into();
            let open = if separator.is_empty() {
                "<｜DSML｜tool_calls>"
            } else {
                "<｜DSML｜ calls>"
            };
            for (suffix, expected) in [
                (format!("</think>\n\n{open}\n"), OutputStart::ToolBody),
                ("<think>".into(), OutputStart::Reasoning),
                ("</think>".into(), OutputStart::Content),
            ] {
                let rendered = RenderedPrompt {
                    image_sources: vec![],
                    text: format!("<｜User｜>Hello<｜Assistant｜>{suffix}"),
                    token_ids: vec![],
                };
                let options = json!({"add_generation_prompt":true,"tools":tools});
                let output = prepare_output(
                    &config,
                    &frontend,
                    &rendered,
                    &[json!({"role":"user","content":"Hello"})],
                    options.as_object().unwrap(),
                    true,
                )
                .unwrap();
                assert_eq!(output.content.start, expected, "{suffix}");
                let envelope = crate::output::OutputPipeline::grammar_envelope(&output).unwrap();
                assert_eq!(envelope.tool_body_start, expected == OutputStart::ToolBody);
            }
        }
    }
}
