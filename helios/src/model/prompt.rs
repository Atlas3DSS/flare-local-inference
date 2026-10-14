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
        Some(ImageLayoutKind::Mimo) => crate::frontend::images::layouts::mimo::prepare_input(
            frontend, processor, rendered, images,
        )?,
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
    let prefix = frontend.render_chat(&messages, &options).ok()?;
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
    let reasoning = syntax.reasoning.as_ref().is_some_and(|r| {
        prefix.is_some_and(|prefix| {
            prefix
                .rfind(&r.open)
                .is_some_and(|open| prefix.rfind(&r.close).is_none_or(|close| open > close))
        })
    });
    let tools = options
        .get("tools")
        .or_else(|| frontend.template_default("tools"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !tools.is_empty() && syntax.tools.is_none() {
        return Err("tool parsing is not prepared".into());
    }
    Ok(crate::output::OutputConfig {
        syntax: syntax.clone(),
        content: crate::output::tagged::OutputConfig {
            structured,
            reasoning,
            tools,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
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
            assert_eq!(output.content.reasoning, reasoning, "{name}");
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
                    .reasoning,
                    expected
                );
            }
        }
        let messages = vec![json!({"role":"user","content":"<think>"})];
        let rendered = RenderedPrompt {
            text: "<|im_start|>user\n<think>".into(),
            token_ids: vec![],
        };
        assert!(
            !prepare_output(
                &config,
                &frontend,
                &rendered,
                &messages,
                json!({"continue_final_message":true}).as_object().unwrap(),
                false
            )
            .unwrap()
            .content
            .reasoning
        );
    }
}
