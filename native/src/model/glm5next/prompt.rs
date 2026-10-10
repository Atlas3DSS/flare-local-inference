use crate::{
    Result,
    frontend::{Frontend, RenderedPrompt, images::RgbImage},
    model::PreparedInput,
};

pub(super) fn prepare_input(
    _: &Frontend,
    rendered: &RenderedPrompt,
    images: Vec<RgbImage>,
) -> Result<PreparedInput<()>> {
    if !images.is_empty()
        || rendered
            .token_ids
            .iter()
            .any(|id| matches!(id, 154830..=154835 | 154854 | 154855))
    {
        return Err("prepared GLM catalog supports text inputs only".into());
    }
    Ok(PreparedInput {
        tokens: rendered.token_ids.clone(),
        data: (),
        conditioning: Vec::new(),
        images: None,
    })
}

pub(super) fn system_boundary(
    frontend: &Frontend,
    prompt_tokens: &[u32],
    messages: &[serde_json::Value],
    options: &serde_json::Map<String, serde_json::Value>,
) -> Option<usize> {
    use serde_json::{Value, json};
    let count = messages
        .iter()
        .take_while(|m| m.get("role").and_then(Value::as_str) == Some("system"))
        .count();
    let mut messages = messages[..count].to_vec();
    messages.push(json!({"role":"user", "content":""}));
    let mut options = options.clone();
    options.insert("add_generation_prompt".into(), Value::Bool(false));
    options.insert("continue_final_message".into(), Value::Bool(true));
    let prefix = frontend.render_chat(&messages, &options).ok()?;
    if !prefix.text.starts_with("[gMASK]<sop><|system|>") || !prefix.text.ends_with("<|user|>") {
        return None;
    }
    let boundary = prefix.token_ids.len();
    (0 < boundary && boundary < prompt_tokens.len() && prompt_tokens.starts_with(&prefix.token_ids))
        .then_some(boundary)
}
