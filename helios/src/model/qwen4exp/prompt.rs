use super::PromptData;
use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        Frontend, RenderedPrompt,
        images::{ImageProcessor, RgbImage},
    },
    model::PreparedInput,
};
pub(super) fn prepare_input(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: Vec<RgbImage>,
) -> Result<PreparedInput<PromptData>> {
    let (tokens, positions, vision_token_indices, conditioning, images) = if images.is_empty() {
        if rendered
            .token_ids
            .iter()
            .any(|id| *id == 248056 || *id == 248057)
        {
            return Err("image/video placeholders require matching supported image inputs".into());
        }
        let positions = (0..rendered.token_ids.len())
            .map(|position| [position as i64; 3])
            .collect();
        let indices = vec![-1; rendered.token_ids.len()];
        (
            rendered.token_ids.clone(),
            positions,
            indices,
            Vec::new(),
            None,
        )
    } else {
        let processor = processor.ok_or("image processing is unavailable")?;
        let processed = processor.process(&images)?;
        let expanded = processor.expand_prompt(frontend, &rendered.text, &processed)?;
        let positions = (0..expanded.token_ids.len())
            .map(|position| {
                [
                    expanded.position_ids[1][position],
                    expanded.position_ids[2][position],
                    expanded.position_ids[3][position],
                ]
            })
            .collect();
        let mut conditioning = Vec::with_capacity(images.len());
        let mut image = 0;
        use sha2::{Digest, Sha256};
        for (position, kind) in expanded.mm_token_type_ids.iter().enumerate() {
            if *kind == 1 && (position == 0 || expanded.mm_token_type_ids[position - 1] != 1) {
                let source = &images[image];
                let mut hash = Sha256::new();
                hash.update(b"flare-rgb-image-v1\0");
                hash.update((source.width() as u64).to_le_bytes());
                hash.update((source.height() as u64).to_le_bytes());
                hash.update(source.pixels());
                conditioning.push(ConditioningEvent {
                    position: position as u64,
                    identity: hash.finalize().into(),
                });
                image += 1;
            }
        }
        if image != images.len() {
            return Err("expanded image spans do not match input image count".into());
        }
        let mut index = 0;
        let indices = expanded
            .mm_token_type_ids
            .iter()
            .map(|kind| {
                if *kind == 1 {
                    let value = index;
                    index += 1;
                    value
                } else {
                    -1
                }
            })
            .collect();
        (
            expanded.token_ids,
            positions,
            indices,
            conditioning,
            Some(processed),
        )
    };
    Ok(PreparedInput {
        tokens,
        data: PromptData {
            positions,
            vision_token_indices,
        },
        conditioning,
        images,
    })
}

/// Exact token prefix through the instruction segment and empty user header.
pub(super) fn system_boundary(
    frontend: &Frontend,
    prompt_tokens: &[u32],
    messages: &[serde_json::Value],
    options: &serde_json::Map<String, serde_json::Value>,
) -> Option<usize> {
    use serde_json::{Value, json};
    let count = messages
        .iter()
        .take_while(|message| {
            matches!(
                message.get("role").and_then(Value::as_str),
                Some("system" | "developer")
            )
        })
        .count();
    let mut prefix_messages = messages[..count].to_vec();
    prefix_messages.push(json!({"role": "user", "content": ""}));
    let mut prefix_options = options.clone();
    prefix_options.insert("add_generation_prompt".into(), Value::Bool(false));
    prefix_options.insert("continue_final_message".into(), Value::Bool(true));
    let prefix = frontend
        .render_chat(&prefix_messages, &prefix_options)
        .ok()?;
    // Qwen can synthesize instructions from tools or reasoning defaults.
    if !prefix.text.starts_with("<|im_start|>system\n") || !prefix.text.contains("<|im_end|>") {
        return None;
    }
    let boundary = prefix.token_ids.len();
    (0 < boundary && boundary < prompt_tokens.len() && prompt_tokens.starts_with(&prefix.token_ids))
        .then_some(boundary)
}
