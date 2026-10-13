use super::{IMAGE_TOKEN, PromptData};
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
    if frontend.encode("<|image_pad|>", false)? != [IMAGE_TOKEN]
        || rendered
            .token_ids
            .iter()
            .filter(|id| **id == IMAGE_TOKEN)
            .count()
            != images.len()
    {
        return Err("MiMo image placeholders do not match the tokenizer or input images".into());
    }
    if rendered.token_ids.contains(&151656) {
        return Err("MiMo video inputs are not prepared".into());
    }
    let processed = if images.is_empty() {
        None
    } else {
        Some(
            processor
                .ok_or("image processing is unavailable")?
                .process(&images)?,
        )
    };
    let mut tokens = Vec::new();
    let mut data = PromptData::default();
    let mut conditioning = Vec::new();
    let (mut image, mut offset) = (0, 0);
    for token in &rendered.token_ids {
        if *token != IMAGE_TOKEN {
            tokens.push(*token);
            data.vision_token_indices.push(-1);
            continue;
        }
        let source = &images[image];
        let count = processed.as_ref().unwrap().geometry[image].tokens;
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"flare-rgb-image-v1\0");
        hash.update((source.width() as u64).to_le_bytes());
        hash.update((source.height() as u64).to_le_bytes());
        hash.update(source.pixels());
        conditioning.push(ConditioningEvent {
            position: tokens.len() as u64,
            identity: hash.finalize().into(),
        });
        tokens.extend(std::iter::repeat_n(IMAGE_TOKEN, count));
        data.vision_token_indices
            .extend((offset..offset + count).map(|i| i as i32));
        offset += count;
        image += 1;
    }
    Ok(PreparedInput {
        tokens,
        data,
        conditioning,
        images: processed,
    })
}
