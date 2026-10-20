use crate::model::PromptData;
use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        Frontend, RenderedPrompt,
        images::{ImageProcessor, RgbImage},
    },
    model::PreparedInput,
};

pub(crate) fn prepare_input(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: Vec<RgbImage>,
) -> Result<PreparedInput> {
    let image_ids = frontend.encode("<|image_pad|>", false)?;
    let [image_token] = image_ids.as_slice() else {
        return Err("image placeholder must encode one token".into());
    };
    let image_token = *image_token;
    if rendered
        .token_ids
        .iter()
        .filter(|id| **id == image_token)
        .count()
        != images.len()
    {
        return Err("MiMo image placeholders do not match the tokenizer or input images".into());
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
    let mut token_expansions = Vec::new();
    for (source_index, token) in rendered.token_ids.iter().enumerate() {
        let before = tokens.len();
        if *token != image_token {
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
        tokens.extend(std::iter::repeat_n(image_token, count));
        data.vision_token_indices
            .extend((offset..offset + count).map(|i| i as i32));
        offset += count;
        image += 1;
        token_expansions.push((source_index, tokens.len() - before - 1));
    }
    Ok(PreparedInput {
        tokens,
        token_expansions,
        data,
        conditioning,
        images: processed,
    })
}
