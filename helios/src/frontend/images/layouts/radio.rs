use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        Frontend, RenderedPrompt,
        images::{ImageProcessor, RgbImage},
    },
    model::{PreparedInput, PromptData},
};

pub(crate) fn prepare_input(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: Vec<RgbImage>,
) -> Result<PreparedInput> {
    let placeholder = frontend.encode("<image>", false)?;
    let [placeholder] = placeholder.as_slice() else {
        return Err("RADIO image placeholder must encode one token".into());
    };
    if rendered
        .token_ids
        .iter()
        .filter(|id| *id == placeholder)
        .count()
        != images.len()
    {
        return Err("RADIO image placeholders do not match input images".into());
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
    let prefix = frontend.encode("<img>", false)?;
    let suffix = frontend.encode("</img>", false)?;
    let mut tokens = Vec::new();
    let mut data = PromptData::default();
    let mut conditioning = Vec::new();
    let (mut image, mut offset) = (0, 0);
    for token in &rendered.token_ids {
        if token != placeholder {
            tokens.push(*token);
            data.vision_token_indices.push(-1);
            continue;
        }
        tokens.extend(&prefix);
        data.vision_token_indices
            .extend(std::iter::repeat_n(-1, prefix.len()));
        let source = &images[image];
        let count = processed.as_ref().unwrap().geometry[image].tokens;
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"flare-radio-image-v1\0");
        hash.update((source.width() as u64).to_le_bytes());
        hash.update((source.height() as u64).to_le_bytes());
        hash.update(source.pixels());
        conditioning.push(ConditioningEvent {
            position: tokens.len() as u64,
            identity: hash.finalize().into(),
        });
        tokens.extend(std::iter::repeat_n(*placeholder, count));
        data.vision_token_indices
            .extend((offset..offset + count).map(|i| i as i32));
        tokens.extend(&suffix);
        data.vision_token_indices
            .extend(std::iter::repeat_n(-1, suffix.len()));
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
