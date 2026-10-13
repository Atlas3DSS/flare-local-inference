use super::{PromptData, VOCABULARY};
use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        Frontend, RenderedPrompt,
        images::{ImageProcessor, RgbImage},
    },
    model::PreparedInput,
};

fn image_block(
    height: usize,
    width: usize,
    position: usize,
    feature_offset: usize,
) -> (Vec<u32>, Vec<i32>, usize) {
    let padding = 3 - position % 4;
    let mut types = vec![1u32; padding];
    let mut indices = vec![-1; padding];
    types.push(0);
    indices.push(-1);
    let rows = height.next_multiple_of(2);
    for pair in 0..rows / 2 {
        for column in 0..=width {
            for row in pair * 2..pair * 2 + 2 {
                let feature = row < height && column < width;
                types.push(if row >= height {
                    1
                } else if feature {
                    2
                } else {
                    3
                });
                indices.push(if feature {
                    (feature_offset + row * width + column) as i32
                } else {
                    -1
                });
            }
        }
    }
    let last = rows / 2 * (width + 1) % 2 * 2;
    types.extend(std::iter::repeat_n(1, last));
    indices.extend(std::iter::repeat_n(-1, last));
    types.push(4);
    indices.push(-1);
    (
        types.into_iter().map(|t| VOCABULARY as u32 + t).collect(),
        indices,
        padding,
    )
}

pub(super) fn prepare_input(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: Vec<RgbImage>,
) -> Result<PreparedInput<PromptData>> {
    let placeholder = frontend.encode("<｜deepseek_image｜>", false)?;
    if placeholder.len() != 1 || rendered.token_ids.iter().any(|t| *t as usize >= VOCABULARY) {
        return Err("invalid DeepSeek image placeholder or prompt vocabulary".into());
    }
    if rendered
        .token_ids
        .iter()
        .filter(|t| **t == placeholder[0])
        .count()
        != images.len()
    {
        return Err("image placeholder count does not match input images".into());
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
    let mut image_index = 0;
    let mut feature_offset = 0;
    for token in &rendered.token_ids {
        if *token != placeholder[0] {
            tokens.push(*token);
            data.vision_token_indices.push(-1);
            continue;
        }
        let source = &images[image_index];
        let geometry = &processed.as_ref().unwrap().geometry[image_index];
        let start = tokens.len();
        let (block, indices, padding) = image_block(
            (geometry.grid_thw[1] as usize).div_ceil(3),
            (geometry.grid_thw[2] as usize).div_ceil(3),
            start,
            feature_offset,
        );
        data.spans.push(start + padding..start + block.len());
        tokens.extend(block);
        data.vision_token_indices.extend(indices);
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"flare-rgb-image-v1\0");
        hash.update((source.width() as u64).to_le_bytes());
        hash.update((source.height() as u64).to_le_bytes());
        hash.update(source.pixels());
        conditioning.push(ConditioningEvent {
            position: start as u64,
            identity: hash.finalize().into(),
        });
        feature_offset += geometry.tokens;
        image_index += 1;
    }
    Ok(PreparedInput {
        tokens,
        data,
        conditioning,
        images: processed,
    })
}
