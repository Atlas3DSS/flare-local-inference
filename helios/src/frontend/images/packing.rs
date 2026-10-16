//! Image-layout tensors consumed through prepared semantic bindings.
use super::ProcessedImages;
use crate::{
    Result,
    execution::Dtype,
    model::{ModelConfig, contract::ImageLayoutKind, inputs::UploadBatch},
    scheduler::RequestHandle,
};

pub(crate) fn tensor_layout(
    kind: ImageLayoutKind,
    name: &str,
    patches: usize,
) -> Result<(Dtype, usize)> {
    use Dtype::{Float32, Int32, Int64};
    let layout = match (kind, name) {
        (ImageLayoutKind::Qwen | ImageLayoutKind::Mimo, "vision_pixel_values") => {
            (Float32, patches * 1536)
        }
        (ImageLayoutKind::Deepseek | ImageLayoutKind::Moonvit, "vision_pixel_values") => {
            (Float32, patches * 588)
        }
        (_, "vision_position_ids") => (Int64, patches * 2),
        (_, "vision_cu_seqlens") => (Int32, 2),
        (_, "vision_num_patches" | "vision_feature_offset") => (Int32, 1),
        (
            ImageLayoutKind::Mimo | ImageLayoutKind::Deepseek | ImageLayoutKind::Moonvit,
            "vision_num_features",
        ) => (Int32, 1),
        (ImageLayoutKind::Qwen | ImageLayoutKind::Moonvit, "vision_interpolation_indices") => {
            (Int64, patches * 4)
        }
        (ImageLayoutKind::Qwen | ImageLayoutKind::Moonvit, "vision_interpolation_weights") => {
            (Float32, patches * 4)
        }
        (ImageLayoutKind::Mimo, "vision_column_indices" | "vision_row_indices") => {
            (Int32, patches / 4)
        }
        (ImageLayoutKind::Deepseek | ImageLayoutKind::Moonvit, "vision_merge_indices") => {
            (Int32, patches)
        }
        _ => return Err(format!("image layout has no tensor {name}").into()),
    };
    Ok(layout)
}

pub(crate) fn prepare(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
) -> Result<UploadBatch> {
    match config
        .contract
        .frontend
        .image_layout
        .ok_or("image layout is not prepared")?
    {
        ImageLayoutKind::Qwen => qwen(config, handle, images, image),
        ImageLayoutKind::Mimo => mimo(config, handle, images, image),
        ImageLayoutKind::Deepseek => deepseek(config, handle, images, image),
        ImageLayoutKind::Moonvit => moonvit(config, handle, images, image),
    }
}

fn qwen(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
) -> Result<UploadBatch> {
    let pixel_width = if config.contract.frontend.image_layout == Some(ImageLayoutKind::Moonvit) {
        588
    } else {
        1536
    };
    let mut staging = UploadBatch::default();
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let start: usize = images.geometry[..image]
        .iter()
        .map(|geometry| geometry.patches)
        .sum();
    let end = start + geometry.patches;
    if geometry.patches == 0
        || geometry.patches
            > *config
                .contract
                .vision
                .patch_capacities
                .last()
                .ok_or("missing vision capacities")?
        || end / config.contract.vision.patches_per_feature
            > config.contract.vision.features_per_request
        || handle.slot as usize >= config.limits.max_active
        || handle.generation == 0
        || handle.generation > i64::MAX as u64
    {
        return Err("invalid vision graph input extent".into());
    }
    staging.put(
        "vision_pixel_values",
        0,
        images
            .pixel_values
            .get(start * pixel_width..end * pixel_width)
            .ok_or("truncated vision pixels")?
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes),
    );
    staging.i64(
        "vision_interpolation_indices",
        0,
        images
            .interpolation_indices
            .get(start..end)
            .ok_or("truncated vision interpolation indices")?
            .iter()
            .flatten()
            .copied(),
    );
    staging.put(
        "vision_interpolation_weights",
        0,
        images
            .interpolation_weights
            .get(start..end)
            .ok_or("truncated vision interpolation weights")?
            .iter()
            .flatten()
            .copied()
            .flat_map(f32::to_le_bytes),
    );
    staging.i64(
        "vision_position_ids",
        0,
        images
            .position_ids
            .get(start..end)
            .ok_or("truncated vision positions")?
            .iter()
            .flatten()
            .copied(),
    );
    staging.i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
    staging.i32("vision_num_patches", 0, [geometry.patches as i32]);
    staging.i32(
        "vision_feature_offset",
        0,
        [(start / config.contract.vision.patches_per_feature) as i32],
    );
    Ok(staging)
}

fn mimo(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
) -> Result<UploadBatch> {
    let mut staging = UploadBatch::default();
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let start: usize = images.geometry[..image].iter().map(|g| g.patches).sum();
    let end = start + geometry.patches;
    let h = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid vision height")?;
    let w = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid vision width")?;
    if h == 0
        || w == 0
        || h % 2 != 0
        || w % 2 != 0
        || geometry.grid_thw[0] != 1
        || geometry.patches > 16384
        || h.checked_mul(w) != Some(geometry.patches)
        || geometry.tokens != geometry.patches / 4
        || end / 4 > config.contract.vision.features_per_request
        || handle.slot as usize >= config.limits.max_active
        || handle.generation == 0
        || handle.generation > i64::MAX as u64
    {
        return Err("invalid MiMo vision graph extent".into());
    }
    staging.put(
        "vision_pixel_values",
        0,
        images
            .pixel_values
            .get(start * 1536..end * 1536)
            .ok_or("truncated vision pixels")?
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes),
    );
    staging.i64(
        "vision_position_ids",
        0,
        images
            .position_ids
            .get(start..end)
            .ok_or("truncated vision positions")?
            .iter()
            .flatten()
            .copied(),
    );
    let (columns, rows) = column_permutation(h / 2, w / 2);
    staging.i32("vision_column_indices", 0, columns);
    staging.i32("vision_row_indices", 0, rows);
    staging.i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
    staging.i32("vision_num_patches", 0, [geometry.patches as i32]);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    staging.i32("vision_feature_offset", 0, [(start / 4) as i32]);
    Ok(staging)
}
fn column_permutation(height: usize, width: usize) -> (Vec<i32>, Vec<i32>) {
    let columns: Vec<_> = (0..width)
        .flat_map(|x| (0..height).map(move |y| (y * width + x) as i32))
        .collect();
    let mut rows = vec![0; columns.len()];
    for (index, column) in columns.iter().enumerate() {
        rows[*column as usize] = index as i32;
    }
    (columns, rows)
}

fn deepseek(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
) -> Result<UploadBatch> {
    let mut staging = UploadBatch::default();
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let start: usize = images.geometry[..image].iter().map(|g| g.patches).sum();
    let end = start + geometry.patches;
    let h = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid vision height")?;
    let w = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid vision width")?;
    let actual = h
        .checked_mul(w)
        .filter(|n| *n > 0 && *n <= 3456)
        .ok_or("invalid vision grid")?;
    if geometry.patches > 3456
        || geometry.patches != h.div_ceil(3) * w.div_ceil(3) * 9
        || geometry.tokens != geometry.patches / 9
        || end / 9 > config.contract.vision.features_per_request
        || handle.slot as usize >= config.limits.max_active
        || handle.generation == 0
        || handle.generation > i64::MAX as u64
    {
        return Err("invalid DeepSeek vision graph extent".into());
    }
    staging.put(
        "vision_pixel_values",
        0,
        images
            .pixel_values
            .get(start * 588..end * 588)
            .ok_or("truncated vision pixels")?
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes),
    );
    staging.i64(
        "vision_position_ids",
        0,
        images
            .position_ids
            .get(start..end)
            .ok_or("truncated vision positions")?
            .iter()
            .flatten()
            .copied(),
    );
    let mut indices = Vec::with_capacity(geometry.patches);
    for row in 0..h.div_ceil(3) {
        for col in 0..w.div_ceil(3) {
            for dy in 0..3 {
                for dx in 0..3 {
                    let (y, x) = (row * 3 + dy, col * 3 + dx);
                    indices.push(if y < h && x < w {
                        (y * w + x) as i32
                    } else {
                        -1
                    });
                }
            }
        }
    }
    staging.i32("vision_merge_indices", 0, indices);
    staging.i32("vision_cu_seqlens", 0, [0, actual as i32]);
    staging.i32("vision_num_patches", 0, [actual as i32]);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    staging.i32("vision_feature_offset", 0, [(start / 9) as i32]);
    Ok(staging)
}

fn moonvit(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
) -> Result<UploadBatch> {
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let h = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid vision height")?;
    let w = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid vision width")?;
    if h == 0
        || w == 0
        || h > 512
        || w > 512
        || h % 2 != 0
        || w % 2 != 0
        || geometry.grid_thw[0] != 1
        || h.checked_mul(w) != Some(geometry.patches)
        || geometry.tokens != geometry.patches / 4
    {
        return Err("invalid MoonViT vision grid".into());
    }
    let mut staging = qwen(config, handle, images, image)?;
    let mut indices = Vec::with_capacity(geometry.patches);
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    indices.push(((y + dy) * w + x + dx) as i32);
                }
            }
        }
    }
    staging.i32("vision_merge_indices", 0, indices);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    Ok(staging)
}
