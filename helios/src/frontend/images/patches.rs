//! Area-bounded resizing and spatially grouped RGB patch packing.

use super::*;

#[derive(Clone, Copy)]
pub(super) enum Normalization {
    Symmetric,
    Channel { mean: [f32; 3], std: [f32; 3] },
}

pub(super) struct PatchProcessor {
    pub(super) limits: ImageLimits,
    pub(super) min_pixels: usize,
    pub(super) max_pixels: usize,
    pub(super) normalization: Normalization,
    pub(super) resize: fn(&RgbImage, usize, usize) -> Vec<u8>,
}

impl PatchProcessor {
    pub fn geometry(&self, width: usize, height: usize) -> Result<ImageGeometry> {
        check_input_extent(self.limits, width, height)?;
        if width.max(height) as f64 / width.min(height) as f64 > 200.0 {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image aspect ratio exceeds 200",
            ))));
        }
        let factor = (PATCH * MERGE) as f64;
        let (h, w) = (height as f64, width as f64);
        let mut target_h = (h / factor).round_ties_even() * factor;
        let mut target_w = (w / factor).round_ties_even() * factor;
        if target_h * target_w > self.max_pixels as f64 {
            let beta = (h * w / self.max_pixels as f64).sqrt();
            target_h = ((h / beta / factor).floor() * factor).max(factor);
            target_w = ((w / beta / factor).floor() * factor).max(factor);
        } else if target_h * target_w < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / (h * w)).sqrt();
            target_h = (h * beta / factor).ceil() * factor;
            target_w = (w * beta / factor).ceil() * factor;
        }
        let (height, width) = (target_h as usize, target_w as usize);
        let patches = height / PATCH * (width / PATCH);
        if patches > self.limits.max_patches_per_image {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                format!(
                    "processed image has {patches} patches, limit is {}",
                    self.limits.max_patches_per_image
                ),
            ))));
        }
        Ok(ImageGeometry {
            height,
            width,
            grid_thw: [1, (height / PATCH) as i64, (width / PATCH) as i64],
            patches,
            tokens: patches / (MERGE * MERGE),
        })
    }

    pub fn resized_rgb(&self, image: &RgbImage) -> Result<(ImageGeometry, Vec<u8>)> {
        let geometry = self.geometry(image.width, image.height)?;
        Ok((
            geometry,
            (self.resize)(image, geometry.width, geometry.height),
        ))
    }

    pub fn process(&self, images: &[RgbImage]) -> Result<ProcessedImages> {
        if images.is_empty() || images.len() > self.limits.max_images {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image count is empty or exceeds the configured limit",
            ))));
        }
        let geometry: Vec<_> = images
            .iter()
            .map(|image| self.geometry(image.width, image.height))
            .collect::<Result<_>>()?;
        let total: usize = geometry
            .iter()
            .try_fold(0usize, |sum, image| sum.checked_add(image.patches))
            .ok_or_else(|| {
                Error::frontend(FrontendError::Input(Failure::new("patch count overflow")))
            })?;
        if total > self.limits.max_total_patches {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "images exceed the configured total patch limit",
            ))));
        }
        let mut result = ProcessedImages {
            pixel_values: Vec::with_capacity(total.checked_mul(FEATURES).ok_or(
                Error::frontend(FrontendError::Input(Failure::new(
                    "patch buffer extent overflow",
                ))),
            )?),
            geometry,
            position_ids: Vec::with_capacity(total),
            interpolation_indices: Vec::new(),
            interpolation_weights: Vec::new(),
            cu_seqlens: vec![0],
        };
        let normalized: [Vec<f32>; 3] = std::array::from_fn(|channel| {
            (0..256)
                .map(|value| match self.normalization {
                    Normalization::Symmetric => (value as f32 - 127.5) / 127.5,
                    Normalization::Channel { mean, std } => {
                        (value as f32 * (1.0 / 255.0) - mean[channel]) / std[channel]
                    }
                })
                .collect()
        });
        for (image, geometry) in images.iter().zip(&result.geometry) {
            let pixels = (self.resize)(image, geometry.width, geometry.height);
            let grid_h = geometry.height / PATCH;
            let grid_w = geometry.width / PATCH;
            for by in 0..grid_h / MERGE {
                for bx in 0..grid_w / MERGE {
                    for dy in 0..MERGE {
                        for dx in 0..MERGE {
                            let (row, col) = (by * MERGE + dy, bx * MERGE + dx);
                            result.position_ids.push([row as i64, col as i64]);
                            for channel in 0..3 {
                                for _temporal in 0..2 {
                                    for y in 0..PATCH {
                                        for x in 0..PATCH {
                                            let byte = pixels[((row * PATCH + y) * geometry.width
                                                + col * PATCH
                                                + x)
                                                * 3
                                                + channel];
                                            result
                                                .pixel_values
                                                .push(normalized[channel][byte as usize]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            result
                .cu_seqlens
                .push(result.cu_seqlens.last().unwrap() + geometry.patches as i32);
        }
        Ok(result)
    }
}
