use super::*;

pub(super) struct QwenProcessor {
    pub(super) limits: ImageLimits,
    pub(super) min_pixels: usize,
    pub(super) max_pixels: usize,
    pub(super) position_side: usize,
    pub(super) image_token_id: u32,
    pub(super) video_token_id: u32,
}

impl QwenProcessor {
    pub(super) fn load(config: &Value, model: &Value, limits: ImageLimits) -> Result<Self> {
        for (key, expected) in [
            ("patch_size", 16),
            ("temporal_patch_size", 2),
            ("merge_size", 2),
        ] {
            if config[key].as_u64() != Some(expected) {
                return Err(Error::frontend(FrontendError::Asset(Failure::new(
                    format!("unsupported image processor {key}"),
                ))));
            }
        }
        if config["processor_class"] != "Qwen3VLProcessor"
            || config["image_processor_type"] != "Qwen2VLImageProcessorFast"
            || config["image_mean"] != serde_json::json!([0.5, 0.5, 0.5])
            || config["image_std"] != serde_json::json!([0.5, 0.5, 0.5])
            || config.get("resample").is_some_and(|value| value != 3)
            || config
                .get("rescale_factor")
                .is_some_and(|value| value.as_f64() != Some(1.0 / 255.0))
            || ["do_resize", "do_rescale", "do_normalize", "do_convert_rgb"]
                .iter()
                .any(|key| config.get(*key).is_some_and(|value| value != true))
        {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                "unsupported image processor RGB8/bicubic/normalization profile",
            ))));
        }
        let vision = &model["vision_config"];
        if model["model_type"] != "qwen4_exp"
            || vision["patch_size"] != 16
            || vision["temporal_patch_size"] != 2
            || vision["spatial_merge_size"] != 2
            || vision["in_channels"] != 3
            || vision["num_position_embeddings"] != 2304
        {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                "unsupported Qwen4Exp vision configuration",
            ))));
        }
        let min_pixels = config["size"]["shortest_edge"].as_u64();
        let max_pixels = config["size"]["longest_edge"].as_u64();
        if min_pixels != Some(65_536) || max_pixels != Some(16_777_216) {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                "unsupported Qwen4Exp resize area limits",
            ))));
        }
        let token = |name: &str| -> Result<u32> {
            model[name]
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| {
                    Error::frontend(FrontendError::Asset(Failure::new(format!(
                        "invalid model {name}"
                    ))))
                })
        };
        Ok(Self {
            limits,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            position_side: 48,
            image_token_id: token("image_token_id")?,
            video_token_id: token("video_token_id")?,
        })
    }

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
            resize::resize(image, geometry.width, geometry.height),
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
            interpolation_indices: Vec::with_capacity(total),
            interpolation_weights: Vec::with_capacity(total),
            cu_seqlens: vec![0],
        };
        let normalized: Vec<_> = (0..256)
            .map(|value| (value as f32 - 127.5) / 127.5)
            .collect();
        for (image, geometry) in images.iter().zip(&result.geometry) {
            let pixels = resize::resize(image, geometry.width, geometry.height);
            let grid_h = geometry.height / PATCH;
            let grid_w = geometry.width / PATCH;
            for by in 0..grid_h / MERGE {
                for bx in 0..grid_w / MERGE {
                    for dy in 0..MERGE {
                        for dx in 0..MERGE {
                            let (row, col) = (by * MERGE + dy, bx * MERGE + dx);
                            result.position_ids.push([row as i64, col as i64]);
                            let (yi, yw) = interpolation_axis(row, grid_h, self.position_side);
                            let (xi, xw) = interpolation_axis(col, grid_w, self.position_side);
                            result.interpolation_indices.push([
                                (yi[0] * self.position_side + xi[0]) as i64,
                                (yi[0] * self.position_side + xi[1]) as i64,
                                (yi[1] * self.position_side + xi[0]) as i64,
                                (yi[1] * self.position_side + xi[1]) as i64,
                            ]);
                            result.interpolation_weights.push([
                                yw[0] * xw[0],
                                yw[0] * xw[1],
                                yw[1] * xw[0],
                                yw[1] * xw[1],
                            ]);
                            for channel in 0..3 {
                                for _temporal in 0..2 {
                                    for y in 0..PATCH {
                                        for x in 0..PATCH {
                                            let byte = pixels[((row * PATCH + y) * geometry.width
                                                + col * PATCH
                                                + x)
                                                * 3
                                                + channel];
                                            result.pixel_values.push(normalized[byte as usize]);
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

    /// Expands the image placeholders in already rendered chat text, in image order.
    /// Inputs contain no padding; the scheduler adds padding outside logical lengths.
    pub fn expand_prompt(
        &self,
        frontend: &Frontend,
        text: &str,
        images: &ProcessedImages,
    ) -> Result<ImagePrompt> {
        if frontend.encode(IMAGE_TOKEN, false)? != [self.image_token_id] {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image token ID does not match tokenizer",
            ))));
        }
        if text.matches(IMAGE_TOKEN).count() != images.geometry.len() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image placeholder count does not match prepared images",
            ))));
        }
        let mut expanded = String::with_capacity(text.len());
        let mut parts = text.split(IMAGE_TOKEN);
        expanded.push_str(parts.next().unwrap_or_default());
        for (part, geometry) in parts.zip(&images.geometry) {
            for _ in 0..geometry.tokens {
                expanded.push_str(IMAGE_TOKEN);
            }
            expanded.push_str(part);
        }
        let token_ids = frontend.encode(&expanded, false)?;
        if token_ids.contains(&self.video_token_id) {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "video inputs are not supported by the image processor",
            ))));
        }
        let mm_token_type_ids: Vec<i32> = token_ids
            .iter()
            .map(|id| i32::from(*id == self.image_token_id))
            .collect();
        let mut positions: [Vec<i64>; 4] =
            std::array::from_fn(|_| Vec::with_capacity(token_ids.len()));
        positions[0].extend((0..token_ids.len()).map(|index| index as i64));
        let (mut offset, mut current, mut image_index) = (0, 0i64, 0);
        while offset < token_ids.len() {
            let modality = mm_token_type_ids[offset];
            let end = offset
                + mm_token_type_ids[offset..]
                    .iter()
                    .take_while(|value| **value == modality)
                    .count();
            if modality == 0 {
                for row in &mut positions[1..] {
                    row.extend((0..end - offset).map(|index| current + index as i64));
                }
                current += (end - offset) as i64;
            } else {
                let geometry = images.geometry.get(image_index).ok_or_else(|| {
                    Error::frontend(FrontendError::Input(Failure::new(
                        "unexpected image token group",
                    )))
                })?;
                if end - offset != geometry.tokens {
                    return Err(Error::frontend(FrontendError::Input(Failure::new(
                        "image token group does not match image grid",
                    ))));
                }
                let (height, width) = (
                    geometry.grid_thw[1] / MERGE as i64,
                    geometry.grid_thw[2] / MERGE as i64,
                );
                for y in 0..height {
                    for x in 0..width {
                        positions[1].push(current);
                        positions[2].push(current + y);
                        positions[3].push(current + x);
                    }
                }
                current += height.max(width);
                image_index += 1;
            }
            offset = end;
        }
        if image_index != images.geometry.len() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "prepared image was not consumed by prompt",
            ))));
        }
        Ok(ImagePrompt {
            text: expanded,
            rope_delta: current - token_ids.len() as i64,
            token_ids,
            mm_token_type_ids,
            position_ids: positions,
        })
    }
}

fn interpolation_axis(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let source = index as f32 * (side - 1) as f32 / (size - 1).max(1) as f32;
    let floor = source.floor();
    let distance = source - floor;
    (
        [floor as usize, (floor as usize + 1).min(side - 1)],
        [1.0 - distance, 1.0 - (distance - 1.0).abs()],
    )
}
