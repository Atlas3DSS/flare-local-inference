//! Qwen4Exp RGB8 image preparation and text/vision position metadata.

#[cfg(test)]
mod decoding_tests;
mod orientation;
mod resize;

use std::io::Cursor;

use serde_json::Value;

use super::{Asset, Frontend};
use crate::Result;

const PATCH: usize = 16;
const MERGE: usize = 2;
const FEATURES: usize = 3 * 2 * PATCH * PATCH;
const IMAGE_TOKEN: &str = "<|image_pad|>";

#[derive(Clone, Copy)]
pub struct ImageLimits {
    pub max_input_pixels: usize,
    pub max_encoded_bytes: usize,
    pub max_images: usize,
    pub max_patches_per_image: usize,
    pub max_total_patches: usize,
}

impl Default for ImageLimits {
    fn default() -> Self {
        Self {
            max_input_pixels: 64 * 1024 * 1024,
            max_encoded_bytes: 32 * 1024 * 1024,
            max_images: 4,
            max_patches_per_image: 16_384,
            max_total_patches: 16_384,
        }
    }
}

pub struct RgbImage {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl RgbImage {
    /// Pixels are row-major RGB8, with no orientation or color-profile conversion.
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|size| size.checked_mul(3));
        if width == 0 || height == 0 || expected != Some(pixels.len()) {
            return Err("RGB8 dimensions do not match pixel extent".into());
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageGeometry {
    pub height: usize,
    pub width: usize,
    pub grid_thw: [i64; 3],
    pub patches: usize,
    pub tokens: usize,
}

/// CPU metadata in the packed patch order consumed by the vision graph.
pub struct ProcessedImages {
    pub pixel_values: Vec<f32>,
    pub geometry: Vec<ImageGeometry>,
    pub position_ids: Vec<[i64; 2]>,
    pub interpolation_indices: Vec<[i64; 4]>,
    pub interpolation_weights: Vec<[f32; 4]>,
    pub cu_seqlens: Vec<i32>,
}

pub struct ImagePrompt {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub mm_token_type_ids: Vec<i32>,
    /// Text, temporal, height, and width position rows, each of length token_ids.len().
    pub position_ids: [Vec<i64>; 4],
    pub rope_delta: i64,
}

pub struct ImageProcessor {
    limits: ImageLimits,
    min_pixels: usize,
    max_pixels: usize,
    position_side: usize,
    image_token_id: u32,
    video_token_id: u32,
}

impl ImageProcessor {
    /// Loads the explicit Qwen4Exp processor profile; incompatible profiles are rejected.
    pub fn load(preprocessor: Asset, model_config: Asset, limits: ImageLimits) -> Result<Self> {
        if [
            limits.max_input_pixels,
            limits.max_encoded_bytes,
            limits.max_images,
            limits.max_patches_per_image,
            limits.max_total_patches,
        ]
        .contains(&0)
            || limits.max_total_patches > i32::MAX as usize
        {
            return Err("image limits must be positive and total patches must fit int32".into());
        }
        let config: Value =
            serde_json::from_slice(&preprocessor.read()?).map_err(|e| e.to_string())?;
        let model: Value =
            serde_json::from_slice(&model_config.read()?).map_err(|e| e.to_string())?;
        for (key, expected) in [
            ("patch_size", 16),
            ("temporal_patch_size", 2),
            ("merge_size", 2),
        ] {
            if config[key].as_u64() != Some(expected) {
                return Err(format!("unsupported image processor {key}"));
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
            return Err("unsupported image processor RGB8/bicubic/normalization profile".into());
        }
        let vision = &model["vision_config"];
        if model["model_type"] != "qwen4_exp"
            || vision["patch_size"] != 16
            || vision["temporal_patch_size"] != 2
            || vision["spatial_merge_size"] != 2
            || vision["in_channels"] != 3
            || vision["num_position_embeddings"] != 2304
        {
            return Err("unsupported Qwen4Exp vision configuration".into());
        }
        let min_pixels = config["size"]["shortest_edge"].as_u64();
        let max_pixels = config["size"]["longest_edge"].as_u64();
        if min_pixels != Some(65_536) || max_pixels != Some(16_777_216) {
            return Err("unsupported Qwen4Exp resize area limits".into());
        }
        let token = |name: &str| -> Result<u32> {
            model[name]
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| format!("invalid model {name}"))
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

    fn check_input_extent(&self, width: usize, height: usize) -> Result<()> {
        if width == 0
            || height == 0
            || width
                .checked_mul(height)
                .is_none_or(|pixels| pixels > self.limits.max_input_pixels)
        {
            return Err("image exceeds the configured input pixel limit or has zero extent".into());
        }
        Ok(())
    }

    pub fn max_encoded_bytes(&self) -> usize {
        self.limits.max_encoded_bytes
    }

    /// Decodes PNG to RGB8 and applies EXIF orientation before resizing.
    /// Alpha is discarded; animated PNG and 16-bit samples are rejected.
    pub fn decode_png(&self, bytes: &[u8]) -> Result<RgbImage> {
        if bytes.len() > self.limits.max_encoded_bytes {
            return Err("encoded image exceeds the configured byte limit".into());
        }
        let mut decoder = png::Decoder::new(Cursor::new(bytes));
        decoder.set_transformations(png::Transformations::EXPAND);
        decoder.set_limits(png::Limits {
            bytes: self.limits.max_input_pixels.saturating_mul(8),
        });
        let mut reader = decoder
            .read_info()
            .map_err(|e| format!("invalid PNG: {e}"))?;
        let info = reader.info();
        self.check_input_extent(info.width as usize, info.height as usize)?;
        if info.bit_depth == png::BitDepth::Sixteen || info.animation_control.is_some() {
            return Err("PNG requires non-animated, at most 8-bit samples".into());
        }
        let size = reader
            .output_buffer_size()
            .ok_or("PNG decoded extent overflow")?;
        let mut decoded = vec![0; size];
        let output = reader
            .next_frame(&mut decoded)
            .map_err(|e| format!("invalid PNG: {e}"))?;
        reader.finish().map_err(|e| format!("invalid PNG: {e}"))?;
        let orientation = reader
            .info()
            .exif_metadata
            .as_deref()
            .map(orientation::exif_orientation)
            .transpose()?
            .unwrap_or(1);
        let mut rgb = Vec::with_capacity(output.width as usize * output.height as usize * 3);
        for pixel in decoded[..output.buffer_size()].chunks_exact(output.color_type.samples()) {
            match output.color_type {
                png::ColorType::Rgb | png::ColorType::Rgba => rgb.extend_from_slice(&pixel[..3]),
                png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha => {
                    rgb.extend_from_slice(&[pixel[0]; 3])
                }
                png::ColorType::Indexed => return Err("PNG palette was not expanded".into()),
            }
        }
        Ok(orientation::orient(
            RgbImage::new(output.width as usize, output.height as usize, rgb)?,
            orientation,
        ))
    }

    /// Decodes JPEG with accurate IDCT and smooth chroma upsampling, then applies EXIF.
    /// Color profiles are not applied, matching Pillow's RGB conversion.
    pub fn decode_jpeg(&self, bytes: &[u8]) -> Result<RgbImage> {
        if bytes.len() > self.limits.max_encoded_bytes {
            return Err("encoded image exceeds the configured byte limit".into());
        }
        let orientation = orientation::jpeg_orientation(bytes)?;
        let error = |e| format!("invalid JPEG: {e}");
        let mut decoder = turbojpeg::Decompressor::new().map_err(error)?;
        decoder.set_scan_limit(100).map_err(error)?;
        decoder.set_fast_upsample(false).map_err(error)?;
        let header = decoder.read_header(bytes).map_err(error)?;
        self.check_input_extent(header.width, header.height)?;
        if header.is_lossless || header.is_arithmetic {
            return Err("JPEG requires baseline or progressive Huffman coding".into());
        }
        let cmyk = matches!(
            header.colorspace,
            turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK
        );
        let channels = if cmyk { 4 } else { 3 };
        let length = header
            .width
            .checked_mul(header.height)
            .and_then(|n| n.checked_mul(channels))
            .ok_or("JPEG decoded extent overflow")?;
        let mut decoded = turbojpeg::Image {
            pixels: vec![0; length],
            width: header.width,
            height: header.height,
            pitch: header.width * channels,
            format: if cmyk {
                turbojpeg::PixelFormat::CMYK
            } else {
                turbojpeg::PixelFormat::RGB
            },
        };
        decoder
            .decompress(bytes, decoded.as_deref_mut())
            .map_err(error)?;
        let rgb = if cmyk {
            // JPEG CMYK samples are inverted; this matches Pillow's CMYK-to-RGB conversion.
            decoded
                .pixels
                .chunks_exact(4)
                .flat_map(|pixel| {
                    pixel[..3].iter().map(|channel| {
                        ((u32::from(*channel) * u32::from(pixel[3]) + 127) / 255) as u8
                    })
                })
                .collect()
        } else {
            decoded.pixels
        };
        Ok(orientation::orient(
            RgbImage::new(header.width, header.height, rgb)?,
            orientation,
        ))
    }

    pub fn geometry(&self, width: usize, height: usize) -> Result<ImageGeometry> {
        self.check_input_extent(width, height)?;
        if width.max(height) as f64 / width.min(height) as f64 > 200.0 {
            return Err("image aspect ratio exceeds 200".into());
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
            return Err(format!(
                "processed image has {patches} patches, limit is {}",
                self.limits.max_patches_per_image
            ));
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
            return Err("image count is empty or exceeds the configured limit".into());
        }
        let geometry: Vec<_> = images
            .iter()
            .map(|image| self.geometry(image.width, image.height))
            .collect::<Result<_>>()?;
        let total: usize = geometry
            .iter()
            .try_fold(0usize, |sum, image| sum.checked_add(image.patches))
            .ok_or("patch count overflow")?;
        if total > self.limits.max_total_patches {
            return Err("images exceed the configured total patch limit".into());
        }
        let mut result = ProcessedImages {
            pixel_values: Vec::with_capacity(
                total
                    .checked_mul(FEATURES)
                    .ok_or("patch buffer extent overflow")?,
            ),
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
            return Err("image token ID does not match tokenizer".into());
        }
        if text.matches(IMAGE_TOKEN).count() != images.geometry.len() {
            return Err("image placeholder count does not match prepared images".into());
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
            return Err("video inputs are not supported by the image processor".into());
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
                let geometry = images
                    .geometry
                    .get(image_index)
                    .ok_or("unexpected image token group")?;
                if end - offset != geometry.tokens {
                    return Err("image token group does not match image grid".into());
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
            return Err("prepared image was not consumed by prompt".into());
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

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn processor(limits: ImageLimits) -> ImageProcessor {
        ImageProcessor {
            limits,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            position_side: 48,
            image_token_id: 248056,
            video_token_id: 248057,
        }
    }

    fn png(
        width: u32,
        height: u32,
        depth: png::BitDepth,
        color: png::ColorType,
        data: &[u8],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_depth(depth);
            encoder.set_color(color);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(data).unwrap();
        }
        bytes
    }

    #[test]
    fn rgb_extent_rejects_empty_truncated_and_overflow() {
        assert!(RgbImage::new(0, 1, vec![]).is_err());
        assert!(RgbImage::new(2, 2, vec![0; 11]).is_err());
        assert!(RgbImage::new(usize::MAX, 2, vec![]).is_err());
    }

    #[test]
    fn admission_checks_processed_capacity_without_changing_resize() {
        let processor = processor(ImageLimits::default());
        assert_eq!(processor.geometry(2033, 2033).unwrap().patches, 16_384);
        assert!(
            processor
                .geometry(2080, 2080)
                .unwrap_err()
                .contains("16900 patches")
        );
        assert!(processor.geometry(200, 1).is_ok());
        assert!(
            processor
                .geometry(201, 1)
                .unwrap_err()
                .contains("aspect ratio")
        );
        assert!(processor.geometry(usize::MAX, 2).is_err());
        assert!(processor.geometry(0, 256).is_err());
    }

    #[test]
    fn image_count_and_total_patch_limits_are_independent() {
        let image = || RgbImage::new(1, 1, vec![0; 3]).unwrap();
        let processor = processor(ImageLimits {
            max_images: 2,
            max_total_patches: 256,
            ..ImageLimits::default()
        });
        assert!(processor.process(&[]).is_err());
        assert!(
            processor
                .process(&[image(), image(), image()])
                .err()
                .unwrap()
                .contains("image count")
        );
        assert!(
            processor
                .process(&[image(), image()])
                .err()
                .unwrap()
                .contains("total patch limit")
        );
        assert_eq!(
            processor.process(&[image()]).unwrap().pixel_values.len(),
            256 * FEATURES
        );
    }

    #[test]
    fn png_limits_apply_before_allocating_decoded_pixels() {
        let bytes = png(2, 1, png::BitDepth::Eight, png::ColorType::Rgb, &[1; 6]);
        let input_limit = processor(ImageLimits {
            max_input_pixels: 1,
            ..ImageLimits::default()
        });
        assert!(
            input_limit
                .decode_png(&bytes)
                .err()
                .unwrap()
                .contains("input pixel limit")
        );
        let byte_limit = processor(ImageLimits {
            max_encoded_bytes: bytes.len() - 1,
            ..ImageLimits::default()
        });
        assert!(
            byte_limit
                .decode_png(&bytes)
                .err()
                .unwrap()
                .contains("byte limit")
        );
    }

    #[test]
    fn png_drops_alpha_without_blending_and_rejects_sixteen_bit_or_truncated() {
        let processor = processor(ImageLimits::default());
        let bytes = png(
            2,
            1,
            png::BitDepth::Eight,
            png::ColorType::Rgba,
            &[5, 6, 7, 0, 8, 9, 10, 127],
        );
        assert_eq!(
            processor.decode_png(&bytes).unwrap().pixels(),
            &[5, 6, 7, 8, 9, 10]
        );
        assert!(processor.decode_png(&bytes[..bytes.len() - 12]).is_err());
        let sixteen = png(
            1,
            1,
            png::BitDepth::Sixteen,
            png::ColorType::Grayscale,
            &[1, 2],
        );
        assert!(
            processor
                .decode_png(&sixteen)
                .err()
                .unwrap()
                .contains("8-bit")
        );
    }
}
