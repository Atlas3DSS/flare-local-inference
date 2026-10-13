use super::{DeepSeek4, PromptData};
use crate::{
    Result,
    execution::{GraphKind, Upload},
    model::{InputStep, inputs::UploadBatch, paged_text},
};

#[derive(Default)]
pub struct Inputs {
    text: paged_text::Inputs,
    staging: UploadBatch,
}

impl Inputs {
    pub fn uploads(&self) -> Vec<Upload<'_>> {
        let mut uploads = self.text.uploads();
        uploads.extend(self.staging.uploads());
        uploads
    }
    pub fn pack(&mut self, frame: &InputStep<'_, DeepSeek4>) -> Result<()> {
        self.staging.clear();
        self.text.pack(frame)?;
        if frame.step.kind == GraphKind::Prefill {
            let lane = &frame.step.lanes[0];
            let data: &PromptData = &frame.admissions[lane.request.slot as usize]
                .as_ref()
                .unwrap()
                .data;
            let start = lane.start_position as usize;
            let end = start + lane.tokens;
            self.staging.i32(
                "vision_token_indices",
                0,
                data.vision_token_indices[start..end].iter().copied(),
            );
            let mut left = vec![0; lane.tokens];
            let mut right = vec![0; lane.tokens];
            for span in data
                .spans
                .iter()
                .filter(|span| span.start < end && span.end > start)
            {
                if span.start < start || span.end > end {
                    return Err("prefill splits a bidirectional image span".into());
                }
                for position in span.clone() {
                    left[position - start] = (position - span.start) as i32;
                    right[position - start] = (span.end - position - 1) as i32;
                }
            }
            self.staging.i32("vision_left", 0, left);
            self.staging.i32("vision_right", 0, right);
        }
        Ok(())
    }
    #[cfg(feature = "images")]
    pub fn pack_vision(
        &mut self,
        config: &crate::model::ModelConfig,
        handle: crate::scheduler::RequestHandle,
        images: &crate::frontend::images::ProcessedImages,
        image: usize,
    ) -> Result<()> {
        self.text.clear();
        self.staging.clear();
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
        self.staging.put(
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
        self.staging.i64(
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
        self.staging.i32("vision_merge_indices", 0, indices);
        self.staging.i32("vision_cu_seqlens", 0, [0, actual as i32]);
        self.staging.i32("vision_num_patches", 0, [actual as i32]);
        self.staging
            .i32("vision_num_features", 0, [geometry.tokens as i32]);
        self.staging.i32("vision_slot", 0, [handle.slot as i32]);
        self.staging
            .i64("vision_generation", 0, [handle.generation as i64]);
        self.staging
            .i32("vision_feature_offset", 0, [(start / 9) as i32]);
        Ok(())
    }
}
