use super::MiMo;
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
        let mut values = self.text.uploads();
        values.extend(self.staging.uploads());
        values
    }
    pub fn pack(&mut self, frame: &InputStep<'_, MiMo>) -> Result<()> {
        self.staging.clear();
        self.text.pack(frame)?;
        if let Some(row) =
            (0..frame.step.lanes.len()).find(|&row| frame.step.lane_kind(row) == GraphKind::Prefill)
        {
            let lane = &frame.step.lanes[row];
            let offset = frame.step.lanes[..row]
                .iter()
                .map(|lane| lane.tokens)
                .sum::<usize>()
                * 4;
            let data = &frame.admissions[lane.request.slot as usize]
                .as_ref()
                .unwrap()
                .data;
            let start = lane.start_position as usize;
            self.staging.i32(
                "vision_token_indices",
                offset,
                data.vision_token_indices[start..start + lane.tokens]
                    .iter()
                    .copied(),
            );
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
        self.staging.put(
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
        let (columns, rows) = column_permutation(h / 2, w / 2);
        self.staging.i32("vision_column_indices", 0, columns);
        self.staging.i32("vision_row_indices", 0, rows);
        self.staging
            .i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
        self.staging
            .i32("vision_num_patches", 0, [geometry.patches as i32]);
        self.staging
            .i32("vision_num_features", 0, [geometry.tokens as i32]);
        self.staging.i32("vision_slot", 0, [handle.slot as i32]);
        self.staging
            .i64("vision_generation", 0, [handle.generation as i64]);
        self.staging
            .i32("vision_feature_offset", 0, [(start / 4) as i32]);
        Ok(())
    }
}

#[cfg(feature = "images")]
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

#[cfg(all(test, feature = "images"))]
mod tests {
    use super::*;
    #[test]
    fn rectangular_patch_groups_restore_row_order() {
        let (columns, rows) = column_permutation(2, 3);
        assert_eq!(columns, [0, 3, 1, 4, 2, 5]);
        assert_eq!(rows, [0, 2, 4, 1, 3, 5]);
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(columns[*row as usize], index as i32);
        }
    }
}
