//! Mutually exclusive ownership of snapshot aliases and KV page ids.

use super::*;

impl RequestScheduler {
    pub fn share_snapshot_pages(&mut self, ranges: &[std::ops::Range<u32>]) -> Result<()> {
        if self.submitted != Fence(0)
            || self.slots.iter().any(|s| s.request.is_some())
            || ranges.len() != self.limits.snapshots as usize
            || ranges
                .iter()
                .any(|r| r.start > r.end || r.end > self.limits.pages)
        {
            return Err(
                "snapshot aliases must cover valid KV ids and be configured before admission"
                    .into(),
            );
        }
        self.snapshot_pages = ranges.to_vec();
        Ok(())
    }

    pub(super) fn available_snapshot(&self, keep_pages: usize) -> Option<SnapshotId> {
        self.free_snapshots.iter().copied().find(|id| {
            let Some(range) = self.snapshot_pages.get(id.0 as usize) else {
                return true;
            };
            let mut cost = 0;
            for page in range.clone() {
                if self.page_snapshot_refs[page as usize] == 0 {
                    if !self.free_pages.contains(&PageId(page)) {
                        return false;
                    }
                    cost += 1;
                }
            }
            self.free_pages.len().saturating_sub(cost) >= keep_pages
        })
    }

    pub(super) fn reserve_snapshot(&mut self, keep_pages: usize) -> Option<SnapshotId> {
        let id = self.available_snapshot(keep_pages)?;
        self.free_snapshots.remove(&id);
        if let Some(range) = self.snapshot_pages.get(id.0 as usize) {
            for page in range.clone() {
                self.free_pages.remove(&PageId(page));
                if self.page_snapshot_refs[page as usize] == 0 {
                    self.stats.snapshot_backing_pages += 1;
                }
                self.page_snapshot_refs[page as usize] += 1;
            }
        }
        Some(id)
    }

    /// Reserves a snapshot, evicting unpinned checkpoints. Only a capture worth more than an
    /// idle conversation's checkpoint (`release_pins`) also releases conversation pins.
    pub(super) fn retain_snapshot(&mut self, release_pins: bool) -> Result<Option<SnapshotId>> {
        loop {
            if let Some(id) = self.reserve_snapshot(0) {
                return Ok(Some(id));
            }
            let evicted = self.cache.evict(EvictionTarget {
                pages: 0,
                snapshots: 1,
            });
            self.stats.snapshot_evictions += evicted.retired.len() as u64;
            let update = self.cache.advance_completed(self.completed)?;
            self.apply_completion(update);
            if evicted.retired.is_empty() && !(release_pins && self.release_conversation_pin()?) {
                return Ok(None);
            }
        }
    }

    pub(super) fn release_snapshot(&mut self, id: SnapshotId) {
        assert!(self.free_snapshots.insert(id), "snapshot released twice");
        if let Some(range) = self.snapshot_pages.get(id.0 as usize) {
            for page in range.clone() {
                let count = &mut self.page_snapshot_refs[page as usize];
                *count = count.checked_sub(1).expect("snapshot alias ownership");
                if *count == 0 {
                    self.stats.snapshot_backing_pages -= 1;
                    assert!(self.free_pages.insert(PageId(page)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_in_execution_alignment_padding_do_not_exclude_pages() {
        let mut s = super::super::tests::scheduler(1, 16, 1);
        s.share_snapshot_pages(&[0..0]).unwrap();
        let snapshot = s.reserve_snapshot(16).unwrap();
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.stats().snapshot_backing_pages, 0);
        s.release_snapshot(snapshot);
        assert_eq!(s.free_page_count(), 16);
    }

    #[test]
    fn overlapping_snapshot_aliases_release_pages_only_after_the_last_owner() {
        let mut s = super::super::tests::scheduler(2, 16, 2);
        s.share_snapshot_pages(&[8..12, 10..14]).unwrap();
        let a = s.reserve_snapshot(0).unwrap();
        let b = s.reserve_snapshot(0).unwrap();
        assert_eq!(s.free_page_count(), 10);
        assert_eq!(s.stats().snapshot_backing_pages, 6);
        s.release_snapshot(a);
        assert_eq!(s.free_page_count(), 12);
        assert_eq!(s.stats().snapshot_backing_pages, 4);
        assert!(!s.free_pages.contains(&PageId(10)));
        s.release_snapshot(b);
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.stats().snapshot_backing_pages, 0);
    }

    #[test]
    fn request_page_reservations_exclude_snapshot_aliases() {
        let mut s = super::super::tests::scheduler(2, 16, 2);
        s.share_snapshot_pages(&[0..8, 8..16]).unwrap();
        let handle = s.admit(super::super::tests::spec(64 * 12, 1)).unwrap();
        assert_eq!(s.free_snapshot_count(), 2);
        assert!(s.reserve_snapshot(0).is_none());
        s.cancel(handle).unwrap();
        assert!(s.reserve_snapshot(8).is_some());
        assert!(s.reserve_snapshot(1).is_none());
    }

    #[test]
    fn cancellation_keeps_snapshot_backing_until_the_last_gpu_fence() {
        let mut s = super::super::tests::scheduler(1, 16, 1);
        s.share_snapshot_pages(&[8..16]).unwrap();
        let handle = s.admit(super::super::tests::spec(64, 1)).unwrap();
        let snapshot = s.reserve_snapshot(0).unwrap();
        s.request_mut(handle).unwrap().terminal_snapshot = Some(snapshot);
        let step = s.plan().unwrap();
        s.commit(&step, Fence(1)).unwrap();
        s.cancel(handle).unwrap();
        assert_eq!(s.stats().snapshot_backing_pages, 8);
        assert!(s.reserve_snapshot(0).is_none());
        s.complete(
            Fence(1),
            &[CompletionRecord::default(); REQUEST_LANES],
            Default::default(),
        )
        .unwrap();
        assert_eq!(s.stats().snapshot_backing_pages, 0);
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.free_snapshot_count(), 1);
    }
}
