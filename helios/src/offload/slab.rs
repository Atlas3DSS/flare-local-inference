//! Registered host memory split into fixed-size page and row slabs.
//!
//! One anonymous mapping, 4 KiB aligned and advised for transparent huge pages,
//! holds every slab. A page slab holds one page record across all ranks and a
//! row slab holds one row record across all ranks, in the cross-rank
//! [`Layout`](super::Layout). Registration with the CUDA driver is a separate
//! step so CPU tests run without a driver. Allocation never blocks: when no
//! slab is free the caller evicts or fails its job.
//!
//! Each sub-pool hands out slabs from a FIFO free list that starts in index
//! order, and tracks which slabs are allocated so that releasing a slab that is
//! free, released twice, or outside the pool is ignored instead of duplicating
//! a free-list entry. The free lists are sized at construction and never grow.

use std::collections::VecDeque;

use crate::{Result, cuda::HostRegistration};

use super::geometry::{BLOCK_BYTES, Layout, invalid};

/// Page slabs in the smallest retaining pool: one outbound and one inbound window.
pub const MINIMUM_PAGE_SLABS: usize = 128;
/// Row slabs in the smallest retaining pool.
pub const MINIMUM_ROW_SLABS: usize = 2;
/// Page slabs in a staging pool: two outbound and two inbound windows.
pub const STAGING_PAGE_SLABS: usize = 256;
/// Row slabs in a staging pool.
pub const STAGING_ROW_SLABS: usize = 4;
/// Pages per transfer window.
pub const WINDOW_PAGES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageSlab(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RowSlab(pub u32);

struct SubPool {
    base: usize,
    slab_bytes: usize,
    count: usize,
    free: VecDeque<u32>,
    allocated: Vec<bool>,
}

impl SubPool {
    fn new(base: usize, slab_bytes: usize, count: usize) -> Self {
        let mut free = VecDeque::with_capacity(count);
        free.extend(0..count as u32);
        Self {
            base,
            slab_bytes,
            count,
            free,
            allocated: vec![false; count],
        }
    }

    /// Slab indices are produced only by `allocate`, so an index outside the
    /// pool is a caller defect; the panic keeps `page` and `row` inside the mapping.
    fn address(&self, index: u32) -> usize {
        assert!(
            (index as usize) < self.count,
            "slab index {index} outside a pool of {} slabs",
            self.count
        );
        self.base + index as usize * self.slab_bytes
    }

    fn allocate(&mut self) -> Option<u32> {
        let index = self.free.pop_front()?;
        self.allocated[index as usize] = true;
        Some(index)
    }

    /// Return `index` to the back of the free list. A slab that is not
    /// allocated, including one released twice or an index outside the pool,
    /// is ignored so the free list never holds a slab twice.
    fn release(&mut self, index: u32) {
        if let Some(allocated) = self.allocated.get_mut(index as usize)
            && *allocated
        {
            *allocated = false;
            self.free.push_back(index);
        }
    }
}

/// Bytes a region of `count` slabs of `slab_bytes` occupies. A zero count
/// occupies nothing and ignores the slab size; otherwise the size must be a
/// nonzero multiple of [`BLOCK_BYTES`] and the region must fit the address space.
fn region_bytes(kind: &str, slab_bytes: u64, count: usize) -> Result<usize> {
    if count == 0 {
        return Ok(0);
    }
    if slab_bytes == 0 || !slab_bytes.is_multiple_of(BLOCK_BYTES) {
        return Err(invalid(format!(
            "{kind} slab size of {slab_bytes} bytes is not a nonzero multiple of {BLOCK_BYTES}"
        )));
    }
    u32::try_from(count)
        .ok()
        .and_then(|_| usize::try_from(slab_bytes).ok()?.checked_mul(count))
        .ok_or_else(|| {
            invalid(format!(
                "{count} {kind} slabs of {slab_bytes} bytes exceed the addressable pool size"
            ))
        })
}

/// Fixed pools of page and row slabs inside one registered mapping.
pub struct SlabPool {
    mapping: usize,
    mapping_bytes: usize,
    pages: SubPool,
    rows: SubPool,
    retain: bool,
    registration: Option<HostRegistration>,
}

// SAFETY: The mapping is plain process memory; the pool is used from one thread
// at a time and slab addresses handed to other threads outlive their transfers.
unsafe impl Send for SlabPool {}

impl SlabPool {
    /// Map `page_count` page slabs of `page_bytes` and `row_count` row slabs of
    /// `row_bytes`, each a multiple of [`BLOCK_BYTES`], populated and advised for
    /// transparent huge pages. Page slabs precede row slabs in the mapping.
    pub fn new(
        page_bytes: u64,
        page_count: usize,
        row_bytes: u64,
        row_count: usize,
        retain: bool,
    ) -> Result<Self> {
        let page_region = region_bytes("page", page_bytes, page_count)?;
        let row_region = region_bytes("row", row_bytes, row_count)?;
        let mapping_bytes = page_region.checked_add(row_region).ok_or_else(|| {
            invalid("page and row slabs together exceed the addressable pool size")
        })?;
        if mapping_bytes == 0 {
            return Err(invalid(
                "a slab pool requires at least one page or row slab",
            ));
        }
        // SAFETY: An anonymous private mapping with no address hint has no
        // preconditions; the result is checked before use.
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapping_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE,
                -1,
                0,
            )
        };
        if mapping == libc::MAP_FAILED {
            return Err(invalid(format!(
                "mapping {mapping_bytes} bytes of host memory for slabs failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        // Huge pages are an optimization; a kernel that refuses the advice
        // leaves the mapping backed by ordinary pages.
        // SAFETY: The range was mapped by this call and is not yet shared.
        unsafe {
            libc::madvise(mapping, mapping_bytes, libc::MADV_HUGEPAGE);
        }
        let base = mapping as usize;
        Ok(Self {
            mapping: base,
            mapping_bytes,
            pages: SubPool::new(base, page_bytes as usize, page_count),
            rows: SubPool::new(base + page_region, row_bytes as usize, row_count),
            retain,
            registration: None,
        })
    }

    /// A retaining pool of at most `bytes`: rows take one quarter and pages the
    /// rest, each at least its minimum window count, with rows yielding to the
    /// page minimum so the pool never exceeds `bytes`. A `bytes` below
    /// [`SlabPool::minimum_bytes`] fails with that minimum in the message.
    pub fn sized(layout: &Layout, bytes: u64) -> Result<Self> {
        let page_record = layout.page_record_bytes;
        let row_record = layout.row_record_bytes;
        if page_record == 0 || row_record == 0 {
            return Err(invalid(
                "record layout declares zero-byte page or row records",
            ));
        }
        let minimum = Self::minimum_bytes(layout);
        if bytes < minimum {
            return Err(invalid(format!(
                "prefix cache memory of {bytes} bytes is below the minimum of {minimum} bytes \
                 for this layout: {MINIMUM_PAGE_SLABS} page records of {page_record} bytes \
                 and {MINIMUM_ROW_SLABS} row records of {row_record} bytes"
            )));
        }
        let quarter_rows = bytes / 4 / row_record;
        let most_rows = (bytes - MINIMUM_PAGE_SLABS as u64 * page_record) / row_record;
        let row_count = quarter_rows.max(MINIMUM_ROW_SLABS as u64).min(most_rows);
        let page_count = (bytes - row_count * row_record) / page_record;
        Self::new(
            page_record,
            page_count as usize,
            row_record,
            row_count as usize,
            true,
        )
    }

    /// A non-retaining staging pool of [`STAGING_PAGE_SLABS`] and [`STAGING_ROW_SLABS`].
    pub fn staging(layout: &Layout) -> Result<Self> {
        Self::new(
            layout.page_record_bytes,
            STAGING_PAGE_SLABS,
            layout.row_record_bytes,
            STAGING_ROW_SLABS,
            false,
        )
    }

    /// Bytes of the smallest retaining pool for a layout.
    pub fn minimum_bytes(layout: &Layout) -> u64 {
        MINIMUM_PAGE_SLABS as u64 * layout.page_record_bytes
            + MINIMUM_ROW_SLABS as u64 * layout.row_record_bytes
    }

    /// Register the mapping with the driver as portable so every rank's context
    /// can address it, with `context` made current on the calling thread.
    /// A pool that is already registered is not registered again.
    pub fn register(&mut self, context: usize) -> Result<()> {
        if self.registration.is_some() {
            return Err(invalid(
                "slab pool is already registered with the CUDA driver",
            ));
        }
        // SAFETY: The mapping is page aligned and stays mapped at this size for
        // the pool's lifetime; `Drop` releases the registration before unmapping.
        let registration =
            unsafe { crate::cuda::register_host(context, self.mapping, self.mapping_bytes) }?;
        self.registration = Some(registration);
        Ok(())
    }

    /// Release the driver registration; a pool that is not registered is unchanged.
    pub fn unregister(&mut self) -> Result<()> {
        self.registration.take();
        Ok(())
    }

    pub fn registered(&self) -> bool {
        self.registration.is_some()
    }

    pub fn retain(&self) -> bool {
        self.retain
    }

    pub fn page_bytes(&self) -> usize {
        self.pages.slab_bytes
    }
    pub fn row_bytes(&self) -> usize {
        self.rows.slab_bytes
    }
    pub fn page_count(&self) -> usize {
        self.pages.count
    }
    pub fn row_count(&self) -> usize {
        self.rows.count
    }
    pub fn free_pages(&self) -> usize {
        self.pages.free.len()
    }
    pub fn free_rows(&self) -> usize {
        self.rows.free.len()
    }
    pub fn bytes(&self) -> u64 {
        self.mapping_bytes as u64
    }

    pub fn allocate_page(&mut self) -> Option<PageSlab> {
        self.pages.allocate().map(PageSlab)
    }
    pub fn allocate_row(&mut self) -> Option<RowSlab> {
        self.rows.allocate().map(RowSlab)
    }
    /// Returns `slab` to the free list; a slab that is not allocated is ignored.
    pub fn release_page(&mut self, slab: PageSlab) {
        self.pages.release(slab.0);
    }
    /// Returns `slab` to the free list; a slab that is not allocated is ignored.
    pub fn release_row(&mut self, slab: RowSlab) {
        self.rows.release(slab.0);
    }

    pub fn page_address(&self, slab: PageSlab) -> usize {
        self.pages.address(slab.0)
    }
    pub fn row_address(&self, slab: RowSlab) -> usize {
        self.rows.address(slab.0)
    }

    pub fn page(&self, slab: PageSlab) -> &[u8] {
        // SAFETY: The mapping outlives the pool and the slab lies inside it.
        unsafe {
            std::slice::from_raw_parts(self.page_address(slab) as *const u8, self.pages.slab_bytes)
        }
    }
    pub fn page_mut(&mut self, slab: PageSlab) -> &mut [u8] {
        // SAFETY: As above, with exclusive access through `&mut self`.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.page_address(slab) as *mut u8,
                self.pages.slab_bytes,
            )
        }
    }
    pub fn row(&self, slab: RowSlab) -> &[u8] {
        // SAFETY: As for `page`.
        unsafe {
            std::slice::from_raw_parts(self.row_address(slab) as *const u8, self.rows.slab_bytes)
        }
    }
    pub fn row_mut(&mut self, slab: RowSlab) -> &mut [u8] {
        // SAFETY: As for `page_mut`.
        unsafe {
            std::slice::from_raw_parts_mut(self.row_address(slab) as *mut u8, self.rows.slab_bytes)
        }
    }
}

impl Drop for SlabPool {
    fn drop(&mut self) {
        self.registration.take();
        if self.mapping != 0 {
            // SAFETY: The mapping was created by this pool and is no longer registered.
            unsafe {
                libc::munmap(self.mapping as *mut libc::c_void, self.mapping_bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offload::geometry::tests::geometry;

    const PAGE: u64 = 8192;
    const ROW: u64 = 4096;

    /// One rank with two 4 KiB planes and a 4 KiB row: 8 KiB page records, 4 KiB row records.
    fn layout() -> Layout {
        Layout::new(&[geometry(64, &[("k", 4096), ("v", 4096)], 4096)]).unwrap()
    }

    #[test]
    fn slabs_are_block_aligned_and_disjoint() {
        let mut pool = SlabPool::new(PAGE, 6, ROW, 3, true).unwrap();
        assert_eq!(pool.bytes(), 6 * PAGE + 3 * ROW);
        let pages: Vec<_> = (0..6).map(|_| pool.allocate_page().unwrap()).collect();
        let rows: Vec<_> = (0..3).map(|_| pool.allocate_row().unwrap()).collect();
        let mut ranges = Vec::new();
        for &slab in &pages {
            let address = pool.page_address(slab);
            assert_eq!(address % BLOCK_BYTES as usize, 0);
            ranges.push((address, address + PAGE as usize));
        }
        for &slab in &rows {
            let address = pool.row_address(slab);
            assert_eq!(address % BLOCK_BYTES as usize, 0);
            ranges.push((address, address + ROW as usize));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            assert!(pair[0].1 <= pair[1].0, "slabs overlap: {pair:?}");
        }
        let end = pool.mapping + pool.mapping_bytes;
        assert!(
            ranges
                .iter()
                .all(|(start, stop)| *start >= pool.mapping && *stop <= end)
        );

        for (fill, &slab) in (1u8..).zip(&pages) {
            pool.page_mut(slab).fill(fill);
        }
        for (fill, &slab) in (101u8..).zip(&rows) {
            pool.row_mut(slab).fill(fill);
        }
        for (fill, &slab) in (1u8..).zip(&pages) {
            assert!(pool.page(slab).iter().all(|&byte| byte == fill));
        }
        for (fill, &slab) in (101u8..).zip(&rows) {
            assert!(pool.row(slab).iter().all(|&byte| byte == fill));
        }
    }

    #[test]
    fn exhaustion_returns_none_then_reuses_released_slabs_in_release_order() {
        let mut pool = SlabPool::new(PAGE, 3, ROW, 2, true).unwrap();
        assert_eq!(pool.free_pages(), 3);
        assert_eq!(pool.free_rows(), 2);
        let pages: Vec<_> = (0..3).map(|_| pool.allocate_page().unwrap()).collect();
        assert_eq!(pages, [PageSlab(0), PageSlab(1), PageSlab(2)]);
        assert_eq!(pool.allocate_page(), None);
        assert_eq!(pool.free_pages(), 0);
        pool.release_page(PageSlab(1));
        pool.release_page(PageSlab(0));
        assert_eq!(pool.free_pages(), 2);
        assert_eq!(pool.allocate_page(), Some(PageSlab(1)));
        assert_eq!(pool.allocate_page(), Some(PageSlab(0)));
        assert_eq!(pool.allocate_page(), None);

        let rows: Vec<_> = (0..2).map(|_| pool.allocate_row().unwrap()).collect();
        assert_eq!(rows, [RowSlab(0), RowSlab(1)]);
        assert_eq!(pool.allocate_row(), None);
        pool.release_row(RowSlab(1));
        assert_eq!(pool.allocate_row(), Some(RowSlab(1)));
        assert_eq!(pool.allocate_row(), None);
        pool.release_row(RowSlab(0));
        pool.release_row(RowSlab(1));
        assert_eq!(pool.free_rows(), 2);
    }

    #[test]
    fn double_release_does_not_duplicate_a_slab() {
        let mut pool = SlabPool::new(PAGE, 2, ROW, 2, true).unwrap();
        let page = pool.allocate_page().unwrap();
        pool.release_page(page);
        pool.release_page(page);
        assert_eq!(pool.free_pages(), 2);
        pool.release_page(PageSlab(1));
        pool.release_page(PageSlab(7));
        assert_eq!(pool.free_pages(), 2);
        let mut pages = [pool.allocate_page().unwrap(), pool.allocate_page().unwrap()];
        pages.sort_unstable();
        assert_eq!(pages, [PageSlab(0), PageSlab(1)]);
        assert_eq!(pool.allocate_page(), None);

        let row = pool.allocate_row().unwrap();
        pool.release_row(row);
        pool.release_row(row);
        pool.release_row(RowSlab(9));
        assert_eq!(pool.free_rows(), 2);
        let mut rows = [pool.allocate_row().unwrap(), pool.allocate_row().unwrap()];
        rows.sort_unstable();
        assert_eq!(rows, [RowSlab(0), RowSlab(1)]);
        assert_eq!(pool.allocate_row(), None);
    }

    #[test]
    fn sized_gives_rows_a_quarter_and_pages_the_rest() {
        let layout = layout();
        let bytes = 16u64 << 20;
        let pool = SlabPool::sized(&layout, bytes).unwrap();
        assert!(pool.retain());
        assert_eq!(pool.page_bytes(), PAGE as usize);
        assert_eq!(pool.row_bytes(), ROW as usize);
        assert_eq!(pool.row_count(), (bytes / 4 / ROW) as usize);
        assert_eq!(pool.page_count(), ((bytes - bytes / 4) / PAGE) as usize);
        assert_eq!(pool.bytes(), bytes);
        assert_eq!(pool.free_pages(), pool.page_count());
        assert_eq!(pool.free_rows(), pool.row_count());
    }

    #[test]
    fn sized_never_goes_below_the_minimum_counts_or_above_the_budget() {
        let layout = layout();
        let minimum = SlabPool::minimum_bytes(&layout);
        let pool = SlabPool::sized(&layout, minimum).unwrap();
        assert_eq!(pool.page_count(), MINIMUM_PAGE_SLABS);
        assert_eq!(pool.row_count(), MINIMUM_ROW_SLABS);
        assert_eq!(pool.bytes(), minimum);

        let budget = minimum + 3 * ROW;
        let pool = SlabPool::sized(&layout, budget).unwrap();
        assert_eq!(pool.page_count(), MINIMUM_PAGE_SLABS);
        assert_eq!(pool.row_count(), MINIMUM_ROW_SLABS + 3);
        assert_eq!(pool.bytes(), budget);

        let budget = minimum + 7 * PAGE + 1;
        let pool = SlabPool::sized(&layout, budget).unwrap();
        assert!(pool.page_count() >= MINIMUM_PAGE_SLABS);
        assert!(pool.row_count() >= MINIMUM_ROW_SLABS);
        assert!(pool.bytes() <= budget);
    }

    #[test]
    fn sized_below_the_minimum_reports_the_minimum() {
        let layout = layout();
        let minimum = SlabPool::minimum_bytes(&layout);
        let Err(error) = SlabPool::sized(&layout, minimum - 1) else {
            panic!("a budget below the minimum must fail");
        };
        let message = error.to_string();
        assert!(message.contains(&minimum.to_string()), "{message}");
        assert!(SlabPool::sized(&layout, 0).is_err());
    }

    #[test]
    fn minimum_bytes_is_one_outbound_and_one_inbound_window() {
        let layout = layout();
        assert_eq!(layout.page_record_bytes, PAGE);
        assert_eq!(layout.row_record_bytes, ROW);
        assert_eq!(SlabPool::minimum_bytes(&layout), 128 * PAGE + 2 * ROW);
    }

    #[test]
    fn staging_pool_has_fixed_counts_and_no_retention() {
        let pool = SlabPool::staging(&layout()).unwrap();
        assert!(!pool.retain());
        assert_eq!(pool.page_count(), STAGING_PAGE_SLABS);
        assert_eq!(pool.row_count(), STAGING_ROW_SLABS);
        assert_eq!(pool.free_pages(), STAGING_PAGE_SLABS);
        assert_eq!(pool.free_rows(), STAGING_ROW_SLABS);
        assert_eq!(
            pool.bytes(),
            STAGING_PAGE_SLABS as u64 * PAGE + STAGING_ROW_SLABS as u64 * ROW
        );
    }

    #[test]
    fn writes_through_mutable_slices_are_visible_through_shared_slices() {
        let mut pool = SlabPool::new(PAGE, 2, ROW, 2, false).unwrap();
        let page = pool.allocate_page().unwrap();
        let row = pool.allocate_row().unwrap();
        assert_eq!(pool.page(page).len(), PAGE as usize);
        assert_eq!(pool.row(row).len(), ROW as usize);
        assert!(pool.page(page).iter().all(|&byte| byte == 0));
        for (index, byte) in pool.page_mut(page).iter_mut().enumerate() {
            *byte = index as u8;
        }
        pool.row_mut(row).copy_from_slice(&[0xA5; 4096]);
        assert!(
            pool.page(page)
                .iter()
                .enumerate()
                .all(|(index, &byte)| byte == index as u8)
        );
        assert!(pool.row(row).iter().all(|&byte| byte == 0xA5));
        assert_eq!(pool.page(page).as_ptr() as usize, pool.page_address(page));
        assert_eq!(pool.row(row).as_ptr() as usize, pool.row_address(row));
    }

    #[test]
    fn rejects_unaligned_empty_or_oversized_slabs() {
        assert!(SlabPool::new(4000, 1, ROW, 1, true).is_err());
        assert!(SlabPool::new(0, 1, ROW, 1, true).is_err());
        assert!(SlabPool::new(PAGE, 1, 4097, 1, true).is_err());
        assert!(SlabPool::new(PAGE, 0, ROW, 0, true).is_err());
        assert!(SlabPool::new(PAGE, usize::MAX, ROW, 1, true).is_err());
        assert!(SlabPool::new(PAGE, 1, ROW, u32::MAX as usize + 1, true).is_err());

        let mut pool = SlabPool::new(0, 0, ROW, 1, true).unwrap();
        assert_eq!(pool.page_count(), 0);
        assert_eq!(pool.allocate_page(), None);
        assert_eq!(pool.allocate_row(), Some(RowSlab(0)));
        assert_eq!(pool.bytes(), ROW);
    }

    #[test]
    fn unregister_without_registration_is_ok() {
        let mut pool = SlabPool::new(PAGE, 1, ROW, 1, true).unwrap();
        assert!(!pool.registered());
        pool.unregister().unwrap();
        pool.unregister().unwrap();
        assert!(!pool.registered());
    }
}
