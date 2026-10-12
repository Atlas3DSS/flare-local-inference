//! Registered host memory split into fixed-size page and row slabs.
//!
//! One anonymous mapping, 4 KiB aligned and advised for transparent huge pages,
//! holds every slab. A page slab holds one page record across all ranks and a
//! row slab holds one row record across all ranks, in the cross-rank
//! [`Layout`](super::Layout). Registration with the CUDA driver is a separate
//! step so CPU tests run without a driver. Allocation never blocks: when no
//! slab is free the caller evicts or fails its job.

use std::collections::VecDeque;

use crate::{Result, cuda::HostRegistration};

use super::geometry::Layout;

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
}

impl SubPool {
    fn new(base: usize, slab_bytes: usize, count: usize) -> Self {
        Self {
            base,
            slab_bytes,
            count,
            free: (0..count as u32).collect(),
        }
    }
    fn address(&self, index: u32) -> usize {
        debug_assert!((index as usize) < self.count);
        self.base + index as usize * self.slab_bytes
    }
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
    /// transparent huge pages.
    pub fn new(
        page_bytes: u64,
        page_count: usize,
        row_bytes: u64,
        row_count: usize,
        retain: bool,
    ) -> Result<Self> {
        let _ = (page_bytes, page_count, row_bytes, row_count, retain);
        todo!("map the pool and split it into sub-pools")
    }

    /// A retaining pool of `bytes` with rows taking one quarter, at least the
    /// minimum window counts; a smaller `bytes` fails with the minimum in the message.
    pub fn sized(layout: &Layout, bytes: u64) -> Result<Self> {
        let _ = (layout, bytes);
        todo!("derive counts from the layout and the byte budget")
    }

    /// A non-retaining staging pool of [`STAGING_PAGE_SLABS`] and [`STAGING_ROW_SLABS`].
    pub fn staging(layout: &Layout) -> Result<Self> {
        let _ = layout;
        todo!("derive the staging pool from the layout")
    }

    /// Bytes of the smallest retaining pool for a layout.
    pub fn minimum_bytes(layout: &Layout) -> u64 {
        MINIMUM_PAGE_SLABS as u64 * layout.page_record_bytes
            + MINIMUM_ROW_SLABS as u64 * layout.row_record_bytes
    }

    /// Register the mapping with the driver as portable so every rank's context
    /// can address it, with `context` made current on the calling thread.
    pub fn register(&mut self, context: usize) -> Result<()> {
        let _ = context;
        todo!("crate::cuda::register_host with the portable flag")
    }

    pub fn unregister(&mut self) -> Result<()> {
        todo!("drop the registration")
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
        self.pages.free.pop_front().map(PageSlab)
    }
    pub fn allocate_row(&mut self) -> Option<RowSlab> {
        self.rows.free.pop_front().map(RowSlab)
    }
    pub fn release_page(&mut self, slab: PageSlab) {
        debug_assert!(!self.pages.free.contains(&slab.0));
        self.pages.free.push_back(slab.0);
    }
    pub fn release_row(&mut self, slab: RowSlab) {
        debug_assert!(!self.rows.free.contains(&slab.0));
        self.rows.free.push_back(slab.0);
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
