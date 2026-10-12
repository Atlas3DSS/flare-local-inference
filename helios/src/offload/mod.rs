//! Host retention of published prefix checkpoints in registered memory and on disk.
//!
//! A GPU prefix checkpoint owns its tail pages, the complete cache pages since its
//! parent, and one continuation-state row. The host store copies those bytes out
//! when the scheduler publishes a checkpoint and copies them back into free GPU
//! ids when a later request matches a checkpoint the GPU no longer holds. The
//! restored bytes are published as an ordinary GPU checkpoint, so the model
//! worker, the scheduler, and the recipes see no difference between a checkpoint
//! produced by prefill and one restored from the host.
//!
//! Three stores implement [`PrefixStore`]: [`memory::MemoryStore`] retains records
//! in registered host memory, [`disk_store::DiskStore`] retains record files, and
//! [`composite::CompositeStore`] composes the two with write-through and
//! read-through. Every store is driven from the model worker thread and never
//! blocks it: `store` and `fetch` enqueue work, `poll` drains completions.

pub mod composite;
pub mod disk;
pub mod disk_store;
pub mod geometry;
pub mod index;
pub mod memory;
pub mod record;
pub mod slab;
pub mod transfer;

use std::path::PathBuf;

use crate::{
    Result,
    cache::{CheckpointId, PageId, PrefixHistory, PrefixKey},
    scheduler::StateSlotId,
};

pub use geometry::{CacheGeometry, Layout, Plane};

/// Identifies one transfer or disk job issued by a store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

/// Where a looked-up record is read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Memory,
    Disk,
}

/// How record files move between disk and the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// Through a registered staging pool and batched stream copies.
    Staged,
    /// Through GPUDirect Storage, with no host payload staging.
    Direct,
}

/// Offload settings as the command line and configuration file supply them.
/// `None` selects the derived default described in `docs/prefix-cache-offload.md`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefixCacheOffload {
    /// Registered host memory for retained records; zero selects no memory store.
    pub memory_bytes: Option<u64>,
    /// Record directory; `Some(None)` selects no disk store.
    pub disk: Option<Option<PathBuf>>,
    /// Bytes of record files kept on disk; zero selects no disk store.
    pub disk_limit: Option<u64>,
    /// `None` selects the direct transport when its probe succeeds, otherwise staged.
    pub transport: Option<Transport>,
}

/// A GPU checkpoint the scheduler published, as the worker reports it to the store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishedCheckpoint {
    pub id: CheckpointId,
    pub key: PrefixKey,
    /// Key of the GPU parent checkpoint, or `None` at the root.
    pub parent: Option<PrefixKey>,
    /// Digest of the aligned hash-block boundary at or below the endpoint.
    pub bucket: [u8; 16],
    /// Every page of the checkpoint's chain from the root in ascending order.
    /// The last `tail_pages` entries belong to the checkpoint itself.
    pub chain_pages: Vec<PageId>,
    pub tail_pages: usize,
    /// State slot holding the checkpoint's row.
    pub slot: StateSlotId,
}

impl PublishedCheckpoint {
    pub fn tail(&self) -> &[PageId] {
        &self.chain_pages[self.chain_pages.len() - self.tail_pages..]
    }
}

/// A host record chain that extends a GPU checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct HostHit {
    pub key: PrefixKey,
    pub consumed_tokens: u64,
    /// Consumed count of the GPU checkpoint the chain extends.
    pub resident_tokens: u64,
    /// Pages the restore writes: complete pages at the endpoint minus complete
    /// pages at the resident checkpoint.
    pub pages: usize,
    pub bytes: u64,
    pub tier: Tier,
    pub estimated_seconds: f64,
    pub(crate) chain: index::Match,
}

/// Destination ids the scheduler reserved for one inbound transfer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InboundTicket {
    pub key: PrefixKey,
    pub pages: Vec<PageId>,
    pub slot: StateSlotId,
}

/// Completion of work a store accepted earlier.
#[derive(Debug)]
pub enum StoreEvent {
    /// The copy-out of a published checkpoint finished; the worker may unpin it.
    Stored {
        checkpoint: CheckpointId,
        key: PrefixKey,
        result: Result<u64>,
    },
    /// The copy-in for an inbound ticket finished; the worker reports it to the scheduler.
    Fetched { key: PrefixKey, result: Result<u64> },
}

/// Counters every store maintains; rates are bytes per second of completed transfers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StoreStats {
    pub lookups: u64,
    pub hits: u64,
    pub reused_tokens: u64,
    pub stored: u64,
    pub store_skipped: u64,
    pub store_failed: u64,
    pub fetched: u64,
    pub fetch_failed: u64,
    pub bytes_out: u64,
    pub bytes_in: u64,
    pub outbound_in_flight: usize,
    pub inbound_in_flight: usize,
    pub memory_records: usize,
    pub memory_bytes: u64,
    pub disk_records: usize,
    pub disk_bytes: u64,
    pub outbound_rate: f64,
    pub inbound_rate: f64,
}

/// A host tier the model worker drives without waiting on it.
///
/// `store` and `fetch` enqueue work and return. `poll` reports completions.
/// The worker pins the GPU checkpoint before `store` and unpins it when `store`
/// returns `Ok(false)` or an error, or when the matching `StoreEvent::Stored`
/// arrives. Ticket ids belong to the scheduler until `StoreEvent::Fetched`.
pub trait PrefixStore {
    /// Deepest record chain extending the GPU checkpoint at `resident` tokens.
    fn lookup(&mut self, history: &PrefixHistory, resident: u64) -> Option<HostHit>;
    /// Copy a published checkpoint out. `Ok(false)` means the key is already
    /// present or in flight, or no slabs could be freed, and no event follows.
    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<bool>;
    /// Copy a looked-up chain into the ticket's ids. An error means no event follows.
    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()>;
    fn poll(&mut self, events: &mut Vec<StoreEvent>);
    fn stats(&self) -> StoreStats;
    /// Drain transfers, join threads, and finish pending disk writes.
    fn close(&mut self) -> Result<()>;
}

/// Bytes per second assumed for a tier before its first transfers are measured.
pub const INITIAL_MEMORY_RATE: f64 = 20e9;
pub const INITIAL_STAGED_DISK_RATE: f64 = 3e9;
pub const INITIAL_DIRECT_DISK_RATE: f64 = 10e9;
/// Prefill tokens per second assumed before the worker supplies a measurement.
pub const INITIAL_PREFILL_RATE: f64 = 8000.0;

/// A fetch is worthwhile when the transfer takes less than half the prefill it replaces.
pub fn worth_fetching(hit: &HostHit, prefill_tokens_per_second: f64) -> bool {
    let gained = (hit.consumed_tokens - hit.resident_tokens) as f64;
    hit.estimated_seconds < 0.5 * gained / prefill_tokens_per_second.max(1.0)
}
