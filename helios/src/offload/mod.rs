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
/// `None` selects the default: no memory store and no disk store until the
/// isolation measurements in `docs/prefix-cache-offload.md` enable derived
/// sizes, one tenth of the directory's free space as the disk limit, and the
/// staged transport.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefixCacheOffload {
    /// Registered host memory for retained records; zero selects no memory store.
    pub memory_bytes: Option<u64>,
    /// Directory under which the model's record directory is created.
    pub disk: Option<PathBuf>,
    /// Bytes of record files kept on disk; zero selects no disk store.
    pub disk_limit: Option<u64>,
    pub transport: Option<Transport>,
}

/// Parse a byte count with an optional binary K, M, G, or T suffix, with or
/// without a trailing `B` or `iB`, as the Ignition configuration does.
pub fn parse_size(text: &str) -> std::result::Result<u64, String> {
    let text = text.trim();
    let digits = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    let (number, suffix) = text.split_at(digits);
    let suffix = suffix.trim().to_ascii_uppercase();
    let shift = match suffix.as_str() {
        "" => 0,
        "K" | "KB" | "KIB" => 10,
        "M" | "MB" | "MIB" => 20,
        "G" | "GB" | "GIB" => 30,
        "T" | "TB" | "TIB" => 40,
        _ => return Err(invalid_size(text)),
    };
    if number.is_empty() || number.chars().filter(|c| *c == '.').count() > 1 {
        return Err(invalid_size(text));
    }
    let value: f64 = number.parse().map_err(|_| invalid_size(text))?;
    let bytes = value * (1u64 << shift) as f64;
    if !bytes.is_finite() || bytes < 0.0 || bytes >= u64::MAX as f64 {
        return Err(invalid_size(text));
    }
    Ok(bytes as u64)
}

fn invalid_size(text: &str) -> String {
    format!("size {text:?} must be a number with an optional K, M, G, or T suffix, such as \"16G\"")
}

/// Settings after defaults are applied and the record directory is resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub memory_bytes: u64,
    /// The model's record directory, or `None` for no disk store.
    pub directory: Option<PathBuf>,
    pub disk_limit: u64,
    pub transport: Transport,
}

/// Host resources the loader prepares once per process and the model worker
/// turns into a store: registered pools, the scanned record directory, and
/// every rank's geometry and CUDA context.
pub struct Prepared {
    pub resolved: Resolved,
    pub identity: [u8; 32],
    pub fingerprint: [u8; 16],
    pub geometry: Vec<CacheGeometry>,
    pub layout: Layout,
    /// Each rank's CUDA context handle and local device ordinal.
    pub contexts: Vec<(usize, i32)>,
    pub memory_pool: Option<slab::SlabPool>,
    pub staging_pool: Option<slab::SlabPool>,
    pub scanned: Vec<record::Scanned>,
}

/// The loader's result: either prepared resources or the reason no store exists.
pub enum Preparation {
    Store(Box<Prepared>),
    None(String),
}

impl Preparation {
    pub fn summary(&self) -> String {
        match self {
            Self::Store(prepared) => prepared.summary(),
            Self::None(reason) => reason.clone(),
        }
    }

    pub fn into_shared(self) -> Option<std::sync::Arc<Shared>> {
        match self {
            Self::Store(prepared) => Some(std::sync::Arc::new(Shared {
                summary: prepared.summary(),
                prepared: std::sync::Mutex::new(Some(*prepared)),
            })),
            Self::None(_) => None,
        }
    }
}

impl Prepared {
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(pool) = &self.memory_pool {
            parts.push(format!(
                "memory {} ({} pages + {} rows)",
                gibibytes(pool.bytes()),
                pool.page_count(),
                pool.row_count()
            ));
        }
        if let Some(directory) = &self.resolved.directory {
            parts.push(format!(
                "disk {} (limit {}, {} records, {})",
                directory.display(),
                gibibytes(self.resolved.disk_limit),
                self.scanned.len(),
                match self.resolved.transport {
                    Transport::Staged => "staged",
                    Transport::Direct => "direct",
                }
            ));
        }
        parts.join(" · ")
    }
}

fn gibibytes(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// Shared ownership of prepared resources across rank manifests; the model
/// worker takes them once when it opens the store.
pub struct Shared {
    prepared: std::sync::Mutex<Option<Prepared>>,
    pub summary: String,
}

/// A rank's handle on the shared prepared resources.
#[derive(Clone)]
pub struct Binding {
    shared: std::sync::Arc<Shared>,
    rank: usize,
}

impl Binding {
    pub fn new(shared: std::sync::Arc<Shared>, rank: usize) -> Self {
        Self { shared, rank }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn summary(&self) -> &str {
        &self.shared.summary
    }

    /// Both bindings name the same prepared resources.
    pub fn shares(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.shared, &other.shared)
    }

    /// The prepared resources, once; later calls return `None`.
    pub fn take(&self) -> Option<Prepared> {
        self.shared
            .prepared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Binding")
            .field("rank", &self.rank)
            .field("summary", &self.shared.summary)
            .finish()
    }
}

/// Resolve the settings against the package and host, map and register the
/// pools with rank zero's context current, and scan the record directory.
///
/// `package` is the package file, beside which the default record directory
/// would be created once derived defaults are enabled.
pub fn prepare(
    options: &PrefixCacheOffload,
    package: &std::path::Path,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
) -> Result<Preparation> {
    let _ = package;
    let layout = Layout::new(&geometry)?;
    let fingerprint = geometry::fingerprint(&geometry);
    let memory_bytes = options.memory_bytes.unwrap_or(0);
    let minimum = slab::SlabPool::minimum_bytes(&layout);
    if memory_bytes != 0 && memory_bytes < minimum {
        return Err(format!(
            "host prefix cache memory {memory_bytes} is below the {minimum} bytes this package needs for one outbound and one inbound window"
        )
        .into());
    }
    let transport = options.transport.unwrap_or(Transport::Staged);
    if transport == Transport::Direct {
        return Err("the direct prefix cache transport is not available".into());
    }
    let mut notes = Vec::new();
    let directory = match &options.disk {
        None => None,
        Some(base) => {
            let directory = base.join(record::directory_name(&identity, &fingerprint));
            match std::fs::create_dir_all(&directory) {
                Ok(()) => Some(directory),
                Err(error) => {
                    notes.push(format!(
                        "no disk store: cannot create {}: {error}",
                        directory.display()
                    ));
                    None
                }
            }
        }
    };
    let disk_limit = match &directory {
        None => 0,
        Some(directory) => match options.disk_limit {
            Some(limit) => limit,
            None => free_space(directory)? / 10,
        },
    };
    let directory = directory.filter(|_| disk_limit != 0);
    if memory_bytes == 0 && directory.is_none() {
        notes.insert(0, "disabled".to_owned());
        return Ok(Preparation::None(notes.join("; ")));
    }
    let context = contexts
        .first()
        .ok_or("the host prefix store requires at least one rank")?
        .0;
    let memory_pool = (memory_bytes != 0)
        .then(|| {
            let mut pool = slab::SlabPool::sized(&layout, memory_bytes)?;
            pool.register(context)?;
            Ok::<_, crate::Error>(pool)
        })
        .transpose()?;
    let staging_pool = (memory_bytes == 0 && directory.is_some())
        .then(|| {
            let mut pool = slab::SlabPool::staging(&layout)?;
            pool.register(context)?;
            Ok::<_, crate::Error>(pool)
        })
        .transpose()?;
    let scanned = match &directory {
        Some(directory) => record::scan(directory, &identity, &fingerprint, &geometry)?,
        None => Vec::new(),
    };
    Ok(Preparation::Store(Box::new(Prepared {
        resolved: Resolved {
            memory_bytes,
            directory,
            disk_limit,
            transport,
        },
        identity,
        fingerprint,
        geometry,
        layout,
        contexts,
        memory_pool,
        staging_pool,
        scanned,
    })))
}

/// Free bytes on the filesystem holding `path`.
fn free_space(path: &std::path::Path) -> Result<u64> {
    let text = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| "record directory path contains a NUL byte")?;
    let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: Both pointers are valid for the call; statvfs fills the struct.
    if unsafe { libc::statvfs(text.as_ptr(), &mut stats) } != 0 {
        return Err(format!(
            "cannot inspect the filesystem of {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        )
        .into());
    }
    Ok(stats.f_bavail * stats.f_frsize)
}

/// Open the configured store over the prepared resources, starting its
/// transfer threads on every rank's context.
pub fn open(prepared: Prepared) -> Result<Box<dyn PrefixStore>> {
    let Prepared {
        resolved,
        identity,
        fingerprint,
        geometry,
        layout,
        contexts,
        memory_pool,
        staging_pool,
        scanned,
    } = prepared;
    let transfers = std::sync::Arc::new(std::sync::Mutex::new(transfer::Transfers::start(
        &contexts, geometry,
    )?));
    let disk = resolved
        .directory
        .map(|directory| {
            let transport = match resolved.transport {
                Transport::Staged => disk_store::Staged::new(
                    staging_pool,
                    transfers.clone(),
                    disk::DiskThread::start("flare-offload-disk")?,
                ),
                Transport::Direct => {
                    return Err("the direct prefix cache transport is not available".into());
                }
            };
            disk_store::DiskStore::new(
                directory,
                identity,
                fingerprint,
                resolved.disk_limit,
                layout.page_tokens,
                scanned,
                Box::new(transport),
            )
        })
        .transpose()?;
    // Inside a composite the disk index owns the tree; memory nodes are flat.
    let memory = memory_pool
        .map(|pool| {
            if disk.is_some() {
                memory::MemoryStore::new_flat(identity, pool, transfers.clone())
            } else {
                memory::MemoryStore::new(identity, pool, transfers.clone())
            }
        })
        .transpose()?;
    match (memory, disk) {
        (Some(memory), Some(disk)) => Ok(Box::new(composite::CompositeStore::new(memory, disk)?)),
        (Some(memory), None) => Ok(Box::new(memory)),
        (None, Some(disk)) => Ok(Box::new(disk)),
        (None, None) => Err("no prefix store tier is configured".into()),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_parse_binary_suffixes_like_the_ignition_configuration() {
        assert_eq!(parse_size("16G").unwrap(), 16 << 30);
        assert_eq!(parse_size("1.5 GiB").unwrap(), 3 << 29);
        assert_eq!(parse_size("512m").unwrap(), 512 << 20);
        assert_eq!(parse_size("2TB").unwrap(), 2 << 40);
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert_eq!(parse_size("0").unwrap(), 0);
        for text in ["", "G", "16X", "1.2.3G", "-1G", "16 GB extra"] {
            assert!(parse_size(text).is_err(), "{text:?}");
        }
    }

    #[test]
    fn fetch_policy_compares_transfer_time_with_half_the_prefill_time() {
        let hit = HostHit {
            key: PrefixKey {
                consumed_tokens: 8000,
                digest: [0; 16],
            },
            consumed_tokens: 8000,
            resident_tokens: 0,
            pages: 125,
            bytes: 1 << 30,
            tier: Tier::Memory,
            estimated_seconds: 0.4,
            chain: index::Match {
                key: PrefixKey {
                    consumed_tokens: 8000,
                    digest: [0; 16],
                },
                consumed_tokens: 8000,
                resident_tokens: 0,
                nodes: Vec::new(),
                skip_pages: 0,
                pages: 125,
            },
        };
        assert!(worth_fetching(&hit, 8000.0));
        assert!(!worth_fetching(&hit, 16000.0));
        let mut slow = hit.clone();
        slow.estimated_seconds = 0.6;
        assert!(!worth_fetching(&slow, 8000.0));
    }
}
