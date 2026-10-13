//! Host- or graph-produced n-gram requests and asynchronous compact-row delivery.

mod reader;
use crate::{
    Error, Result,
    cuda::{PinnedHost, Stream},
    execution::{ExecutionManifest, GraphKind, Upload},
};
pub(crate) use reader::Plane;
use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Storage {
    #[default]
    Disk,
    Ram,
}

impl Storage {
    /// Storage when the launcher names none. A discrete GPU keeps tables in RAM when they
    /// take at most half the available host memory: Qwen's 29 GB PLE table read from disk
    /// lengthened 32-request GB300 rounds by a quarter, since rows are needed at layer 1. An
    /// integrated GPU (GB10) would take resident tables from the memory its KV cache uses.
    pub fn choose(integrated: bool, table_bytes: u64, available: Option<u64>) -> Self {
        match available {
            Some(available) if !integrated && table_bytes <= available / 2 => Self::Ram,
            _ => Self::Disk,
        }
    }

    /// `MemAvailable` from `/proc/meminfo`, in bytes.
    pub fn available_host_bytes() -> Option<u64> {
        std::fs::read_to_string("/proc/meminfo")
            .ok()?
            .lines()
            .find_map(|line| line.strip_prefix("MemAvailable:"))?
            .trim()
            .strip_suffix("kB")?
            .trim()
            .parse::<u64>()
            .ok()?
            .checked_mul(1024)
    }
}

#[derive(Debug)]
pub(crate) struct Table {
    pub(crate) planes: [Plane; 2],
    pub(crate) heads: usize,
    pub(crate) weights: String,
    pub(crate) scales: String,
    pub(crate) hashes: String,
    pub(crate) event: String,
    pub(crate) request: Option<String>,
}

#[derive(Debug)]
pub(crate) struct Index {
    pub(crate) vocabulary: usize,
    pub(crate) tables: Vec<Table>,
}

struct Staging {
    data: PinnedHost,
    ready: PinnedHost,
    error: Mutex<Option<Error>>,
}
impl Staging {
    fn signal(&self, generation: u32) {
        unsafe { &*(self.ready.address() as *const AtomicU32) }
            .store(generation, Ordering::Release);
    }
}
enum Requests {
    Host(Vec<i64>),
    Device {
        address: usize,
        flag: usize,
        sequence: u64,
        rows: usize,
    },
}
impl Requests {
    fn rows(&self) -> usize {
        match self {
            Self::Host(ids) => ids.len(),
            Self::Device { rows, .. } => *rows,
        }
    }
    fn read(self, cancelled: &AtomicBool) -> Result<Vec<i64>> {
        match self {
            Self::Host(ids) => Ok(ids),
            Self::Device {
                address,
                flag,
                sequence,
                rows,
            } => {
                let ready = unsafe { &*(flag as *const AtomicU64) };
                while ready.load(Ordering::Acquire) != sequence {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("n-gram request cancelled".into());
                    }
                    std::hint::spin_loop();
                }
                Ok(unsafe { std::slice::from_raw_parts(address as *const i64, rows) }.to_vec())
            }
        }
    }
}
struct Job {
    generation: u32,
    requests: Requests,
}
struct State {
    sequence: u64,
    generation: u32,
    submitted: usize,
    completed: usize,
}
struct Layer {
    staging: Arc<Staging>,
    sender: Option<mpsc::SyncSender<Job>>,
    worker: Option<JoinHandle<()>>,
    state: Mutex<State>,
    cancelled: Arc<AtomicBool>,
}
impl Drop for Layer {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub(crate) struct Shared {
    pub index: Arc<Index>,
    layers: Vec<Layer>,
    ranks: usize,
    max_tokens: usize,
}
impl Shared {
    pub fn new(
        index: Index,
        path: &Path,
        ranks: usize,
        max_tokens: usize,
        context: usize,
        device: i32,
        storage: Storage,
    ) -> Result<Arc<Self>> {
        let stream = Stream::open(context, device)?;
        let index = Arc::new(index);
        let mut layers = Vec::new();
        for table in &index.tables {
            let mut reader = reader::Reader::open(path, &table.planes, storage == Storage::Ram)?;
            let staging = Arc::new(Staging {
                data: stream.mapped_host_buffer(
                    max_tokens
                        * table.heads
                        * table.planes.iter().map(|p| p.row_bytes).sum::<usize>(),
                )?,
                ready: stream.mapped_host_buffer(4)?,
                error: Mutex::new(None),
            });
            let (sender, receiver) = mpsc::sync_channel::<Job>(1);
            let buffers = staging.clone();
            let planes = table.planes;
            let row_bytes = planes.iter().map(|p| p.row_bytes).sum::<usize>();
            let cancelled = Arc::new(AtomicBool::new(false));
            let stop = cancelled.clone();
            let worker = thread::Builder::new()
                .name(format!("helios-{}", table.event))
                .spawn(move || {
                    while let Ok(job) = receiver.recv() {
                        // A single reader owns these bytes. Every rank completes the previous
                        // consumer graph before another job can reuse them; flag publication
                        // precedes GPU DMA through the queued stream memory wait.
                        let output = unsafe {
                            std::slice::from_raw_parts_mut(
                                buffers.data.address() as *mut u8,
                                job.requests.rows() * row_bytes,
                            )
                        };
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let ids = job.requests.read(&stop)?;
                            reader.gather(&planes, &ids, output)
                        }))
                        .unwrap_or_else(|_| Err("Engram reader panicked".into()));
                        if let Err(error) = result {
                            output.fill(0);
                            *buffers.error.lock().unwrap() = Some(error);
                        }
                        // Release failed reads too, so graph shutdown can drain before reporting the error.
                        buffers.signal(job.generation);
                    }
                })
                .map_err(|e| format!("start Engram reader: {e}"))?;
            layers.push(Layer {
                staging,
                cancelled,
                sender: Some(sender),
                worker: Some(worker),
                state: Mutex::new(State {
                    sequence: 0,
                    generation: 0,
                    submitted: 0,
                    completed: 0,
                }),
            });
        }
        Ok(Arc::new(Self {
            index,
            layers,
            ranks,
            max_tokens,
        }))
    }
    fn start(
        &self,
        table: usize,
        rank: usize,
        sequence: u64,
        requests: Requests,
    ) -> Result<(usize, u32)> {
        let rows = requests.rows();
        if rows > self.max_tokens * self.index.tables[table].heads {
            return Err("Engram input exceeds staged capacity".into());
        }
        let layer = &self.layers[table];
        let mut state = layer.state.lock().unwrap();
        if let Some(error) = layer.staging.error.lock().unwrap().as_ref() {
            return Err(error.clone());
        }
        if state.sequence != sequence {
            if state.sequence > sequence
                || (state.sequence != 0 && state.completed != (1 << self.ranks) - 1)
            {
                return Err("Engram staging reused before all ranks completed".into());
            }
            let generation = state.generation.wrapping_add(1);
            layer
                .sender
                .as_ref()
                .unwrap()
                .send(Job {
                    generation,
                    requests,
                })
                .map_err(|_| "Engram reader disconnected")?;
            *state = State {
                sequence,
                generation,
                submitted: 0,
                completed: 0,
            };
        }
        if state.submitted & (1 << rank) != 0 {
            return Err("rank submitted Engram generation twice".into());
        }
        state.submitted |= 1 << rank;
        Ok((rows, state.generation))
    }
    fn finish(&self, rank: usize, sequence: u64) -> Result<()> {
        let mut error = None;
        for layer in &self.layers {
            let mut state = layer.state.lock().unwrap();
            if state.sequence != sequence || state.submitted & (1 << rank) == 0 {
                return Err("Engram completion does not match its submission".into());
            }
            state.completed |= 1 << rank;
            error = error.or_else(|| layer.staging.error.lock().unwrap().clone());
        }
        error.map_or(Ok(()), Err)
    }
}

#[derive(Clone)]
pub struct Binding {
    pub(crate) shared: Arc<Shared>,
    pub(crate) rank: usize,
    pub(crate) events: Vec<usize>,
}
impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngramBinding")
            .field("rank", &self.rank)
            .finish()
    }
}
impl Binding {
    pub(crate) fn index(&self) -> Arc<Index> {
        self.shared.index.clone()
    }
}

struct Transfer {
    stream: Stream,
    flag: usize,
    weights: usize,
    scales: usize,
    event: usize,
    request: Option<(usize, usize)>,
}
pub(crate) struct Transfers {
    binding: Binding,
    layers: Vec<Transfer>,
}
impl Transfers {
    pub fn new(binding: Binding, manifest: &ExecutionManifest) -> Result<Self> {
        let mut layers = Vec::new();
        for (i, table) in binding.shared.index.tables.iter().enumerate() {
            let stream = Stream::open(manifest.context, manifest.device)?;
            let flag = stream.mapped_address(&binding.shared.layers[i].staging.ready)?;
            layers.push(Transfer {
                stream,
                flag,
                weights: manifest.buffer(&table.weights)?.address,
                scales: manifest.buffer(&table.scales)?.address,
                event: binding.events[i],
                request: table
                    .request
                    .as_ref()
                    .map(|name| {
                        let buffer = manifest.buffer(name)?;
                        Ok::<_, Error>((buffer.address, buffer.address + buffer.nbytes - 8))
                    })
                    .transpose()?,
            });
        }
        Ok(Self { binding, layers })
    }
    pub fn enqueue(&self, sequence: u64, tokens: usize, uploads: &[Upload<'_>]) -> Result<()> {
        for (i, (transfer, table)) in self
            .layers
            .iter()
            .zip(&self.binding.shared.index.tables)
            .enumerate()
        {
            let heads = table.heads;
            let requests = if let Some((address, flag)) = transfer.request {
                Requests::Device {
                    address,
                    flag,
                    sequence,
                    rows: tokens * heads,
                }
            } else {
                let upload = uploads
                    .iter()
                    .find(|u| u.buffer == table.hashes)
                    .ok_or("step lacks n-gram hash IDs")?;
                if upload.offset != 0 || upload.bytes.len() % (heads * 8) != 0 {
                    return Err("invalid n-gram hash upload".into());
                }
                Requests::Host(
                    upload
                        .bytes
                        .chunks_exact(8)
                        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                )
            };
            let (rows, generation) =
                self.binding
                    .shared
                    .start(i, self.binding.rank, sequence, requests)?;
            let staging = &self.binding.shared.layers[i].staging;
            unsafe {
                transfer.stream.wait_value(transfer.flag, generation)?;
                transfer.stream.upload(
                    transfer.weights,
                    &staging.data,
                    0,
                    rows * table.planes[0].row_bytes,
                )?;
                transfer.stream.upload(
                    transfer.scales,
                    &staging.data,
                    rows * table.planes[0].row_bytes,
                    rows * table.planes[1].row_bytes,
                )?;
                transfer.stream.record_borrowed(transfer.event)?;
            }
        }
        Ok(())
    }
    pub fn finish(&self, sequence: u64) -> Result<()> {
        self.binding.shared.finish(self.binding.rank, sequence)
    }
    pub fn cancel(&self) {
        for layer in &self.binding.shared.layers {
            layer.cancelled.store(true, Ordering::Release);
        }
    }
    pub fn drain(&self) -> Result<()> {
        let mut error = None;
        for transfer in &self.layers {
            error = error.or(transfer.stream.synchronize().err());
        }
        error.map_or(Ok(()), Err)
    }
}

pub(crate) fn consumes(kind: GraphKind) -> bool {
    matches!(
        kind,
        GraphKind::Prefill | GraphKind::Decode | GraphKind::Speculative
    )
}

#[cfg(test)]
mod tests {
    use super::Storage;

    #[test]
    fn discrete_gpus_keep_tables_in_ram_only_with_room_for_them() {
        const GIB: u64 = 1 << 30;
        assert_eq!(
            Storage::choose(false, 29 * GIB, Some(415 * GIB)),
            Storage::Ram
        );
        assert_eq!(
            Storage::choose(false, 29 * GIB, Some(57 * GIB)),
            Storage::Disk
        );
        assert_eq!(Storage::choose(false, 29 * GIB, None), Storage::Disk);
        assert_eq!(
            Storage::choose(true, 29 * GIB, Some(415 * GIB)),
            Storage::Disk
        );
        assert!(Storage::available_host_bytes().is_some_and(|bytes| bytes > 0));
    }
}
