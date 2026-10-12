//! A thread that moves record segments between files and caller-owned buffers
//! with direct I/O.
//!
//! The thread owns one `io_uring` ring. Files are opened with `O_DIRECT`, so
//! every buffer, offset, and length is a multiple of [`BLOCK_BYTES`], which the
//! record layout guarantees. Transfers are chunked at [`CHUNK_BYTES`]. Writes
//! hash every page record and the row as they pass through; reads verify them
//! against the record's checksum segment and header. The thread runs under
//! `SCHED_BATCH` with a positive nice value so it yields to serving threads.
//!
//! Jobs execute one at a time in submission order, so a record's `Create`,
//! page and row writes, and `Commit` or `Abort` form a sequence under one id,
//! as do `Open`, reads, and `Close`. Every job produces exactly one [`Reply`].
//! A job that fails leaves no partial record behind: a failed `Commit` closes
//! and unlinks the temporary file. The thread allocates its ring and its
//! aligned scratch block once at start; the scratch holds the header block and
//! a checksum segment of up to [`MAX_TAIL_PAGES`] pages, which bounds the
//! records the thread accepts.

use std::{
    cell::Cell,
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, SyncSender, TrySendError},
};

use io_uring::{IoUring, opcode, types};

use crate::{Error, Result};

use super::{
    JobId,
    geometry::{BLOCK_BYTES, invalid},
    record::{HEADER_BYTES, Header, checksum, io_failure},
};

pub const CHUNK_BYTES: usize = 8 << 20;
/// Jobs the submitting store may have outstanding; `submit` fails when full.
pub const QUEUE_DEPTH: usize = 64;
pub const NICE: i32 = 10;
/// Chunks the ring keeps in flight for one transfer.
const RING_DEPTH: usize = 64;
/// Bytes of the thread's aligned checksum scratch.
const CHECKSUM_SCRATCH_BYTES: usize = 1 << 20;
/// Tail pages per record the thread can write or open, bounded by the scratch
/// that holds a record's checksum segment during `Commit` and `Open`.
pub const MAX_TAIL_PAGES: u32 = (CHECKSUM_SCRATCH_BYTES / 8) as u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Create,
    WritePages,
    WriteRow,
    Commit,
    Abort,
    Open,
    ReadPages,
    ReadRow,
    Close,
    Truncate,
    Unlink,
}

/// Work for the disk thread. Buffers are host addresses of page or row slabs.
/// `Create` through `Commit` or `Abort` share one id and run in submission
/// order; `Open` through `Close` likewise. A buffer must stay valid and
/// untouched until the job's reply arrives.
#[derive(Debug)]
pub enum Job {
    /// Create the temporary file for a record and keep it open under `id`.
    Create {
        id: JobId,
        temporary: PathBuf,
        header: Header,
    },
    /// Write page records `first..first + buffers.len()` from `buffers`.
    WritePages {
        id: JobId,
        window: u32,
        first: u32,
        buffers: Vec<usize>,
    },
    WriteRow {
        id: JobId,
        buffer: usize,
    },
    /// Write the checksum segment and header, optionally `fdatasync`, rename to
    /// `path`, and close. Every page record and the row must have been written.
    Commit {
        id: JobId,
        path: PathBuf,
        sync: bool,
    },
    /// Close and unlink the temporary file.
    Abort {
        id: JobId,
    },
    /// Open a record file for reading under `id` and load its header and checksums.
    Open {
        id: JobId,
        path: PathBuf,
    },
    /// Read page records `first..first + buffers.len()` into `buffers`, verifying checksums.
    ReadPages {
        id: JobId,
        window: u32,
        first: u32,
        buffers: Vec<usize>,
    },
    ReadRow {
        id: JobId,
        buffer: usize,
    },
    Close {
        id: JobId,
    },
    /// Drop the row segment of a committed record.
    Truncate {
        id: JobId,
        path: PathBuf,
        length: u64,
    },
    Unlink {
        id: JobId,
        path: PathBuf,
    },
}

#[derive(Debug)]
pub struct Reply {
    pub id: JobId,
    pub operation: Operation,
    pub window: u32,
    pub result: Result<()>,
    /// Bytes moved by this job; for `Commit`, every byte written for the record.
    pub bytes: u64,
    /// The header read by `Open`.
    pub header: Option<Header>,
}

pub struct DiskThread {
    sender: Option<SyncSender<Job>>,
    receiver: Receiver<Reply>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Jobs submitted whose replies have not been polled.
    outstanding: Cell<usize>,
}

impl DiskThread {
    /// Spawn the thread under `name` with its ring and scratch allocated.
    pub fn start(name: &str) -> Result<Self> {
        let ring = Ring::new()?;
        let scratch = AlignedBuffer::new(HEADER_BYTES + CHECKSUM_SCRATCH_BYTES)?;
        let (sender, jobs) = mpsc::sync_channel(QUEUE_DEPTH);
        let (replies, receiver) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || run(ring, scratch, jobs, replies))
            .map_err(|error| io_failure(format!("spawn disk thread {name}"), error))?;
        Ok(Self {
            sender: Some(sender),
            receiver,
            thread: Some(thread),
            outstanding: Cell::new(0),
        })
    }

    /// Enqueue without waiting; an error means the queue is full or the thread stopped.
    pub fn submit(&self, job: Job) -> Result<()> {
        let sender = self
            .sender
            .as_ref()
            .ok_or_else(|| invalid("disk thread is closed"))?;
        match sender.try_send(job) {
            Ok(()) => {
                self.outstanding.set(self.outstanding.get() + 1);
                Ok(())
            }
            Err(TrySendError::Full(_)) => Err(invalid(format!(
                "disk thread queue is full with {QUEUE_DEPTH} jobs outstanding"
            ))),
            Err(TrySendError::Disconnected(_)) => Err(invalid("disk thread stopped")),
        }
    }

    pub fn poll(&self, replies: &mut Vec<Reply>) {
        while let Ok(reply) = self.receiver.try_recv() {
            self.outstanding
                .set(self.outstanding.get().saturating_sub(1));
            replies.push(reply);
        }
    }

    /// Finish outstanding jobs and join the thread. The thread answers every
    /// queued job before it exits, and those replies remain available through
    /// `poll` after this returns.
    pub fn close(&mut self) -> Result<()> {
        self.sender.take();
        let Some(thread) = self.thread.take() else {
            return Ok(());
        };
        thread.join().map_err(|_| {
            invalid(format!(
                "disk thread stopped abnormally; {} submitted jobs have no polled reply",
                self.outstanding.get()
            ))
        })
    }
}

impl Drop for DiskThread {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Every direct-I/O span must be a multiple of the block size.
pub fn aligned(value: u64) -> bool {
    value.is_multiple_of(BLOCK_BYTES)
}

fn run(ring: Ring, scratch: AlignedBuffer, jobs: Receiver<Job>, replies: mpsc::Sender<Reply>) {
    lower_priority();
    let mut worker = Worker {
        ring,
        scratch,
        records: HashMap::new(),
    };
    while let Ok(job) = jobs.recv() {
        if replies.send(worker.execute(job)).is_err() {
            break;
        }
    }
}

/// Batch scheduling with a raised nice value; a host that refuses either
/// request leaves the thread at its inherited priority.
fn lower_priority() {
    // SAFETY: Both calls take plain values and act on the calling thread only.
    unsafe {
        let param: libc::sched_param = std::mem::zeroed();
        libc::sched_setscheduler(0, libc::SCHED_BATCH, &param);
        libc::setpriority(libc::PRIO_PROCESS, 0, NICE);
    }
}

/// A record being written: its temporary file, header, and the checksums of
/// the page records written so far.
struct Writing {
    file: File,
    temporary: PathBuf,
    header: Header,
    checksums: Vec<u64>,
    written: Vec<bool>,
    row_written: bool,
    bytes: u64,
}

/// A committed record open for reading with its checksum segment loaded.
struct Reading {
    file: File,
    path: PathBuf,
    header: Header,
    checksums: Vec<u64>,
}

enum Record {
    Writing(Writing),
    Reading(Reading),
}

struct Worker {
    ring: Ring,
    scratch: AlignedBuffer,
    records: HashMap<JobId, Record>,
}

impl Worker {
    fn execute(&mut self, job: Job) -> Reply {
        match job {
            Job::Create {
                id,
                temporary,
                header,
            } => reply(id, Operation::Create, 0, self.create(id, temporary, header)),
            Job::WritePages {
                id,
                window,
                first,
                buffers,
            } => reply(
                id,
                Operation::WritePages,
                window,
                self.write_pages(id, first, &buffers),
            ),
            Job::WriteRow { id, buffer } => {
                reply(id, Operation::WriteRow, 0, self.write_row(id, buffer))
            }
            Job::Commit { id, path, sync } => {
                reply(id, Operation::Commit, 0, self.commit(id, &path, sync))
            }
            Job::Abort { id } => reply(id, Operation::Abort, 0, self.abort(id)),
            Job::Open { id, path } => match self.open(id, path) {
                Ok((bytes, header)) => Reply {
                    id,
                    operation: Operation::Open,
                    window: 0,
                    result: Ok(()),
                    bytes,
                    header: Some(header),
                },
                Err(error) => reply(id, Operation::Open, 0, Err(error)),
            },
            Job::ReadPages {
                id,
                window,
                first,
                buffers,
            } => reply(
                id,
                Operation::ReadPages,
                window,
                self.read_pages(id, first, &buffers),
            ),
            Job::ReadRow { id, buffer } => {
                reply(id, Operation::ReadRow, 0, self.read_row(id, buffer))
            }
            Job::Close { id } => reply(id, Operation::Close, 0, self.close(id)),
            Job::Truncate { id, path, length } => {
                reply(id, Operation::Truncate, 0, truncate(&path, length))
            }
            Job::Unlink { id, path } => reply(id, Operation::Unlink, 0, unlink(&path)),
        }
    }

    fn create(&mut self, id: JobId, temporary: PathBuf, header: Header) -> Result<u64> {
        if self.records.contains_key(&id) {
            return Err(invalid(format!("disk job {} is already open", id.0)));
        }
        check_tail_pages(&header)?;
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .custom_flags(libc::O_DIRECT)
            .open(&temporary)
            .map_err(|error| direct_open_failure("create record file", &temporary, error))?;
        let tail = header.tail_pages as usize;
        self.records.insert(
            id,
            Record::Writing(Writing {
                file,
                temporary,
                header,
                checksums: vec![0; tail],
                written: vec![false; tail],
                row_written: false,
                bytes: 0,
            }),
        );
        Ok(0)
    }

    fn write_pages(&mut self, id: JobId, first: u32, buffers: &[usize]) -> Result<u64> {
        let writing = writing(&mut self.records, id)?;
        let page_bytes = writing.header.page_record_bytes();
        let pages = page_range(&writing.header, first, buffers.len())?;
        let mut spans = Vec::with_capacity(buffers.len());
        for (page, address) in pages.zip(buffers) {
            // SAFETY: The caller keeps the slab at `address` valid and unchanged
            // for `page_bytes` until this job's reply.
            let bytes =
                unsafe { std::slice::from_raw_parts(*address as *const u8, page_bytes as usize) };
            writing.checksums[page] = checksum(bytes);
            writing.written[page] = true;
            spans.push(Span {
                address: *address,
                offset: writing.header.page_segment + page as u64 * page_bytes,
                length: page_bytes as usize,
            });
        }
        let moved = self.ring.transfer(&writing.file, &spans, true)?;
        writing.bytes += moved;
        Ok(moved)
    }

    fn write_row(&mut self, id: JobId, buffer: usize) -> Result<u64> {
        let writing = writing(&mut self.records, id)?;
        let row_bytes = writing.header.row_record_bytes();
        // SAFETY: As for page buffers.
        let bytes = unsafe { std::slice::from_raw_parts(buffer as *const u8, row_bytes as usize) };
        writing.header.row_checksum = checksum(bytes);
        writing.row_written = true;
        let span = Span {
            address: buffer,
            offset: writing.header.row_segment,
            length: row_bytes as usize,
        };
        let moved = self.ring.transfer(&writing.file, &[span], true)?;
        writing.bytes += moved;
        Ok(moved)
    }

    /// Remove the record being written under `id`.
    fn take_writing(&mut self, id: JobId) -> Result<Writing> {
        writing(&mut self.records, id)?;
        match self.records.remove(&id) {
            Some(Record::Writing(writing)) => Ok(writing),
            Some(other) => {
                self.records.insert(id, other);
                Err(invalid(format!(
                    "disk job {} is not a record being written",
                    id.0
                )))
            }
            None => Err(invalid(format!("disk job {} names no open record", id.0))),
        }
    }

    fn commit(&mut self, id: JobId, path: &Path, sync: bool) -> Result<u64> {
        let mut writing = self.take_writing(id)?;
        let outcome = self.finish(&mut writing, path, sync);
        if outcome.is_err() {
            drop(writing.file);
            let _ = fs::remove_file(&writing.temporary);
        }
        outcome
    }

    fn finish(&mut self, writing: &mut Writing, path: &Path, sync: bool) -> Result<u64> {
        let missing = writing.written.iter().filter(|written| !**written).count();
        if missing > 0 || !writing.row_written {
            return Err(invalid(format!(
                "record {} cannot be committed: {missing} of {} page records and {} row are unwritten",
                writing.temporary.display(),
                writing.written.len(),
                if writing.row_written { "no" } else { "its" }
            )));
        }
        let checksum_bytes = writing.header.checksum_bytes() as usize;
        let encoded = writing.header.encode()?;
        let scratch = self.scratch.as_mut_slice();
        scratch[..HEADER_BYTES].copy_from_slice(&encoded);
        let segment = &mut scratch[HEADER_BYTES..HEADER_BYTES + checksum_bytes];
        for (slot, sum) in segment.chunks_exact_mut(8).zip(&writing.checksums) {
            slot.copy_from_slice(&sum.to_le_bytes());
        }
        segment[writing.checksums.len() * 8..].fill(0);
        let mut spans = vec![Span {
            address: self.scratch.address(),
            offset: 0,
            length: HEADER_BYTES,
        }];
        if checksum_bytes > 0 {
            spans.push(Span {
                address: self.scratch.address() + HEADER_BYTES,
                offset: HEADER_BYTES as u64,
                length: checksum_bytes,
            });
        }
        writing.bytes += self.ring.transfer(&writing.file, &spans, true)?;
        if sync {
            writing.file.sync_data().map_err(|error| {
                io_failure(
                    format!("fdatasync record file {}", writing.temporary.display()),
                    error,
                )
            })?;
        }
        fs::rename(&writing.temporary, path).map_err(|error| {
            io_failure(
                format!(
                    "rename record file {} to {}",
                    writing.temporary.display(),
                    path.display()
                ),
                error,
            )
        })?;
        Ok(writing.bytes)
    }

    fn abort(&mut self, id: JobId) -> Result<u64> {
        let writing = self.take_writing(id)?;
        drop(writing.file);
        match fs::remove_file(&writing.temporary) {
            Ok(()) => Ok(0),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(0),
            Err(error) => Err(io_failure(
                format!(
                    "unlink temporary record file {}",
                    writing.temporary.display()
                ),
                error,
            )),
        }
    }

    fn open(&mut self, id: JobId, path: PathBuf) -> Result<(u64, Header)> {
        if self.records.contains_key(&id) {
            return Err(invalid(format!("disk job {} is already open", id.0)));
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&path)
            .map_err(|error| direct_open_failure("open record file", &path, error))?;
        let header_span = Span {
            address: self.scratch.address(),
            offset: 0,
            length: HEADER_BYTES,
        };
        let mut bytes = self.ring.transfer(&file, &[header_span], false)?;
        let header = Header::decode(&self.scratch.as_slice()[..HEADER_BYTES])
            .map_err(|error| invalid(format!("record file {}: {error}", path.display())))?;
        check_tail_pages(&header)?;
        let checksum_bytes = header.checksum_bytes() as usize;
        if checksum_bytes > 0 {
            let span = Span {
                address: self.scratch.address() + HEADER_BYTES,
                offset: HEADER_BYTES as u64,
                length: checksum_bytes,
            };
            bytes += self.ring.transfer(&file, &[span], false)?;
        }
        let segment = &self.scratch.as_slice()[HEADER_BYTES..HEADER_BYTES + checksum_bytes];
        let checksums = segment
            .chunks_exact(8)
            .take(header.tail_pages as usize)
            .map(|slot| u64::from_le_bytes(slot.try_into().unwrap()))
            .collect();
        self.records.insert(
            id,
            Record::Reading(Reading {
                file,
                path,
                header: header.clone(),
                checksums,
            }),
        );
        Ok((bytes, header))
    }

    fn read_pages(&mut self, id: JobId, first: u32, buffers: &[usize]) -> Result<u64> {
        let reading = reading(&mut self.records, id)?;
        let page_bytes = reading.header.page_record_bytes();
        let pages = page_range(&reading.header, first, buffers.len())?;
        let spans: Vec<Span> = pages
            .clone()
            .zip(buffers)
            .map(|(page, address)| Span {
                address: *address,
                offset: reading.header.page_segment + page as u64 * page_bytes,
                length: page_bytes as usize,
            })
            .collect();
        let moved = self.ring.transfer(&reading.file, &spans, false)?;
        for (page, address) in pages.zip(buffers) {
            // SAFETY: The caller keeps the slab at `address` valid for
            // `page_bytes` until this job's reply, and the read into it completed.
            let bytes =
                unsafe { std::slice::from_raw_parts(*address as *const u8, page_bytes as usize) };
            if checksum(bytes) != reading.checksums[page] {
                return Err(invalid(format!(
                    "page record {page} of {} failed checksum verification",
                    reading.path.display()
                )));
            }
        }
        Ok(moved)
    }

    fn read_row(&mut self, id: JobId, buffer: usize) -> Result<u64> {
        let reading = reading(&mut self.records, id)?;
        let row_bytes = reading.header.row_record_bytes();
        let span = Span {
            address: buffer,
            offset: reading.header.row_segment,
            length: row_bytes as usize,
        };
        let moved = self.ring.transfer(&reading.file, &[span], false)?;
        // SAFETY: As for page buffers.
        let bytes = unsafe { std::slice::from_raw_parts(buffer as *const u8, row_bytes as usize) };
        if checksum(bytes) != reading.header.row_checksum {
            return Err(invalid(format!(
                "row record of {} failed checksum verification",
                reading.path.display()
            )));
        }
        Ok(moved)
    }

    fn close(&mut self, id: JobId) -> Result<u64> {
        reading(&mut self.records, id)?;
        self.records.remove(&id);
        Ok(0)
    }
}

fn reply(id: JobId, operation: Operation, window: u32, outcome: Result<u64>) -> Reply {
    let (result, bytes) = match outcome {
        Ok(bytes) => (Ok(()), bytes),
        Err(error) => (Err(error), 0),
    };
    Reply {
        id,
        operation,
        window,
        result,
        bytes,
        header: None,
    }
}

fn writing(open: &mut HashMap<JobId, Record>, id: JobId) -> Result<&mut Writing> {
    match open.get_mut(&id) {
        Some(Record::Writing(writing)) => Ok(writing),
        Some(Record::Reading(_)) => Err(invalid(format!(
            "disk job {} is a record open for reading, not one being written",
            id.0
        ))),
        None => Err(invalid(format!("disk job {} names no open record", id.0))),
    }
}

fn reading(open: &mut HashMap<JobId, Record>, id: JobId) -> Result<&mut Reading> {
    match open.get_mut(&id) {
        Some(Record::Reading(reading)) => Ok(reading),
        Some(Record::Writing(_)) => Err(invalid(format!(
            "disk job {} is a record being written; commit or abort it instead",
            id.0
        ))),
        None => Err(invalid(format!("disk job {} names no open record", id.0))),
    }
}

fn check_tail_pages(header: &Header) -> Result<()> {
    if header.tail_pages > MAX_TAIL_PAGES {
        return Err(invalid(format!(
            "record has {} tail pages; the disk thread handles at most {MAX_TAIL_PAGES}",
            header.tail_pages
        )));
    }
    Ok(())
}

fn page_range(header: &Header, first: u32, count: usize) -> Result<std::ops::Range<usize>> {
    let end = (first as usize)
        .checked_add(count)
        .filter(|end| *end <= header.tail_pages as usize)
        .ok_or_else(|| {
            invalid(format!(
                "page records {first}..{} lie outside the record's {} tail pages",
                first as u64 + count as u64,
                header.tail_pages
            ))
        })?;
    Ok(first as usize..end)
}

fn direct_open_failure(action: &str, path: &Path, error: io::Error) -> Error {
    if error.raw_os_error() == Some(libc::EINVAL) {
        return invalid(format!(
            "{action} {}: the filesystem refuses direct I/O (O_DIRECT)",
            path.display()
        ));
    }
    io_failure(format!("{action} {}", path.display()), error)
}

fn truncate(path: &Path, length: u64) -> Result<u64> {
    OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|file| file.set_len(length))
        .map(|()| 0)
        .map_err(|error| {
            io_failure(
                format!("truncate record file {} to {length} bytes", path.display()),
                error,
            )
        })
}

fn unlink(path: &Path) -> Result<u64> {
    fs::remove_file(path)
        .map(|()| 0)
        .map_err(|error| io_failure(format!("unlink record file {}", path.display()), error))
}

/// One contiguous direct-I/O transfer between a buffer and a file offset.
#[derive(Clone, Copy, Debug)]
struct Span {
    address: usize,
    offset: u64,
    length: usize,
}

/// A span piece of at most [`CHUNK_BYTES`], the unit the ring carries.
#[derive(Clone, Copy, Debug, Default)]
struct Chunk {
    address: usize,
    offset: u64,
    length: u32,
}

struct Chunks<'a> {
    spans: &'a [Span],
    index: usize,
    done: usize,
}

impl Iterator for Chunks<'_> {
    type Item = Chunk;

    fn next(&mut self) -> Option<Chunk> {
        loop {
            let span = self.spans.get(self.index)?;
            if self.done >= span.length {
                self.index += 1;
                self.done = 0;
                continue;
            }
            let length = (span.length - self.done).min(CHUNK_BYTES);
            let chunk = Chunk {
                address: span.address + self.done,
                offset: span.offset + self.done as u64,
                length: length as u32,
            };
            self.done += length;
            return Some(chunk);
        }
    }
}

/// The ring and the per-slot bookkeeping of chunks in flight.
struct Ring {
    ring: IoUring,
    slots: Vec<Chunk>,
    free: Vec<usize>,
    /// Set when a submission failed with chunks possibly still in flight; the
    /// ring is not used again, and its slots stay reserved until it is dropped.
    broken: Option<String>,
}

impl Ring {
    fn new() -> Result<Self> {
        let ring = IoUring::new(RING_DEPTH as u32)
            .map_err(|error| io_failure("create the disk thread's io_uring", error))?;
        Ok(Self {
            ring,
            slots: vec![Chunk::default(); RING_DEPTH],
            free: (0..RING_DEPTH).collect(),
            broken: None,
        })
    }

    /// Move every span, keeping up to [`RING_DEPTH`] chunks in flight, and
    /// return the bytes moved. Returns after every issued chunk has completed,
    /// so the caller's buffers are quiescent whatever the outcome, except when
    /// a submission fails outright: that marks the ring broken and no later
    /// transfer is attempted.
    fn transfer(&mut self, file: &File, spans: &[Span], write: bool) -> Result<u64> {
        if let Some(reason) = &self.broken {
            return Err(invalid(format!(
                "disk thread ring failed earlier: {reason}"
            )));
        }
        let verb = if write { "write" } else { "read" };
        for span in spans {
            if !aligned(span.address as u64)
                || !aligned(span.offset)
                || !aligned(span.length as u64)
            {
                return Err(invalid(format!(
                    "direct {verb} of {} bytes at file offset {} from address {:#x} is not \
                     {BLOCK_BYTES}-byte aligned",
                    span.length, span.offset, span.address
                )));
            }
        }
        let fd = types::Fd(file.as_raw_fd());
        let mut chunks = Chunks {
            spans,
            index: 0,
            done: 0,
        };
        let mut pending = chunks.next();
        let mut moved = 0u64;
        let mut failure: Option<Error> = None;
        loop {
            while failure.is_none() {
                let Some(chunk) = pending else { break };
                let Some(slot) = self.free.pop() else { break };
                let entry = if write {
                    opcode::Write::new(fd, chunk.address as *const u8, chunk.length)
                        .offset(chunk.offset)
                        .build()
                } else {
                    opcode::Read::new(fd, chunk.address as *mut u8, chunk.length)
                        .offset(chunk.offset)
                        .build()
                }
                .user_data(slot as u64);
                // SAFETY: The buffer stays valid until this function returns,
                // which happens only after the chunk's completion is consumed.
                if unsafe { self.ring.submission().push(&entry) }.is_err() {
                    self.free.push(slot);
                    break;
                }
                self.slots[slot] = chunk;
                pending = chunks.next();
            }
            if self.free.len() == RING_DEPTH {
                break;
            }
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
                    ) =>
                {
                    continue;
                }
                Err(error) => {
                    let reason = format!("submit direct {verb}: {error}");
                    self.broken = Some(reason.clone());
                    return Err(invalid(reason));
                }
            }
            for completion in self.ring.completion() {
                let slot = completion.user_data() as usize;
                let chunk = self.slots[slot];
                self.free.push(slot);
                let result = completion.result();
                if result < 0 {
                    failure.get_or_insert_with(|| {
                        io_failure(
                            format!(
                                "direct {verb} of {} bytes at file offset {}",
                                chunk.length, chunk.offset
                            ),
                            io::Error::from_raw_os_error(-result),
                        )
                    });
                } else if result as u32 != chunk.length {
                    failure.get_or_insert_with(|| {
                        invalid(format!(
                            "short direct {verb} of {result} bytes at file offset {}; {} were requested",
                            chunk.offset, chunk.length
                        ))
                    });
                } else {
                    moved += chunk.length as u64;
                }
            }
        }
        match failure {
            None => Ok(moved),
            Some(error) => Err(error),
        }
    }
}

/// Heap memory aligned to [`BLOCK_BYTES`], zero-filled, usable as a direct-I/O buffer.
struct AlignedBuffer {
    address: usize,
    length: usize,
}

// SAFETY: The buffer is plain process memory owned by whoever holds the value.
unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    fn new(length: usize) -> Result<Self> {
        let mut pointer = std::ptr::null_mut();
        // SAFETY: `posix_memalign` receives a valid out-pointer and a power-of-two alignment.
        let status = unsafe { libc::posix_memalign(&mut pointer, BLOCK_BYTES as usize, length) };
        if status != 0 {
            return Err(io_failure(
                format!("allocate {length} aligned bytes"),
                io::Error::from_raw_os_error(status),
            ));
        }
        // SAFETY: The allocation holds `length` bytes.
        unsafe { std::ptr::write_bytes(pointer as *mut u8, 0, length) };
        Ok(Self {
            address: pointer as usize,
            length,
        })
    }

    fn address(&self) -> usize {
        self.address
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: The allocation lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.address as *const u8, self.length) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: As above, with exclusive access through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.address as *mut u8, self.length) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: The pointer came from `posix_memalign` and is freed once.
        unsafe { libc::free(self.address as *mut libc::c_void) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cache::PrefixKey,
        offload::{
            geometry::tests::geometry,
            record::{file_name, temporary_name},
        },
    };
    use std::{
        io::{Seek, SeekFrom, Write as _},
        time::{Duration, Instant, SystemTime},
    };

    /// A fresh directory for one test under `FLARE_OFFLOAD_TEST_DIR` or the
    /// system temporary directory, or `None` when it refuses direct I/O.
    fn test_directory(test: &str) -> Option<PathBuf> {
        let base = std::env::var_os("FLARE_OFFLOAD_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let directory = base.join(format!("flare-offload-disk-{test}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let probe = directory.join(".probe");
        match OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_DIRECT)
            .open(&probe)
        {
            Ok(_) => {
                fs::remove_file(&probe).unwrap();
                Some(directory)
            }
            Err(error) if error.raw_os_error() == Some(libc::EINVAL) => {
                eprintln!(
                    "skipping {test}: {} refuses direct I/O",
                    directory.display()
                );
                fs::remove_dir_all(&directory).unwrap();
                None
            }
            Err(error) => panic!("probe {}: {error}", probe.display()),
        }
    }

    fn wait(thread: &DiskThread, count: usize) -> Vec<Reply> {
        let started = Instant::now();
        let mut replies = Vec::new();
        while replies.len() < count {
            thread.poll(&mut replies);
            if replies.len() >= count {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(60),
                "{} of {count} replies after 60 s",
                replies.len()
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(replies.len(), count);
        replies
    }

    fn succeeded(reply: &Reply) -> u64 {
        if let Err(error) = &reply.result {
            panic!("{:?} failed: {error}", reply.operation);
        }
        reply.bytes
    }

    fn failed(reply: &Reply) -> String {
        match &reply.result {
            Ok(()) => panic!("{:?} succeeded unexpectedly", reply.operation),
            Err(error) => error.to_string(),
        }
    }

    fn buffer(length: usize, seed: u8) -> AlignedBuffer {
        let mut buffer = AlignedBuffer::new(length).unwrap();
        for (index, byte) in buffer.as_mut_slice().iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(31).wrapping_add(seed);
        }
        buffer
    }

    fn corrupt(path: &Path, offset: u64) {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        let mut byte = [0u8; 1];
        std::io::Read::read_exact(&mut file, &mut byte).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[byte[0] ^ 0x5a]).unwrap();
    }

    fn header(key: PrefixKey, tail_pages: u32) -> Header {
        let geometry = [geometry(
            64,
            &[("k", 8192), ("v", 8192), ("sel", 100)],
            10_000,
        )];
        Header::new(
            &geometry,
            [1; 32],
            [2; 16],
            key,
            None,
            [3; 16],
            tail_pages,
            SystemTime::now(),
        )
        .unwrap()
    }

    #[test]
    fn record_round_trips_through_direct_io() {
        let Some(directory) = test_directory("round-trip") else {
            return;
        };
        let key = PrefixKey {
            consumed_tokens: 200,
            digest: [7; 16],
        };
        let header = header(key, 3);
        let page_bytes = header.page_record_bytes() as usize;
        let row_bytes = header.row_record_bytes() as usize;
        assert_eq!(page_bytes, 8192 + 8192 + 4096);
        assert_eq!(row_bytes, 12288);
        let pages: Vec<AlignedBuffer> = (0..3u8)
            .map(|index| buffer(page_bytes, index * 17 + 1))
            .collect();
        let row = buffer(row_bytes, 99);
        let temporary = directory.join(temporary_name(&key));
        let path = directory.join(file_name(&key));

        let mut thread = DiskThread::start("flare-offload-disk").unwrap();
        let writer = JobId(1);
        thread
            .submit(Job::Create {
                id: writer,
                temporary: temporary.clone(),
                header: header.clone(),
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id: writer,
                window: 0,
                first: 0,
                buffers: vec![pages[0].address(), pages[1].address()],
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id: writer,
                window: 1,
                first: 2,
                buffers: vec![pages[2].address()],
            })
            .unwrap();
        thread
            .submit(Job::WriteRow {
                id: writer,
                buffer: row.address(),
            })
            .unwrap();
        thread
            .submit(Job::Commit {
                id: writer,
                path: path.clone(),
                sync: true,
            })
            .unwrap();
        let replies = wait(&thread, 5);
        let operations: Vec<_> = replies.iter().map(|reply| reply.operation).collect();
        assert_eq!(
            operations,
            [
                Operation::Create,
                Operation::WritePages,
                Operation::WritePages,
                Operation::WriteRow,
                Operation::Commit
            ]
        );
        assert!(replies.iter().all(|reply| reply.id == writer));
        assert_eq!(succeeded(&replies[0]), 0);
        assert_eq!(succeeded(&replies[1]), 2 * page_bytes as u64);
        assert_eq!(replies[1].window, 0);
        assert_eq!(succeeded(&replies[2]), page_bytes as u64);
        assert_eq!(replies[2].window, 1);
        assert_eq!(succeeded(&replies[3]), row_bytes as u64);
        assert_eq!(
            succeeded(&replies[4]),
            3 * page_bytes as u64 + row_bytes as u64 + 2 * HEADER_BYTES as u64
        );
        assert!(!temporary.exists());
        assert_eq!(fs::metadata(&path).unwrap().len(), header.length_with_row());

        let contents = fs::read(&path).unwrap();
        let mut expected = header.clone();
        expected.row_checksum = checksum(row.as_slice());
        assert_eq!(Header::decode(&contents[..HEADER_BYTES]).unwrap(), expected);
        let page_segment = header.page_segment as usize;
        for (index, page) in pages.iter().enumerate() {
            let slot = HEADER_BYTES + index * 8;
            assert_eq!(
                u64::from_le_bytes(contents[slot..slot + 8].try_into().unwrap()),
                checksum(page.as_slice())
            );
            let start = page_segment + index * page_bytes;
            assert_eq!(&contents[start..start + page_bytes], page.as_slice());
        }
        let row_segment = header.row_segment as usize;
        assert_eq!(
            &contents[row_segment..row_segment + row_bytes],
            row.as_slice()
        );
        assert!(
            contents[HEADER_BYTES + 24..page_segment]
                .iter()
                .all(|b| *b == 0)
        );

        let reader = JobId(2);
        let fresh: Vec<AlignedBuffer> = (0..2).map(|_| buffer(page_bytes, 0xee)).collect();
        let fresh_row = buffer(row_bytes, 0xee);
        thread
            .submit(Job::Open {
                id: reader,
                path: path.clone(),
            })
            .unwrap();
        thread
            .submit(Job::ReadPages {
                id: reader,
                window: 3,
                first: 1,
                buffers: vec![fresh[0].address(), fresh[1].address()],
            })
            .unwrap();
        thread
            .submit(Job::ReadRow {
                id: reader,
                buffer: fresh_row.address(),
            })
            .unwrap();
        thread
            .submit(Job::ReadPages {
                id: reader,
                window: 4,
                first: 2,
                buffers: vec![fresh[0].address(), fresh[1].address()],
            })
            .unwrap();
        let replies = wait(&thread, 4);
        assert_eq!(replies[0].operation, Operation::Open);
        assert_eq!(succeeded(&replies[0]), 2 * HEADER_BYTES as u64);
        assert_eq!(replies[0].header, Some(expected.clone()));
        assert_eq!(succeeded(&replies[1]), 2 * page_bytes as u64);
        assert_eq!(replies[1].window, 3);
        assert_eq!(fresh[0].as_slice(), pages[1].as_slice());
        assert_eq!(fresh[1].as_slice(), pages[2].as_slice());
        assert_eq!(succeeded(&replies[2]), row_bytes as u64);
        assert_eq!(fresh_row.as_slice(), row.as_slice());
        assert!(failed(&replies[3]).contains("outside the record's 3 tail pages"));
        assert_eq!(replies[3].window, 4);

        corrupt(&path, header.page_segment + 2 * page_bytes as u64 + 17);
        thread
            .submit(Job::ReadPages {
                id: reader,
                window: 0,
                first: 1,
                buffers: vec![fresh[0].address(), fresh[1].address()],
            })
            .unwrap();
        thread
            .submit(Job::ReadPages {
                id: reader,
                window: 0,
                first: 0,
                buffers: vec![fresh[0].address(), fresh[1].address()],
            })
            .unwrap();
        let replies = wait(&thread, 2);
        assert!(failed(&replies[0]).contains("page record 2"));
        succeeded(&replies[1]);
        assert_eq!(fresh[0].as_slice(), pages[0].as_slice());
        assert_eq!(fresh[1].as_slice(), pages[1].as_slice());

        corrupt(&path, header.row_segment + 4096 + 1);
        thread
            .submit(Job::ReadRow {
                id: reader,
                buffer: fresh_row.address(),
            })
            .unwrap();
        thread.submit(Job::Close { id: reader }).unwrap();
        thread.submit(Job::Close { id: reader }).unwrap();
        let replies = wait(&thread, 3);
        assert!(failed(&replies[0]).contains("row record"));
        succeeded(&replies[1]);
        assert!(failed(&replies[2]).contains("names no open record"));

        thread
            .submit(Job::Truncate {
                id: JobId(3),
                path: path.clone(),
                length: header.row_segment,
            })
            .unwrap();
        let replies = wait(&thread, 1);
        assert_eq!(replies[0].operation, Operation::Truncate);
        succeeded(&replies[0]);
        let length = fs::metadata(&path).unwrap().len();
        assert_eq!(length, header.row_segment);
        assert!(!header.row_present(length));

        let reopened = JobId(4);
        thread
            .submit(Job::Open {
                id: reopened,
                path: path.clone(),
            })
            .unwrap();
        thread
            .submit(Job::ReadPages {
                id: reopened,
                window: 0,
                first: 0,
                buffers: vec![fresh[0].address()],
            })
            .unwrap();
        thread
            .submit(Job::ReadRow {
                id: reopened,
                buffer: fresh_row.address(),
            })
            .unwrap();
        thread.submit(Job::Close { id: reopened }).unwrap();
        thread
            .submit(Job::Unlink {
                id: JobId(5),
                path: path.clone(),
            })
            .unwrap();
        let replies = wait(&thread, 5);
        assert_eq!(replies[0].header, Some(expected));
        succeeded(&replies[1]);
        assert_eq!(fresh[0].as_slice(), pages[0].as_slice());
        assert!(failed(&replies[2]).contains("short direct read"));
        succeeded(&replies[3]);
        assert_eq!(replies[4].operation, Operation::Unlink);
        succeeded(&replies[4]);
        assert!(!path.exists());

        thread
            .submit(Job::Unlink {
                id: JobId(6),
                path: path.clone(),
            })
            .unwrap();
        thread.close().unwrap();
        assert!(thread.submit(Job::Close { id: JobId(6) }).is_err());
        let replies = wait(&thread, 1);
        assert_eq!(replies[0].operation, Operation::Unlink);
        failed(&replies[0]);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn abort_and_failed_commit_remove_the_temporary_file() {
        let Some(directory) = test_directory("abort") else {
            return;
        };
        let key = PrefixKey {
            consumed_tokens: 192,
            digest: [8; 16],
        };
        let header = header(key, 3);
        let page = buffer(header.page_record_bytes() as usize, 5);
        let temporary = directory.join(temporary_name(&key));
        let path = directory.join(file_name(&key));
        let thread = DiskThread::start("flare-offload-disk").unwrap();

        let id = JobId(1);
        thread
            .submit(Job::Create {
                id,
                temporary: temporary.clone(),
                header: header.clone(),
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id,
                window: 0,
                first: 0,
                buffers: vec![page.address()],
            })
            .unwrap();
        let replies = wait(&thread, 2);
        succeeded(&replies[0]);
        succeeded(&replies[1]);
        assert!(temporary.exists());
        thread
            .submit(Job::WritePages {
                id,
                window: 1,
                first: 1,
                buffers: vec![page.address() + 1],
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id: JobId(9),
                window: 0,
                first: 0,
                buffers: vec![page.address()],
            })
            .unwrap();
        thread.submit(Job::Close { id }).unwrap();
        thread.submit(Job::Abort { id }).unwrap();
        thread.submit(Job::Abort { id }).unwrap();
        let replies = wait(&thread, 5);
        assert!(failed(&replies[0]).contains("aligned"));
        assert!(failed(&replies[1]).contains("names no open record"));
        assert!(failed(&replies[2]).contains("being written"));
        succeeded(&replies[3]);
        assert!(!temporary.exists());
        assert!(failed(&replies[4]).contains("names no open record"));

        thread
            .submit(Job::Create {
                id,
                temporary: temporary.clone(),
                header,
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id,
                window: 0,
                first: 0,
                buffers: vec![page.address()],
            })
            .unwrap();
        thread
            .submit(Job::Commit {
                id,
                path: path.clone(),
                sync: false,
            })
            .unwrap();
        let replies = wait(&thread, 3);
        succeeded(&replies[0]);
        succeeded(&replies[1]);
        let message = failed(&replies[2]);
        assert!(message.contains("2 of 3 page records"), "{message}");
        assert!(message.contains("its row"), "{message}");
        assert!(!temporary.exists());
        assert!(!path.exists());
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn full_queue_fails_submit_without_blocking() {
        let missing = PathBuf::from("/nonexistent/flare-offload/record.ckpt");
        let (sender, jobs) = mpsc::sync_channel(QUEUE_DEPTH);
        let (_replies, receiver) = mpsc::channel();
        let (release, parked) = mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            let _jobs = jobs;
            let _ = parked.recv();
        });
        let stalled = DiskThread {
            sender: Some(sender),
            receiver,
            thread: Some(thread),
            outstanding: Cell::new(0),
        };
        for index in 0..QUEUE_DEPTH {
            stalled
                .submit(Job::Unlink {
                    id: JobId(index as u64),
                    path: missing.clone(),
                })
                .unwrap();
        }
        let error = stalled
            .submit(Job::Unlink {
                id: JobId(QUEUE_DEPTH as u64),
                path: missing.clone(),
            })
            .unwrap_err();
        assert!(error.to_string().contains("full"), "{error}");
        assert_eq!(stalled.outstanding.get(), QUEUE_DEPTH);
        drop(release);
        drop(stalled);

        let mut live = DiskThread::start("flare-offload-disk").unwrap();
        let (mut accepted, mut refused) = (0, 0);
        for index in 0..4 * QUEUE_DEPTH {
            match live.submit(Job::Unlink {
                id: JobId(index as u64),
                path: missing.clone(),
            }) {
                Ok(()) => accepted += 1,
                Err(error) => {
                    assert!(error.to_string().contains("full"), "{error}");
                    refused += 1;
                }
            }
        }
        assert_eq!(accepted + refused, 4 * QUEUE_DEPTH);
        live.close().unwrap();
        let mut replies = Vec::new();
        live.poll(&mut replies);
        assert_eq!(replies.len(), accepted);
        for reply in &replies {
            assert_eq!(reply.operation, Operation::Unlink);
            failed(reply);
        }
        assert!(live.submit(Job::Close { id: JobId(0) }).is_err());
    }

    #[test]
    fn chunks_split_spans_at_the_chunk_size() {
        let spans = [
            Span {
                address: 0x1000,
                offset: 4096,
                length: 2 * CHUNK_BYTES + 4096,
            },
            Span {
                address: 0x9000,
                offset: 0,
                length: 0,
            },
            Span {
                address: 0xa000,
                offset: 1 << 30,
                length: 4096,
            },
        ];
        let chunks: Vec<(usize, u64, u32)> = Chunks {
            spans: &spans,
            index: 0,
            done: 0,
        }
        .map(|chunk| (chunk.address, chunk.offset, chunk.length))
        .collect();
        assert_eq!(
            chunks,
            [
                (0x1000, 4096, CHUNK_BYTES as u32),
                (
                    0x1000 + CHUNK_BYTES,
                    4096 + CHUNK_BYTES as u64,
                    CHUNK_BYTES as u32
                ),
                (
                    0x1000 + 2 * CHUNK_BYTES,
                    4096 + 2 * CHUNK_BYTES as u64,
                    4096
                ),
                (0xa000, 1 << 30, 4096),
            ]
        );
        assert!(aligned(0));
        assert!(aligned(8192));
        assert!(!aligned(8193));
    }
}
