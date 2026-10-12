//! A thread that moves record segments between files and caller-owned buffers
//! with direct I/O.
//!
//! The thread owns one `io_uring` ring. Files are opened with `O_DIRECT`, so
//! every buffer, offset, and length is a multiple of [`BLOCK_BYTES`], which the
//! record layout guarantees. Transfers are chunked at [`CHUNK_BYTES`]. Writes
//! hash every page record and the row as they pass through; reads verify them
//! against the record's checksum segment and header. The thread runs under
//! `SCHED_BATCH` with a positive nice value so it yields to serving threads.

use std::path::PathBuf;

use crate::Result;

use super::{JobId, geometry::BLOCK_BYTES, record::Header};

pub const CHUNK_BYTES: usize = 8 << 20;
/// Jobs the submitting store may have outstanding; `submit` fails when full.
pub const QUEUE_DEPTH: usize = 64;
pub const NICE: i32 = 10;

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
/// order; `Open` through `Close` likewise.
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
    /// `path`, and close.
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
    pub bytes: u64,
    /// The header read by `Open`.
    pub header: Option<Header>,
}

pub struct DiskThread {
    sender: Option<std::sync::mpsc::SyncSender<Job>>,
    receiver: std::sync::mpsc::Receiver<Reply>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DiskThread {
    pub fn start(name: &str) -> Result<Self> {
        let _ = name;
        todo!("spawn the thread with its ring and lower its scheduling priority")
    }

    /// Enqueue without waiting; an error means the queue is full or the thread stopped.
    pub fn submit(&self, job: Job) -> Result<()> {
        let _ = job;
        todo!()
    }

    pub fn poll(&self, replies: &mut Vec<Reply>) {
        let _ = replies;
        todo!("drain try_recv")
    }

    /// Finish outstanding jobs and join the thread.
    pub fn close(&mut self) -> Result<()> {
        todo!()
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
