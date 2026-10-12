//! Per-rank threads that move checkpoint bytes between GPU ids and host slabs.
//!
//! Each thread makes its rank's context current, owns one stream at the lowest
//! priority and one event, and receives jobs naming page ids, a state slot, and
//! slab addresses. The thread expands ids into device and host addresses from
//! its rank's geometry and the cross-rank [`Layout`], issues batched copies of
//! at most [`BATCH_BYTES`] and [`BATCH_ENTRIES`] per call, records its event,
//! waits on it, and replies. A failure is reported only after the stream has
//! been synchronized, so destination ids are quiescent when the scheduler
//! frees them. The model stream never waits on these streams: host-observed
//! completion orders later model work after the copies.

use std::time::Duration;

use crate::{Result, cache::PageId, scheduler::StateSlotId};

use super::{CacheGeometry, JobId, Layout};

pub use crate::cuda::{BATCH_BYTES, BATCH_ENTRIES};

/// Jobs a store may have outstanding per rank; `submit` fails when full.
pub const QUEUE_DEPTH: usize = 16;
pub const NICE: i32 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// GPU ids to host slabs.
    Out,
    /// Host slabs to GPU ids.
    In,
}

/// One window of a record: up to [`super::slab::WINDOW_PAGES`] pages and
/// optionally the row. `pages` pairs a GPU page id with the host address of
/// the page slab holding that page's record across ranks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    pub id: JobId,
    pub window: u32,
    pub direction: Direction,
    pub pages: Vec<(PageId, usize)>,
    pub row: Option<(StateSlotId, usize)>,
}

#[derive(Debug)]
pub struct Reply {
    pub id: JobId,
    pub window: u32,
    pub direction: Direction,
    pub result: Result<()>,
    /// Bytes copied across every rank.
    pub bytes: u64,
    /// Wall time from submission to the last rank's completion.
    pub elapsed: Duration,
}

pub struct Transfers {
    layout: Layout,
    ranks: Vec<RankThread>,
    receiver: std::sync::mpsc::Receiver<RankReply>,
    pending: std::collections::HashMap<(JobId, u32), Pending>,
}

struct RankThread {
    sender: Option<std::sync::mpsc::SyncSender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

struct RankReply {
    id: JobId,
    window: u32,
    rank: usize,
    result: Result<()>,
    bytes: u64,
}

struct Pending {
    direction: Direction,
    remaining: usize,
    bytes: u64,
    error: Option<crate::Error>,
    started: std::time::Instant,
}

impl Transfers {
    /// Start one thread per rank. `contexts` pairs each rank's CUDA context with
    /// its device ordinal, in rank order matching `geometry`.
    pub fn start(contexts: &[(usize, i32)], geometry: Vec<CacheGeometry>) -> Result<Self> {
        let _ = (contexts, geometry);
        todo!("open a lowest-priority stream per rank and spawn its thread")
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Send the job to every rank without waiting; an error means a queue is
    /// full or a thread stopped, and no reply follows.
    pub fn submit(&mut self, job: Job) -> Result<()> {
        let _ = job;
        todo!()
    }

    /// Collect replies whose every rank has completed.
    pub fn poll(&mut self, replies: &mut Vec<Reply>) {
        let _ = replies;
        todo!()
    }

    pub fn in_flight(&self) -> usize {
        self.pending.len()
    }

    /// Wait for outstanding jobs, then join every thread.
    pub fn close(&mut self) -> Result<()> {
        todo!()
    }
}

impl Drop for Transfers {
    fn drop(&mut self) {
        for rank in &mut self.ranks {
            rank.sender.take();
        }
        for rank in &mut self.ranks {
            if let Some(thread) = rank.thread.take() {
                let _ = thread.join();
            }
        }
    }
}
