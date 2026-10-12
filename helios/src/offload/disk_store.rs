//! Records retained as files, moved by a staged or direct transport.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use crate::{Result, cache::PrefixHistory};

use super::{
    HostHit, InboundTicket, PrefixStore, PublishedCheckpoint, StoreEvent, StoreStats, Transport,
    disk::DiskThread, index::Index, record::Header, slab::SlabPool, transfer::Transfers,
};

/// A record file as the disk index sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskRecord {
    pub path: PathBuf,
    pub header: Header,
    pub bytes: u64,
}

/// Completion the transport reports to the disk store.
#[derive(Debug)]
pub enum TransportEvent {
    Stored {
        id: super::JobId,
        result: Result<u64>,
    },
    Fetched {
        id: super::JobId,
        result: Result<u64>,
    },
    Truncated {
        id: super::JobId,
        result: Result<()>,
    },
    Unlinked {
        id: super::JobId,
        result: Result<()>,
    },
}

/// Moves whole records between GPU ids and record files.
pub trait DiskTransport {
    fn kind(&self) -> Transport;
    /// Stream a published GPU checkpoint into a new record file at `temporary`,
    /// renamed to `path` on completion.
    fn store(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        checkpoint: &PublishedCheckpoint,
    ) -> Result<()>;
    /// Stream the chain into the ticket's ids: for each record, the page
    /// records `first..first + count` and, for the last record, the row.
    fn fetch(
        &mut self,
        id: super::JobId,
        plan: Vec<FetchSegment>,
        ticket: &InboundTicket,
    ) -> Result<()>;
    fn truncate(&mut self, id: super::JobId, path: PathBuf, length: u64) -> Result<()>;
    fn unlink(&mut self, id: super::JobId, path: PathBuf) -> Result<()>;
    fn poll(&mut self, events: &mut Vec<TransportEvent>);
    fn close(&mut self) -> Result<()>;
}

/// One record's contribution to a fetch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchSegment {
    pub path: PathBuf,
    pub header: Header,
    pub first_page: u32,
    pub pages: u32,
    pub row: bool,
}

/// The staged transport: record files move through a non-retaining staging
/// pool and the shared transfer threads. It also moves records between files
/// and caller-owned registered buffers for the composite store.
pub struct Staged {
    /// Absent inside a composite store, whose memory slabs are the buffers.
    pool: Option<SlabPool>,
    transfers: Arc<Mutex<Transfers>>,
    disk: DiskThread,
}

impl Staged {
    pub fn new(pool: Option<SlabPool>, transfers: Arc<Mutex<Transfers>>, disk: DiskThread) -> Self {
        Self {
            pool,
            transfers,
            disk,
        }
    }

    pub fn pool(&self) -> Option<&SlabPool> {
        self.pool.as_ref()
    }

    /// Write a record whose page and row bytes already sit in caller-owned
    /// registered buffers (host addresses in record order).
    pub fn write_from(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        pages: Vec<usize>,
        row: Option<usize>,
    ) -> Result<()> {
        let _ = (id, header, temporary, path, pages, row);
        todo!()
    }

    /// Read a record's page records `first..first + buffers.len()` and row into
    /// caller-owned registered buffers.
    pub fn read_into(
        &mut self,
        id: super::JobId,
        path: PathBuf,
        first_page: u32,
        pages: Vec<usize>,
        row: Option<usize>,
    ) -> Result<()> {
        let _ = (id, path, first_page, pages, row);
        todo!()
    }
}

impl DiskTransport for Staged {
    fn kind(&self) -> Transport {
        Transport::Staged
    }

    fn store(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        checkpoint: &PublishedCheckpoint,
    ) -> Result<()> {
        let _ = (id, header, temporary, path, checkpoint);
        todo!()
    }

    fn fetch(
        &mut self,
        id: super::JobId,
        plan: Vec<FetchSegment>,
        ticket: &InboundTicket,
    ) -> Result<()> {
        let _ = (id, plan, ticket);
        todo!()
    }

    fn truncate(&mut self, id: super::JobId, path: PathBuf, length: u64) -> Result<()> {
        let _ = (id, path, length);
        todo!()
    }

    fn unlink(&mut self, id: super::JobId, path: PathBuf) -> Result<()> {
        let _ = (id, path);
        todo!()
    }

    fn poll(&mut self, events: &mut Vec<TransportEvent>) {
        let _ = events;
        todo!()
    }

    fn close(&mut self) -> Result<()> {
        todo!()
    }
}

pub struct DiskStore {
    index: Index<DiskRecord>,
    directory: PathBuf,
    identity: [u8; 32],
    fingerprint: [u8; 16],
    limit: u64,
    used: u64,
    transport: Box<dyn DiskTransport>,
    stats: StoreStats,
    next_job: u64,
}

impl DiskStore {
    /// Open a store over `directory`, already scanned by
    /// [`super::record::scan`], whose records are inserted parents first.
    pub fn new(
        directory: PathBuf,
        identity: [u8; 32],
        fingerprint: [u8; 16],
        limit: u64,
        page_tokens: u64,
        scanned: Vec<super::record::Scanned>,
        transport: Box<dyn DiskTransport>,
    ) -> Result<Self> {
        let _ = scanned;
        Ok(Self {
            index: Index::new(page_tokens),
            directory,
            identity,
            fingerprint,
            limit,
            used: 0,
            transport,
            stats: StoreStats::default(),
            next_job: 1,
        })
    }

    pub fn index(&self) -> &Index<DiskRecord> {
        &self.index
    }

    pub fn directory(&self) -> &PathBuf {
        &self.directory
    }

    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    pub fn fingerprint(&self) -> [u8; 16] {
        self.fingerprint
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn transport(&self) -> Transport {
        self.transport.kind()
    }

    pub(crate) fn next_job(&mut self) -> super::JobId {
        let id = super::JobId(self.next_job);
        self.next_job += 1;
        id
    }
}

impl PrefixStore for DiskStore {
    fn lookup(&mut self, history: &PrefixHistory, resident: u64) -> Option<HostHit> {
        let _ = (history, resident);
        todo!()
    }

    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<bool> {
        let _ = checkpoint;
        todo!()
    }

    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()> {
        let _ = (hit, ticket);
        todo!()
    }

    fn poll(&mut self, events: &mut Vec<StoreEvent>) {
        let _ = events;
        todo!()
    }

    fn stats(&self) -> StoreStats {
        self.stats
    }

    fn close(&mut self) -> Result<()> {
        todo!()
    }
}
