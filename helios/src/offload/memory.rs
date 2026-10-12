//! Records retained in registered host memory.

use std::sync::{Arc, Mutex};

use crate::{Result, cache::PrefixHistory};

use super::{
    HostHit, InboundTicket, PrefixStore, PublishedCheckpoint, StoreEvent, StoreStats,
    index::Index,
    slab::{PageSlab, RowSlab, SlabPool},
    transfer::Transfers,
};

/// Slabs holding one record. `durable` is set by the composite once the disk
/// copy is complete; a standalone memory store treats every record as durable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRecord {
    pub pages: Vec<PageSlab>,
    pub row: Option<RowSlab>,
    pub durable: bool,
}

pub struct MemoryStore {
    index: Index<MemoryRecord>,
    pool: SlabPool,
    transfers: Arc<Mutex<Transfers>>,
    identity: [u8; 32],
    stats: StoreStats,
    next_job: u64,
}

impl MemoryStore {
    pub fn new(
        identity: [u8; 32],
        pool: SlabPool,
        transfers: Arc<Mutex<Transfers>>,
    ) -> Result<Self> {
        let page_tokens = transfers.lock().unwrap().layout().page_tokens;
        Ok(Self {
            index: Index::new(page_tokens),
            pool,
            transfers,
            identity,
            stats: StoreStats::default(),
            next_job: 1,
        })
    }

    pub fn index(&self) -> &Index<MemoryRecord> {
        &self.index
    }

    pub fn pool(&self) -> &SlabPool {
        &self.pool
    }

    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    pub(crate) fn next_job(&mut self) -> super::JobId {
        let id = super::JobId(self.next_job);
        self.next_job += 1;
        id
    }
}

impl PrefixStore for MemoryStore {
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
