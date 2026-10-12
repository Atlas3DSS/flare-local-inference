//! Memory and disk stores composed with write-through and read-through.
//!
//! Every stored checkpoint lands in the memory store and is written through to
//! disk; the memory record becomes durable when the write completes. Lookups
//! run on the disk index, a superset of the memory index including writes in
//! flight. A fetch uploads memory-resident records from their slabs and reads
//! other records from disk into memory slabs first, so they become residents.
//! Memory eviction is restricted to durable records, and disk eviction removes
//! the memory copy, so memory contents remain a subset of disk contents.

use crate::{Result, cache::PrefixHistory};

use super::{
    HostHit, InboundTicket, PrefixStore, PublishedCheckpoint, StoreEvent, StoreStats,
    disk_store::DiskStore, memory::MemoryStore,
};

pub struct CompositeStore {
    memory: MemoryStore,
    disk: DiskStore,
    stats: StoreStats,
}

impl CompositeStore {
    pub fn new(memory: MemoryStore, disk: DiskStore) -> Result<Self> {
        if memory.identity() != disk.identity() {
            return Err("composite store tiers belong to different models".into());
        }
        Ok(Self {
            memory,
            disk,
            stats: StoreStats::default(),
        })
    }

    pub fn memory(&self) -> &MemoryStore {
        &self.memory
    }

    pub fn disk(&self) -> &DiskStore {
        &self.disk
    }
}

impl PrefixStore for CompositeStore {
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
