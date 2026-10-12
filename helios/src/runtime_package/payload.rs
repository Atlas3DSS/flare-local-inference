//! Bounded pinned staging overlaps file reads with asynchronous GPU uploads.

use std::{
    collections::VecDeque,
    os::unix::fs::FileExt,
    ptr::null_mut,
    sync::{Arc, Mutex, mpsc},
    thread,
};

use super::{
    NativeResources,
    driver::{Driver, Handle},
    format::{Memory, Package, invalid},
};
use crate::{Result, startup::Phase};

const CHUNK: usize = 4 << 20;
const SLOTS: usize = 4;
const READERS: usize = 2;

struct Staging<'a> {
    driver: &'a Driver,
    buffers: Vec<Handle>,
    events: Vec<Handle>,
    stream: Handle,
}

impl<'a> Staging<'a> {
    unsafe fn new(driver: &'a Driver, count: usize, chunk: usize) -> Result<Self> {
        let mut result = Self {
            driver,
            buffers: vec![],
            events: vec![],
            stream: null_mut(),
        };
        // SAFETY: The caller has made the owning CUDA context current. Partial
        // construction retains every allocated resource for Drop.
        unsafe {
            driver.check(
                "cuStreamCreate",
                (driver.stream_create)(&mut result.stream, 1),
            )?;
            for _ in 0..count {
                let mut buffer = null_mut();
                driver.check(
                    "cuMemHostAlloc",
                    (driver.host_allocate)(&mut buffer, chunk, 0),
                )?;
                result.buffers.push(buffer);
                let mut event = null_mut();
                driver.check("cuEventCreate", (driver.event_create)(&mut event, 2))?;
                result.events.push(event);
            }
        }
        Ok(result)
    }
}
impl Drop for Staging<'_> {
    fn drop(&mut self) {
        // SAFETY: Readers have joined before staging is dropped. A failed drain
        // retains the buffers because an upload may still reference them.
        unsafe {
            if !self.stream.is_null() && (self.driver.stream_sync)(self.stream) != 0 {
                return;
            }
            for event in &self.events {
                (self.driver.event_destroy)(*event);
            }
            for buffer in &self.buffers {
                (self.driver.host_free)(*buffer);
            }
            if !self.stream.is_null() {
                (self.driver.stream_destroy)(self.stream);
            }
        }
    }
}

struct Chunk {
    file_offset: u64,
    destination: u64,
    nbytes: usize,
    memory: Memory,
}

/// The owner's mapped allocations must be live and unused by graph execution.
pub(super) unsafe fn load(
    owner: &NativeResources,
    archive: &super::archive::Archive,
    package: &Package,
    phase: &Phase,
    ranks: usize,
) -> Result<()> {
    let chunk_bytes = CHUNK / ranks;
    let mut chunks = Vec::new();
    for allocation in &package.allocations {
        let Some(file) = &allocation.file else {
            continue;
        };
        let mut file_offset = archive.start + archive.span(package, file)?.offset;
        for [offset, nbytes] in allocation.spans() {
            for start in (0..nbytes).step_by(chunk_bytes) {
                let nbytes = chunk_bytes.min(nbytes - start);
                chunks.push(Chunk {
                    file_offset,
                    destination: owner.allocations[&allocation.name].address
                        + (offset + start) as u64,
                    nbytes,
                    memory: allocation.memory,
                });
                file_offset += nbytes as u64;
            }
        }
    }
    if chunks.is_empty() {
        return Ok(());
    }
    let count = SLOTS.min(chunks.len());
    let chunk_size = chunks.iter().map(|chunk| chunk.nbytes).max().unwrap();
    // SAFETY: The owning context is current throughout loading and cleanup.
    let staging = unsafe { Staging::new(&owner.driver, count, chunk_size)? };
    let buffers: Vec<_> = staging
        .buffers
        .iter()
        .map(|buffer| *buffer as usize)
        .collect();
    thread::scope(|scope| -> Result<()> {
        let (work, available) = mpsc::sync_channel::<(usize, usize)>(count);
        let available = Arc::new(Mutex::new(available));
        let (ready, finished) = mpsc::sync_channel(count);
        for index in 0..READERS.min(count) {
            let (available, chunks, buffers, ready) =
                (available.clone(), &chunks, &buffers, ready.clone());
            thread::Builder::new()
                .name(format!("helios-reader-{index}"))
                .spawn_scoped(scope, move || {
                    let file = &archive.file;
                    loop {
                        let Ok((slot, index)) = available.lock().unwrap().recv() else {
                            break;
                        };
                        let chunk = &chunks[index];
                        let result = (|| -> Result<()> {
                            let address = match chunk.memory {
                                Memory::Device => buffers[slot],
                                Memory::MappedHost => chunk.destination as usize,
                            };
                            // SAFETY: A slot has one reader and is returned only after
                            // its upload event completes. Host chunks are disjoint
                            // validated ranges in mappings retained until scope exit.
                            let destination = unsafe {
                                std::slice::from_raw_parts_mut(address as *mut u8, chunk.nbytes)
                            };
                            file.read_exact_at(destination, chunk.file_offset)
                                .map_err(|e| {
                                    invalid(format!(
                                        "read {} at {}: {e}",
                                        archive.path.display(),
                                        chunk.file_offset
                                    ))
                                })
                        })();
                        if ready.send((slot, index, result)).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|e| e.to_string())?;
        }
        drop(ready);
        let mut next = 0;
        for slot in 0..count {
            work.send((slot, next)).map_err(|e| e.to_string())?;
            next += 1;
        }
        let mut inflight = VecDeque::new();
        let mut completed = 0;
        while completed < chunks.len() {
            let read = match finished.try_recv() {
                Ok(read) => Some(read),
                Err(mpsc::TryRecvError::Empty) if !inflight.is_empty() => None,
                Err(mpsc::TryRecvError::Empty) => Some(finished.recv().map_err(|e| e.to_string())?),
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(invalid("payload readers stopped"));
                }
            };
            let (slot, index) = if let Some((slot, index, result)) = read {
                result?;
                let chunk = &chunks[index];
                if matches!(chunk.memory, Memory::Device) {
                    // SAFETY: This slot remains unavailable to readers until its
                    // event has completed; the owner retains the destination.
                    unsafe {
                        owner.driver.check(
                            "cuMemcpyHtoDAsync_v2",
                            (owner.driver.upload_async)(
                                chunk.destination,
                                staging.buffers[slot],
                                chunk.nbytes,
                                staging.stream,
                            ),
                        )?;
                        owner.driver.check(
                            "cuEventRecord",
                            (owner.driver.event_record)(staging.events[slot], staging.stream),
                        )?;
                    }
                    inflight.push_back((slot, index));
                    continue;
                }
                (slot, index)
            } else {
                let (slot, index) = inflight.pop_front().unwrap();
                // SAFETY: Each event follows the upload that owns this slot.
                unsafe {
                    owner.driver.check(
                        "cuEventSynchronize",
                        (owner.driver.event_sync)(staging.events[slot]),
                    )?;
                }
                (slot, index)
            };
            completed += 1;
            phase.advance_bytes(chunks[index].nbytes as u64);
            if next < chunks.len() {
                work.send((slot, next)).map_err(|e| e.to_string())?;
                next += 1;
            }
        }
        drop(work);
        Ok(())
    })
}
