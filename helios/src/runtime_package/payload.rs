//! Bounded pinned staging overlaps direct file reads with asynchronous GPU uploads.

use crate::os::FileReadAt;
use std::{
    collections::VecDeque,
    fs::File,
    io,
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
const READERS: usize = 4;
const BLOCK: usize = 4096;

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
    file: Arc<File>,
    file_offset: u64,
    destination: u64,
    nbytes: usize,
    memory: Memory,
}

impl Chunk {
    fn read_range(&self) -> (u64, usize, usize) {
        let skip = self.file_offset as usize % BLOCK;
        (
            self.file_offset - skip as u64,
            skip,
            (skip + self.nbytes).next_multiple_of(BLOCK),
        )
    }

    fn read<'a>(&self, file: &File, buffer: &'a mut [u8]) -> io::Result<&'a [u8]> {
        let (offset, skip, nbytes) = self.read_range();
        let read = loop {
            match file.read_at(&mut buffer[..nbytes], offset) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => break result?,
            }
        };
        if read < skip + self.nbytes {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short model payload read",
            ));
        }
        Ok(&buffer[skip..skip + self.nbytes])
    }
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
        let span = archive.span(package, file)?;
        let (source, mut file_offset) = archive.source(span)?;
        let file = source.direct()?;
        for [offset, nbytes] in allocation.spans() {
            for start in (0..nbytes).step_by(chunk_bytes) {
                let nbytes = chunk_bytes.min(nbytes - start);
                chunks.push(Chunk {
                    file: file.clone(),
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
    let chunk_size = chunks
        .iter()
        .map(|chunk| chunk.read_range().2)
        .max()
        .unwrap();
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
                .name(format!("flare-reader-{index}"))
                .spawn_scoped(scope, move || {
                    loop {
                        let Ok((slot, index)) = available.lock().unwrap().recv() else {
                            break;
                        };
                        let chunk = &chunks[index];
                        let result = (|| -> Result<()> {
                            let (_, _, nbytes) = chunk.read_range();
                            // SAFETY: A slot has one reader and is returned only after
                            // its upload event completes. Host chunks are disjoint
                            // validated ranges in mappings retained until scope exit.
                            let destination = unsafe {
                                std::slice::from_raw_parts_mut(buffers[slot] as *mut u8, nbytes)
                            };
                            let bytes = chunk.read(&chunk.file, destination).map_err(|error| {
                                invalid(format!(
                                    "read {} at {}: {error}",
                                    archive.path.display(),
                                    chunk.file_offset
                                ))
                            })?;
                            if matches!(chunk.memory, Memory::MappedHost) {
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        bytes.as_ptr(),
                                        chunk.destination as *mut u8,
                                        chunk.nbytes,
                                    );
                                }
                            }
                            Ok(())
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
                                staging.buffers[slot].byte_add(chunk.read_range().1),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom, Write};

    #[repr(align(4096))]
    struct Aligned([u8; 3 * BLOCK]);

    #[test]
    fn direct_reads_preserve_unaligned_payloads_and_accept_partial_final_blocks() {
        let path = std::env::temp_dir().join(format!("helios-direct-read-{}", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let reader = crate::os::direct_file(&path, false).unwrap();
        std::fs::remove_file(path).unwrap();
        let payload: Vec<_> = (0..BLOCK + 37).map(|i| (i % 251) as u8).collect();
        for offset in [0, 7, BLOCK as u64 - 1, BLOCK as u64, (1u64 << 32) + 7] {
            file.set_len(0).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&payload).unwrap();
            let chunk = Chunk {
                file: Arc::new(reader.try_clone().unwrap()),
                file_offset: offset,
                destination: 0,
                nbytes: payload.len(),
                memory: Memory::Device,
            };
            let mut buffer = Aligned([0xa5; 3 * BLOCK]);
            assert_eq!(chunk.read(&reader, &mut buffer.0).unwrap(), payload);
            file.set_len(offset + payload.len() as u64 - 1).unwrap();
            assert_eq!(
                chunk.read(&reader, &mut buffer.0).unwrap_err().kind(),
                io::ErrorKind::UnexpectedEof
            );
        }
    }
}
