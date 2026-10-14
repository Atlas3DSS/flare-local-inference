//! Bounded, direct-I/O row gathering into the same buffers used by resident tables.

use crate::Result;
#[cfg(unix)]
use io_uring::{IoUring, opcode, types};
use std::fs::File;

#[cfg(unix)]
use std::os::fd::AsRawFd;

#[cfg(test)]
use std::path::Path;

const BLOCK: usize = 4096;
const CHUNK: usize = 64 << 10;
const DEPTH: usize = 128;
const LOAD_CHUNK: usize = 4 << 20;
const LOAD_READERS: usize = 4;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Plane {
    pub offset: u64,
    pub nbytes: usize,
    pub row_bytes: usize,
}

#[derive(Clone, Copy)]
struct Fragment {
    source: u64,
    destination: usize,
    nbytes: usize,
}

struct Read {
    offset: u64,
    nbytes: usize,
    required: usize,
    fragments: std::ops::Range<usize>,
}

struct Aligned(usize, usize);
impl Aligned {
    fn new(nbytes: usize) -> Result<Self> {
        Ok(Self(
            crate::os::allocate_aligned(nbytes).map_err(|e| format!("Engram allocation: {e}"))?,
            nbytes,
        ))
    }
    fn at(&self, slot: usize) -> *mut u8 {
        (self.0 + slot * CHUNK) as *mut u8
    }
}

#[cfg(test)]
fn open_file(path: &Path) -> Result<File> {
    crate::os::direct_file(path, false).map_err(|e| format!("open Engram package: {e}").into())
}

pub(crate) struct Resident {
    storage: Aligned,
    skip: usize,
    nbytes: usize,
}

impl Resident {
    fn load(file: &File, plane: Plane) -> Result<Self> {
        let skip = (plane.offset % BLOCK as u64) as usize;
        let required = skip
            .checked_add(plane.nbytes)
            .ok_or("resident Engram plane exceeds host range")?;
        let extent = required
            .checked_add(BLOCK - 1)
            .ok_or("resident Engram allocation exceeds host range")?
            / BLOCK
            * BLOCK;
        let storage = Aligned::new(extent.max(BLOCK))?;
        let offset = plane.offset - skip as u64;
        std::thread::scope(|scope| -> Result<()> {
            let mut workers = Vec::new();
            for reader in 0..LOAD_READERS.min(extent.div_ceil(LOAD_CHUNK)) {
                let address = storage.0;
                workers.push(
                    std::thread::Builder::new()
                        .name(format!("helios-table-loader-{reader}"))
                        .spawn_scoped(scope, move || -> Result<()> {
                            for begin in (reader * LOAD_CHUNK..extent)
                                .step_by(LOAD_READERS * LOAD_CHUNK)
                            {
                                let nbytes = LOAD_CHUNK.min(extent - begin);
                                let source = offset
                                    .checked_add(begin as u64)
                                    .ok_or("resident Engram file offset exceeds range")?;
                                // SAFETY: Each reader initializes a disjoint aligned allocation range.
                                let count = unsafe { crate::os::read_at_raw(file, (address + begin) as *mut u8, nbytes, source) }
                                    .map_err(|e| format!("resident Engram read at {source}: {e}"))?;
                                let needed = nbytes.min(required - begin);
                                if count < needed {
                                    return Err(format!(
                                        "resident Engram read at {source} returned {count}, requires {needed} bytes"
                                    ).into());
                                }
                            }
                            Ok(())
                        })
                        .map_err(|e| format!("start resident Engram loader: {e}"))?,
                );
            }
            for worker in workers {
                worker
                    .join()
                    .map_err(|_| "resident Engram loader panicked")??;
            }
            Ok(())
        })?;
        Ok(Self {
            storage,
            skip,
            nbytes: plane.nbytes,
        })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: Every logical byte was initialized by successful reads, and
        // scoped readers joined before this immutable view becomes available.
        unsafe {
            std::slice::from_raw_parts((self.storage.0 + self.skip) as *const u8, self.nbytes)
        }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { crate::os::free_aligned(self.0, self.1) };
    }
}

pub(crate) struct DiskReader {
    // Destroying the ring cancels and drains outstanding I/O before the buffers are freed.
    #[cfg(unix)]
    ring: IoUring,
    #[cfg(windows)]
    ring: crate::os::completion::Queue,
    staging: Aligned,
    file: File,
    fragments: Vec<Fragment>,
    reads: Vec<Read>,
}

impl DiskReader {
    fn from_file(source: &File) -> Result<Self> {
        let file = crate::os::direct_reopen(source)
            .map_err(|e| format!("open Engram direct reader: {e}"))?;
        let staging = Aligned::new(DEPTH * CHUNK)?;
        #[cfg(unix)]
        let ring = {
            let ring = IoUring::new(DEPTH as u32).map_err(|e| format!("Engram io_uring: {e}"))?;
            let buffers: Vec<_> = (0..DEPTH)
                .map(|i| libc::iovec {
                    iov_base: staging.at(i).cast(),
                    iov_len: CHUNK,
                })
                .collect();
            unsafe { ring.submitter().register_buffers(&buffers) }
                .map_err(|e| format!("register Engram staging: {e}"))?;
            ring
        };
        #[cfg(windows)]
        let ring = crate::os::completion::Queue::new(DEPTH)
            .map_err(|e| format!("Engram completion port: {e}"))?;
        Ok(Self {
            ring,
            staging,
            file,
            fragments: Vec::new(),
            reads: Vec::new(),
        })
    }

    pub fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        self.fragments.clear();
        let mut base = 0;
        for plane in planes {
            for (row, id) in ids.iter().copied().enumerate() {
                let destination = base + row * plane.row_bytes;
                if id < 0 {
                    output[destination..destination + plane.row_bytes].fill(0);
                    continue;
                }
                let source = plane.offset + id as u64 * plane.row_bytes as u64;
                let mut copied = 0;
                while copied < plane.row_bytes {
                    let address = source + copied as u64;
                    let nbytes = (plane.row_bytes - copied).min(BLOCK - address as usize % BLOCK);
                    self.fragments.push(Fragment {
                        source: address,
                        destination: destination + copied,
                        nbytes,
                    });
                    copied += nbytes;
                }
            }
            base += ids.len() * plane.row_bytes;
        }
        self.fragments.sort_unstable_by_key(|f| f.source);
        self.reads.clear();
        for (i, fragment) in self.fragments.iter().enumerate() {
            let block = fragment.source / BLOCK as u64 * BLOCK as u64;
            if let Some(read) = self.reads.last_mut().filter(|r| {
                block <= r.offset + r.nbytes as u64
                    && block + BLOCK as u64 <= r.offset + CHUNK as u64
            }) {
                read.nbytes = (block - read.offset) as usize + BLOCK;
                read.required = read
                    .required
                    .max((fragment.source - read.offset) as usize + fragment.nbytes);
                read.fragments.end = i + 1;
            } else {
                self.reads.push(Read {
                    offset: block,
                    nbytes: BLOCK,
                    required: (fragment.source - block) as usize + fragment.nbytes,
                    fragments: i..i + 1,
                });
            }
        }
        #[cfg(unix)]
        for batch in self.reads.chunks(DEPTH) {
            for (slot, read) in batch.iter().enumerate() {
                let entry = opcode::ReadFixed::new(
                    types::Fd(self.file.as_raw_fd()),
                    self.staging.at(slot),
                    read.nbytes as u32,
                    slot as u16,
                )
                .offset(read.offset)
                .build()
                .user_data(slot as u64);
                unsafe { self.ring.submission().push(&entry) }
                    .map_err(|_| "Engram submission queue full")?;
            }
            let mut completed = 0;
            let mut failure = None;
            while completed < batch.len() {
                match self.ring.submit_and_wait(1) {
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(format!("submit Engram reads: {error}").into()),
                }
                for entry in self.ring.completion() {
                    completed += 1;
                    let slot = entry.user_data() as usize;
                    let read = &batch[slot];
                    let count = entry.result();
                    if count < 0 || (count as usize) < read.required {
                        failure = Some(format!(
                            "Engram read at {} returned {count}, requires {} bytes",
                            read.offset, read.required
                        ));
                        continue;
                    }
                    for fragment in &self.fragments[read.fragments.clone()] {
                        let source = unsafe {
                            std::slice::from_raw_parts(
                                self.staging
                                    .at(slot)
                                    .add((fragment.source - read.offset) as usize),
                                fragment.nbytes,
                            )
                        };
                        output[fragment.destination..fragment.destination + fragment.nbytes]
                            .copy_from_slice(source);
                    }
                }
            }
            if let Some(error) = failure {
                return Err(error.into());
            }
        }
        #[cfg(windows)]
        for batch in self.reads.chunks(DEPTH) {
            let mut failure = None;
            for (slot, read) in batch.iter().enumerate() {
                if let Err(error) = unsafe {
                    self.ring.submit(
                        slot,
                        &self.file,
                        self.staging.at(slot) as usize,
                        read.nbytes as u32,
                        read.offset,
                        false,
                    )
                } {
                    failure = Some(format!("submit Engram read: {error}"));
                    self.ring.cancel();
                    break;
                }
            }
            while self.ring.pending() != 0 {
                match self.ring.wait() {
                    Err(error) => {
                        failure.get_or_insert_with(|| format!("wait for Engram read: {error}"));
                        self.ring.cancel();
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Ok((slot, result)) => {
                        let read = &batch[slot];
                        match result {
                            Ok(count) if count >= read.required => {
                                for fragment in &self.fragments[read.fragments.clone()] {
                                    let source = unsafe {
                                        std::slice::from_raw_parts(
                                            self.staging
                                                .at(slot)
                                                .add((fragment.source - read.offset) as usize),
                                            fragment.nbytes,
                                        )
                                    };
                                    output[fragment.destination
                                        ..fragment.destination + fragment.nbytes]
                                        .copy_from_slice(source);
                                }
                            }
                            result => {
                                failure.get_or_insert_with(|| {
                                    format!(
                                        "Engram read at {} requires {} bytes: {result:?}",
                                        read.offset, read.required
                                    )
                                });
                            }
                        }
                    }
                }
            }
            if let Some(error) = failure {
                return Err(error.into());
            }
        }

        Ok(())
    }
}

pub(crate) enum Reader {
    Disk(DiskReader),
    Ram([Resident; 2]),
}
impl Reader {
    #[cfg(test)]
    fn open(path: &Path, planes: &[Plane; 2], resident: bool) -> Result<Self> {
        Self::from_file(&File::open(path).unwrap(), planes, resident)
    }
    pub fn from_file(source: &File, planes: &[Plane; 2], resident: bool) -> Result<Self> {
        if resident {
            let file = crate::os::direct_reopen(source)
                .map_err(|e| format!("open resident Engram reader: {e}"))?;
            Ok(Self::Ram([
                Resident::load(&file, planes[0])?,
                Resident::load(&file, planes[1])?,
            ]))
        } else {
            Ok(Self::Disk(DiskReader::from_file(source)?))
        }
    }
    pub fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        if output.len() != ids.len() * planes.iter().map(|p| p.row_bytes).sum::<usize>() {
            return Err("Engram staging size differs from row count".into());
        }
        for plane in planes {
            if ids
                .iter()
                .any(|id| *id < -1 || (*id >= 0 && *id as usize >= plane.nbytes / plane.row_bytes))
            {
                return Err("Engram row exceeds its table".into());
            }
        }
        match self {
            Self::Disk(reader) => reader.gather(planes, ids, output),
            Self::Ram(data) => {
                let mut base = 0;
                for (plane, data) in planes.iter().zip(data) {
                    let data = data.bytes();
                    for (row, id) in ids.iter().copied().enumerate() {
                        let target = &mut output
                            [base + row * plane.row_bytes..base + (row + 1) * plane.row_bytes];
                        if id < 0 {
                            target.fill(0);
                        } else {
                            let start = id as usize * plane.row_bytes;
                            target.copy_from_slice(&data[start..start + plane.row_bytes]);
                        }
                    }
                    base += ids.len() * plane.row_bytes;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;

    #[test]
    fn resident_load_covers_parallel_chunks_and_rejects_truncated_planes() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-bulk-{}.bin", std::process::id()));
        let rows = LOAD_CHUNK * 5 / 80 + 1;
        let first = 123;
        let second = first + rows * 80 + 37;
        let bytes: Vec<_> = (0..second + rows * 10)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                offset: first as u64,
                nbytes: rows * 80,
                row_bytes: 80,
            },
            Plane {
                offset: second as u64,
                nbytes: rows * 10,
                row_bytes: 10,
            },
        ];
        let file = open_file(&path).unwrap();
        for plane in planes {
            let data = Resident::load(&file, plane).unwrap();
            let begin = plane.offset as usize;
            assert_eq!(data.bytes(), &bytes[begin..begin + plane.nbytes]);
        }
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((first + LOAD_CHUNK + 17) as u64)
            .unwrap();
        assert!(
            Resident::load(&file, planes[0])
                .err()
                .unwrap()
                .to_string()
                .contains("requires")
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn disk_and_ram_gather_identical_rows_across_unaligned_planes() {
        let path = std::env::temp_dir().join(format!("helios-engram-{}.bin", std::process::id()));
        let rows = 4097;
        let first = 4093;
        let second = first + rows * 256 + 7;
        let bytes: Vec<_> = (0..second + rows * 8)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                offset: first as u64,
                nbytes: rows * 256,
                row_bytes: 256,
            },
            Plane {
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        let ids: Vec<_> = (0..8192)
            .map(|i| {
                if i % 19 == 0 {
                    -1
                } else {
                    (i * 101 % rows) as i64
                }
            })
            .collect();
        let mut expected = Vec::new();
        for plane in &planes {
            for id in &ids {
                if *id == -1 {
                    expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                } else {
                    let offset = plane.offset as usize + *id as usize * plane.row_bytes;
                    expected.extend_from_slice(&bytes[offset..offset + plane.row_bytes]);
                }
            }
        }
        for resident in [false, true] {
            let mut reader = Reader::open(&path, &planes, resident).unwrap();
            for _ in 0..3 {
                let mut actual = vec![0xA5; expected.len()];
                reader.gather(&planes, &ids, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            assert!(
                reader
                    .gather(&planes, &[rows as i64], &mut [0; 264])
                    .is_err()
            );
        }
        let mut reader = Reader::open(&path, &planes, false).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(4096)
            .unwrap();
        assert!(reader.gather(&planes, &[100], &mut [0; 264]).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
