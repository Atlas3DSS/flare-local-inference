//! Bounded, direct-I/O row gathering into the same buffers used by resident tables.

use crate::Result;
use io_uring::{IoUring, opcode, types};
use std::{
    fs::{File, OpenOptions},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::Path,
};

const BLOCK: usize = 4096;
const CHUNK: usize = 64 << 10;
const DEPTH: usize = 128;

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

struct Aligned(usize);
impl Aligned {
    fn new() -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let status = unsafe { libc::posix_memalign(&mut ptr, BLOCK, DEPTH * CHUNK) };
        if status != 0 {
            return Err(format!(
                "Engram I/O staging allocation: {}",
                std::io::Error::from_raw_os_error(status)
            )
            .into());
        }
        Ok(Self(ptr as usize))
    }
    fn at(&self, slot: usize) -> *mut u8 {
        (self.0 + slot * CHUNK) as *mut u8
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { libc::free(self.0 as *mut libc::c_void) };
    }
}

pub(crate) struct DiskReader {
    // Destroying the ring cancels and drains outstanding I/O before the buffers are freed.
    ring: IoUring,
    staging: Aligned,
    file: File,
    fragments: Vec<Fragment>,
    reads: Vec<Read>,
}

impl DiskReader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .map_err(|e| format!("open Engram package for direct I/O: {e}"))?;
        let ring = IoUring::new(DEPTH as u32).map_err(|e| format!("Engram io_uring: {e}"))?;
        let staging = Aligned::new()?;
        let buffers: Vec<_> = (0..DEPTH)
            .map(|i| libc::iovec {
                iov_base: staging.at(i).cast(),
                iov_len: CHUNK,
            })
            .collect();
        unsafe { ring.submitter().register_buffers(&buffers) }
            .map_err(|e| format!("register Engram I/O staging: {e}"))?;
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
        Ok(())
    }

    fn load(&mut self, plane: Plane) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(plane.nbytes)
            .map_err(|e| format!("resident Engram table: {e}"))?;
        output.resize(plane.nbytes, 0);
        // Reuse the direct-I/O gatherer to populate resident storage without a second page-cache copy.
        let rows_per_chunk = (8 << 20) / plane.row_bytes;
        let mut begin = 0;
        while begin < plane.nbytes / plane.row_bytes {
            let end = (begin + rows_per_chunk).min(plane.nbytes / plane.row_bytes);
            let ids: Vec<_> = (begin..end).map(|i| i as i64).collect();
            self.gather(
                &[
                    plane,
                    Plane {
                        offset: 0,
                        nbytes: 0,
                        row_bytes: 0,
                    },
                ],
                &ids,
                &mut output[begin * plane.row_bytes..end * plane.row_bytes],
            )?;
            begin = end;
        }
        Ok(output)
    }
}

pub(crate) enum Reader {
    Disk(DiskReader),
    Ram([Vec<u8>; 2]),
}
impl Reader {
    pub fn open(path: &Path, planes: &[Plane; 2], resident: bool) -> Result<Self> {
        let mut disk = DiskReader::open(path)?;
        if resident {
            Ok(Self::Ram([disk.load(planes[0])?, disk.load(planes[1])?]))
        } else {
            Ok(Self::Disk(disk))
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
