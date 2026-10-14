//! Bounded, direct-I/O row gathering into the same buffers used by resident tables.

use crate::Result;
use io_uring::{IoUring, opcode, types};
use std::{
    fs::{File, OpenOptions},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::OpenOptionsExt,
    },
    path::Path,
    sync::Arc,
};

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

struct Aligned(usize);
impl Aligned {
    fn new(nbytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let status = unsafe { libc::posix_memalign(&mut ptr, BLOCK, nbytes) };
        if status != 0 {
            return Err(format!(
                "Engram aligned allocation: {}",
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

pub(super) fn open_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .map_err(|e| format!("open Engram package for direct I/O: {e}").into())
}

pub(crate) struct Resident {
    storage: ResidentMapping,
    file: File,
    skip: usize,
    nbytes: usize,
}

struct ResidentMapping {
    address: usize,
    nbytes: usize,
}

impl ResidentMapping {
    fn new(file: &File, nbytes: usize, writable: bool) -> Result<Self> {
        let protection = libc::PROT_READ | if writable { libc::PROT_WRITE } else { 0 };
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                nbytes,
                protection,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(format!(
                "map resident embedding table: {}",
                std::io::Error::last_os_error()
            )
            .into());
        }
        Ok(Self {
            address: address as usize,
            nbytes,
        })
    }
}

impl Drop for ResidentMapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.address as *mut libc::c_void, self.nbytes) };
    }
}

const RESIDENT_SEALS: i32 =
    libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;

impl Resident {
    pub(super) fn load(file: &File, plane: Plane) -> Result<Self> {
        let skip = (plane.offset % BLOCK as u64) as usize;
        let required = skip
            .checked_add(plane.nbytes)
            .ok_or("resident Engram plane exceeds host range")?;
        let extent = required
            .checked_add(BLOCK - 1)
            .ok_or("resident Engram allocation exceeds host range")?
            / BLOCK
            * BLOCK;
        let extent = extent.max(BLOCK);
        let descriptor = unsafe {
            libc::memfd_create(
                c"flare-embedding-table".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        if descriptor < 0 {
            return Err(format!(
                "create resident embedding table: {}",
                std::io::Error::last_os_error()
            )
            .into());
        }
        // The descriptor is newly created and owned exclusively until published.
        let backing = unsafe { File::from_raw_fd(descriptor) };
        backing
            .set_len(extent as u64)
            .map_err(|e| format!("size resident embedding table: {e}"))?;
        let storage = ResidentMapping::new(&backing, extent, true)?;
        let offset = plane.offset - skip as u64;
        std::thread::scope(|scope| -> Result<()> {
            let mut workers = Vec::new();
            for reader in 0..LOAD_READERS.min(extent.div_ceil(LOAD_CHUNK)) {
                let address = storage.address;
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
                                    .and_then(|n| libc::off_t::try_from(n).ok())
                                    .ok_or("resident Engram file offset exceeds range")?;
                                let count = loop {
                                    // SAFETY: Workers own disjoint aligned ranges. Raw I/O
                                    // initializes storage before any byte slice is exposed.
                                    let count = unsafe {
                                        libc::pread(
                                            file.as_raw_fd(),
                                            (address + begin) as *mut libc::c_void,
                                            nbytes,
                                            source,
                                        )
                                    };
                                    if count >= 0 {
                                        break count as usize;
                                    }
                                    let error = std::io::Error::last_os_error();
                                    if error.kind() != std::io::ErrorKind::Interrupted {
                                        return Err(format!("resident Engram read at {source}: {error}").into());
                                    }
                                };
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
        drop(storage);
        if unsafe { libc::fcntl(backing.as_raw_fd(), libc::F_ADD_SEALS, RESIDENT_SEALS) } < 0 {
            return Err(format!(
                "seal resident embedding table: {}",
                std::io::Error::last_os_error()
            )
            .into());
        }
        Self::from_shared(backing, plane)
    }

    pub(super) fn from_shared(file: File, plane: Plane) -> Result<Self> {
        let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        if seals < 0 || seals & RESIDENT_SEALS != RESIDENT_SEALS {
            return Err("resident embedding table must have immutable size and contents".into());
        }
        let skip = (plane.offset % BLOCK as u64) as usize;
        let required = skip
            .checked_add(plane.nbytes)
            .ok_or("resident embedding plane exceeds host range")?;
        let extent = required
            .checked_add(BLOCK - 1)
            .ok_or("resident embedding plane exceeds host range")?
            / BLOCK
            * BLOCK;
        let extent = extent.max(BLOCK);
        if file
            .metadata()
            .map_err(|e| format!("inspect resident embedding table: {e}"))?
            .len()
            != extent as u64
        {
            return Err("shared resident embedding table extent differs from its plane".into());
        }
        Ok(Self {
            storage: ResidentMapping::new(&file, extent, false)?,
            file,
            skip,
            nbytes: plane.nbytes,
        })
    }

    pub(super) fn descriptor(&self) -> i32 {
        self.file.as_raw_fd()
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: Every logical byte was initialized by successful reads, and
        // scoped readers joined before this immutable view becomes available.
        unsafe {
            std::slice::from_raw_parts((self.storage.address + self.skip) as *const u8, self.nbytes)
        }
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
        let file = open_file(path)?;
        let ring = IoUring::new(DEPTH as u32).map_err(|e| format!("Engram io_uring: {e}"))?;
        let staging = Aligned::new(DEPTH * CHUNK)?;
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
}

pub(crate) enum Reader {
    Disk(DiskReader),
    Ram([Arc<Resident>; 2]),
}
impl Reader {
    pub fn open(path: &Path, planes: &[Plane; 2], resident: bool) -> Result<Self> {
        if resident {
            let file = open_file(path)?;
            Ok(Self::Ram([
                Arc::new(Resident::load(&file, planes[0])?),
                Arc::new(Resident::load(&file, planes[1])?),
            ]))
        } else {
            Ok(Self::Disk(DiskReader::open(path)?))
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

    #[test]
    fn shared_resident_plane_outlives_source_and_loader_and_rejects_mutation() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-shared-{}.bin", std::process::id()));
        let bytes: Vec<_> = (0..12345).map(|i| (i * 19 + i / 7) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let plane = Plane {
            offset: 97,
            nbytes: 8192,
            row_bytes: 128,
        };
        let original = Resident::load(&open_file(&path).unwrap(), plane).unwrap();
        let peer = Resident::from_shared(original.file.try_clone().unwrap(), plane).unwrap();
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            original.file.metadata().unwrap().ino(),
            peer.file.metadata().unwrap().ino()
        );
        assert!(original.file.set_len(1).is_err());
        let value = 255u8;
        assert_eq!(
            unsafe { libc::pwrite(original.descriptor(), (&value as *const u8).cast(), 1, 97) },
            -1
        );
        let wrong = Plane {
            nbytes: plane.nbytes + BLOCK,
            ..plane
        };
        assert!(Resident::from_shared(original.file.try_clone().unwrap(), wrong).is_err());
        std::fs::remove_file(path).unwrap();
        drop(original);
        assert_eq!(peer.bytes(), &bytes[97..97 + 8192]);
    }

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
