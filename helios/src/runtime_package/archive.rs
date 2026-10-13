//! Small protobuf indexes and bounded reads from a single model file.

use super::{
    format::{Package, invalid},
    protobuf::wire,
};
use crate::Result;
use prost::Message;
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom, Take},
    os::{fd::AsRawFd, unix::fs::FileExt},
    path::{Path, PathBuf},
};

const MAGIC: &[u8; 8] = b"FLAREPKG";
const MAX_INDEX: usize = 16 << 20;
const MAX_RECORD: usize = 1 << 20;

pub(super) struct Archive {
    pub path: PathBuf,
    pub start: u64,
    pub file: File,
}
impl Archive {
    pub fn open(path: &Path) -> Result<(Self, wire::ModelIndex)> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let mut prefix = [0u8; 16];
        file.read_exact_at(&mut prefix, 0)
            .map_err(|e| e.to_string())?;
        if &prefix[..8] != MAGIC {
            return Err(invalid("not a Flare model package"));
        }
        let size = u64::from_le_bytes(prefix[8..].try_into().unwrap());
        if size == 0 || size > MAX_INDEX as u64 {
            return Err(invalid("invalid model index size"));
        }
        let start = (16 + size).div_ceil(4096) * 4096;
        let mut bytes = vec![0; size as usize];
        file.read_exact_at(&mut bytes, 16)
            .map_err(|e| e.to_string())?;
        let index = wire::ModelIndex::decode(bytes.as_slice())
            .map_err(|e| invalid(format!("invalid model index: {e}")))?;
        if index.version != 1 || ![1, 2, 4].contains(&index.ranks.len()) {
            return Err(invalid("unsupported model index"));
        }
        let length = file.metadata().map_err(|e| e.to_string())?.len();
        let payload = length
            .checked_sub(start)
            .ok_or_else(|| invalid("truncated model package"))?;
        for span in index
            .ranks
            .iter()
            .flat_map(|r| r.resources.values())
            .chain(index.assets.iter().filter_map(|a| a.payload.as_ref()))
            .chain(
                index
                    .engram
                    .iter()
                    .flat_map(|e| &e.tables)
                    .flat_map(|t| [&t.weights, &t.scales])
                    .filter_map(Option::as_ref),
            )
        {
            if span
                .offset
                .checked_add(span.nbytes)
                .is_none_or(|n| n > payload)
            {
                return Err(invalid("resource exceeds model package"));
            }
        }
        Ok((
            Self {
                path: path.to_owned(),
                start,
                file,
            },
            index,
        ))
    }
    pub fn span<'a>(&self, package: &'a Package, key: &str) -> Result<&'a wire::Span> {
        package
            .resources
            .get(key)
            .ok_or_else(|| invalid(format!("missing package resource {key}")))
    }
    pub fn validate(&self, package: &Package) -> Result<()> {
        for allocation in &package.allocations {
            if let Some(key) = &allocation.file {
                if self.span(package, key)?.nbytes != allocation.stored_bytes() as u64 {
                    return Err(invalid("allocation payload size differs from its spans"));
                }
            }
        }
        for module in &package.modules {
            if self.span(package, &module.file)?.nbytes < 4 {
                return Err(invalid("empty module image"));
            }
            for global in &module.globals {
                if self.span(package, &global.file)?.nbytes != global.nbytes as u64 {
                    return Err(invalid("module global payload size mismatch"));
                }
            }
        }
        for graph in &package.graphs {
            if graph
                .node_count
                .checked_add(graph.edge_count)
                .and_then(|n| n.checked_mul(4))
                .is_none_or(|n| {
                    n as u64 > self.span(package, &graph.resource).map_or(0, |s| s.nbytes)
                })
            {
                return Err(invalid(
                    "graph resource cannot contain its declared records",
                ));
            }
        }
        Ok(())
    }
    /// Drop the archive's pages from the page cache once loading has copied them out.
    ///
    /// Caches are sized from the free device memory the driver reports. An integrated GPU such as
    /// GB10 allocates from system memory, where the driver does not count cached file pages as free,
    /// so weights read through the cache would otherwise leave no room for caches.
    pub fn release_page_cache(&self) {
        // SAFETY: an advisory call on an open descriptor; failure only leaves the pages cached.
        unsafe {
            libc::posix_fadvise(self.file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
    pub fn read(&self, package: &Package, key: &str) -> Result<Vec<u8>> {
        let span = self.span(package, key)?;
        let mut bytes = vec![0; super::protobuf::size(span.nbytes)?];
        self.file
            .read_exact_at(&mut bytes, self.start + span.offset)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    }
    pub fn map(&self, package: &Package, key: &str) -> Result<Mapped> {
        let span = self.span(package, key)?;
        let offset = self.start + span.offset;
        // mmap offsets must be multiples of the host page size, which is 64 KiB on some Arm kernels.
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as u64;
        let base = offset / page * page;
        let skip = (offset - base) as usize;
        let size = super::protobuf::size(span.nbytes)?;
        let length = size
            .checked_add(skip)
            .ok_or_else(|| invalid("module mapping overflow"))?;
        if length == 0 {
            return Err(invalid("empty module image"));
        }
        // SAFETY: The index bounds this read-only mapping within the open file.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                self.file.as_raw_fd(),
                base as libc::off_t,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().to_string().into());
        }
        Ok(Mapped {
            pointer,
            length,
            skip,
            size,
        })
    }
    pub fn records(&self, package: &Package, key: &str) -> Result<Records> {
        let span = self.span(package, key)?;
        let mut file = File::open(&self.path).map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(self.start + span.offset))
            .map_err(|e| e.to_string())?;
        Ok(Records {
            reader: BufReader::with_capacity(16 << 10, file).take(span.nbytes),
            buffer: Vec::new(),
        })
    }
}

pub(super) struct Mapped {
    pointer: *mut libc::c_void,
    length: usize,
    skip: usize,
    size: usize,
}
impl Mapped {
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: The validated file range remains mapped for this borrow.
        unsafe { std::slice::from_raw_parts(self.pointer.cast::<u8>().add(self.skip), self.size) }
    }
}
impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: This object exclusively owns the mapping.
        unsafe {
            libc::munmap(self.pointer, self.length);
        }
    }
}

pub(super) struct Records {
    reader: Take<BufReader<File>>,
    buffer: Vec<u8>,
}
impl Records {
    pub fn read<T: Message + Default>(&mut self) -> Result<T> {
        let mut prefix = [0u8; 4];
        self.reader
            .read_exact(&mut prefix)
            .map_err(|e| invalid(format!("truncated graph record: {e}")))?;
        let size = u32::from_le_bytes(prefix) as usize;
        if size > MAX_RECORD || size as u64 > self.reader.limit() {
            return Err(invalid("invalid graph record size"));
        }
        self.buffer.resize(size, 0);
        self.reader
            .read_exact(&mut self.buffer)
            .map_err(|e| e.to_string())?;
        T::decode(self.buffer.as_slice()).map_err(|e| invalid(format!("invalid graph record: {e}")))
    }
    pub fn finish(self) -> Result<()> {
        if self.reader.limit() != 0 {
            return Err(invalid("unexpected trailing graph records"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new(index: &wire::ModelIndex, payload: &[u8]) -> Self {
            let path = std::env::temp_dir().join(format!(
                "flare-package-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let bytes = index.encode_to_vec();
            let mut file = File::create(&path).unwrap();
            file.write_all(MAGIC).unwrap();
            file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
            file.write_all(&bytes).unwrap();
            file.seek(SeekFrom::Start(
                (16 + bytes.len() as u64).div_ceil(4096) * 4096,
            ))
            .unwrap();
            file.write_all(payload).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    #[test]
    fn graph_records_are_read_independently_and_reject_truncation() {
        let node = wire::Node {
            operation: Some(wire::node::Operation::Empty(wire::Empty {})),
        }
        .encode_to_vec();
        let mut payload = (node.len() as u32).to_le_bytes().to_vec();
        payload.extend_from_slice(&node);
        payload.extend_from_slice(&u32::MAX.to_le_bytes());
        let mut rank = wire::BundleIndex::default();
        rank.context = Some(wire::Context::default());
        rank.architecture = vec![12, 0];
        rank.resources.insert(
            "selected".into(),
            wire::Span {
                offset: 0,
                nbytes: 4 + node.len() as u64,
            },
        );
        rank.resources.insert(
            "unselected".into(),
            wire::Span {
                offset: 4 + node.len() as u64,
                nbytes: 4,
            },
        );
        let index = wire::ModelIndex {
            version: 1,
            ranks: vec![rank],
            ..Default::default()
        };
        let fixture = Fixture::new(&index, &payload);
        let (archive, mut index) = Archive::open(&fixture.0).unwrap();
        let package = super::super::protobuf::decode(index.ranks.remove(0)).unwrap();
        let mut records = archive.records(&package, "selected").unwrap();
        assert!(records.read::<wire::Node>().unwrap().operation.is_some());
        records.finish().unwrap();
        assert!(
            archive
                .records(&package, "unselected")
                .unwrap()
                .read::<wire::Node>()
                .is_err()
        );
        let mut rank = wire::BundleIndex::default();
        rank.resources.insert(
            "overflow".into(),
            wire::Span {
                offset: u64::MAX,
                nbytes: 4,
            },
        );
        index.ranks.push(rank);
        let overflow = Fixture::new(&index, &[0]);
        assert!(Archive::open(&overflow.0).is_err());
    }
}
