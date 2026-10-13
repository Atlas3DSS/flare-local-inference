//! Bounded protobuf indexes and resource reads inside a safetensors container.

use super::{
    format::{Package, invalid},
    protobuf::wire,
};
use crate::Result;
use prost::Message;
use serde::Deserialize;
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom, Take},
    os::{fd::AsRawFd, unix::fs::FileExt},
    path::{Path, PathBuf},
};

const MAX_HEADER: usize = 64 << 10;
const MAX_INDEX: usize = 16 << 20;
const MAX_RECORD: usize = 1 << 20;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    #[serde(rename = "__metadata__")]
    format: Format,
    #[serde(rename = "flare.payload")]
    payload: ByteTensor,
    #[serde(rename = "flare.metadata")]
    index: ByteTensor,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Format {
    format: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ByteTensor {
    dtype: String,
    shape: [u64; 1],
    data_offsets: [u64; 2],
}
impl ByteTensor {
    fn size(&self) -> Result<u64> {
        let [begin, end] = self.data_offsets;
        if self.dtype != "U8" || end.checked_sub(begin) != Some(self.shape[0]) {
            return Err(invalid("invalid package byte tensor"));
        }
        Ok(self.shape[0])
    }
}

pub(super) struct Archive {
    pub path: PathBuf,
    pub start: u64,
    pub file: File,
}
impl Archive {
    pub fn open(path: &Path) -> Result<(Self, wire::ModelIndex)> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let mut prefix = [0u8; 8];
        file.read_exact_at(&mut prefix, 0)
            .map_err(|e| e.to_string())?;
        let header_size = u64::from_le_bytes(prefix);
        if header_size == 0 || header_size > MAX_HEADER as u64 {
            return Err(invalid("invalid safetensors header size"));
        }
        let start = 8 + header_size;
        let mut header = vec![0; header_size as usize];
        file.read_exact_at(&mut header, 8)
            .map_err(|e| e.to_string())?;
        if header[0] != b'{' || start % 4096 != 0 {
            return Err(invalid("invalid or unaligned safetensors header"));
        }
        let header: Header = serde_json::from_slice(&header)
            .map_err(|e| invalid(format!("invalid safetensors header: {e}")))?;
        if header.format.format != "flare" {
            return Err(invalid("not a Flare safetensors package"));
        }
        let payload = header.payload.size()?;
        let size = header.index.size()?;
        if size == 0 || size > MAX_INDEX as u64 {
            return Err(invalid("invalid model index size"));
        }
        if header.payload.data_offsets[0] != 0 || header.index.data_offsets[0] != payload {
            return Err(invalid(
                "package tensors must cover the data buffer without gaps or overlap",
            ));
        }
        let length = file.metadata().map_err(|e| e.to_string())?.len();
        if length.checked_sub(start) != Some(header.index.data_offsets[1]) {
            return Err(invalid(
                "safetensors data length differs from model package",
            ));
        }
        let mut bytes = vec![0; size as usize];
        file.read_exact_at(&mut bytes, start + payload)
            .map_err(|e| e.to_string())?;
        let index = wire::ModelIndex::decode(bytes.as_slice())
            .map_err(|e| invalid(format!("invalid model index: {e}")))?;
        if index.version != 1 || ![1, 2, 4].contains(&index.ranks.len()) {
            return Err(invalid("unsupported model index"));
        }
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
            let bytes = index.encode_to_vec();
            let header = serde_json::json!({
                "__metadata__": {"format": "flare"},
                "flare.payload": {"dtype": "U8", "shape": [payload.len()], "data_offsets": [0, payload.len()]},
                "flare.metadata": {"dtype": "U8", "shape": [bytes.len()], "data_offsets": [payload.len(), payload.len() + bytes.len()]},
            });
            Self::with_header(&header, &bytes, payload)
        }
        fn with_header(header: &serde_json::Value, index: &[u8], payload: &[u8]) -> Self {
            let path = std::env::temp_dir().join(format!(
                "flare-package-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let mut bytes = serde_json::to_vec(header).unwrap();
            bytes.resize((8 + bytes.len()).div_ceil(4096) * 4096 - 8, b' ');
            let mut file = File::create(&path).unwrap();
            file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
            file.write_all(&bytes).unwrap();
            file.write_all(payload).unwrap();
            file.write_all(index).unwrap();
            Self(path)
        }
    }
    #[test]
    fn safetensors_header_checks_types_extents_and_metadata_bounds() {
        let index = wire::ModelIndex {
            version: 1,
            ranks: vec![wire::BundleIndex::default()],
            ..Default::default()
        };
        let bytes = index.encode_to_vec();
        let header = serde_json::json!({
            "__metadata__": {"format": "flare"},
            "flare.payload": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
            "flare.metadata": {"dtype": "U8", "shape": [bytes.len()], "data_offsets": [4, 4 + bytes.len()]},
        });
        let fixture = Fixture::with_header(&header, &bytes, b"data");
        let (archive, decoded) = Archive::open(&fixture.0).unwrap();
        assert_eq!(archive.start, 4096);
        assert_eq!(decoded, index);
        for (path, value) in [
            ("/__metadata__/format", serde_json::json!("pt")),
            ("/flare.payload/dtype", serde_json::json!("F32")),
            ("/flare.payload/shape", serde_json::json!([2, 2])),
            ("/flare.payload/data_offsets", serde_json::json!([1, 5])),
            ("/flare.metadata/data_offsets/0", serde_json::json!(3)),
            (
                "/flare.metadata/data_offsets/1",
                serde_json::json!(u64::MAX),
            ),
            ("/flare.metadata/shape/0", serde_json::json!(MAX_INDEX + 1)),
        ] {
            let mut invalid = header.clone();
            *invalid.pointer_mut(path).unwrap() = value;
            let fixture = Fixture::with_header(&invalid, &bytes, b"data");
            assert!(Archive::open(&fixture.0).is_err(), "{path}");
        }
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&fixture.0)
            .unwrap();
        file.set_len(4096 + 4 + bytes.len() as u64 - 1).unwrap();
        assert!(Archive::open(&fixture.0).is_err());
        file.write_all_at(&((MAX_HEADER + 1) as u64).to_le_bytes(), 0)
            .unwrap();
        assert!(Archive::open(&fixture.0).is_err());
    }

    #[test]
    fn resources_cannot_address_the_protobuf_tensor() {
        let index = wire::ModelIndex {
            version: 1,
            ranks: vec![wire::BundleIndex {
                resources: [(
                    "outside".into(),
                    wire::Span {
                        offset: 4,
                        nbytes: 1,
                    },
                )]
                .into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let fixture = Fixture::new(&index, b"data");
        assert!(
            Archive::open(&fixture.0)
                .err()
                .unwrap()
                .to_string()
                .contains("resource exceeds")
        );
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
