//! Bounded protobuf indexes and resource reads inside a safetensors container.

use super::{
    format::{Package, invalid},
    protection::{Protection, Publisher, ResourceReader},
    protobuf::wire,
};
use crate::Result;
use crate::os::FileReadAt;
use prost::Message;
use serde::Deserialize;
use std::{
    fs::File,
    io::{BufReader, Read, Take},
    path::{Path, PathBuf},
    sync::Arc,
};

const MAX_HEADER: usize = 64 << 10;
const MAX_INDEX: usize = 128 << 20;
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

/// Byte ranges of a package file of `length` bytes, payload at `start`, that serving `rank` reads,
/// including the signed stored-data ranges `stored` (payload offset, length) that rank verifies.
fn rank_ranges(
    index: &wire::ModelIndex,
    rank: usize,
    stored: &[(u64, u64)],
    start: u64,
    payload: u64,
    length: u64,
) -> Result<Vec<(u64, u64)>> {
    let bundle = index.ranks.get(rank).ok_or_else(|| {
        invalid(format!(
            "rank {rank} is outside the package's {} ranks",
            index.ranks.len()
        ))
    })?;
    let index_start = start + payload;
    let mut ranges: Vec<(u64, u64)> = bundle
        .resources
        .values()
        .chain(index.assets.iter().filter_map(|a| a.payload.as_ref()))
        .chain(
            index
                .engram
                .iter()
                .flat_map(|e| &e.tables)
                .flat_map(|t| [&t.weights, &t.scales])
                .filter_map(Option::as_ref),
        )
        .map(|span| (start + span.offset, span.nbytes))
        .chain(
            stored
                .iter()
                .map(|&(offset, nbytes)| (start + offset, nbytes)),
        )
        .chain([(0, start), (index_start, length - index_start)])
        .filter(|&(_, nbytes)| nbytes > 0)
        .collect();
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (offset, nbytes) in ranges {
        match merged.last_mut() {
            Some((first, extent)) if offset <= *first + *extent => {
                *extent = (*extent).max(offset + nbytes - *first);
            }
            _ => merged.push((offset, nbytes)),
        }
    }
    Ok(merged)
}

pub(super) struct Archive {
    pub path: PathBuf,
    pub start: u64,
    /// Payload bytes after `start`; the signed index follows them to the end of the file.
    pub payload: u64,
    pub file: Arc<File>,
    pub(super) protection: Protection,
    pub(super) accepted: bool,
}
impl Archive {
    pub fn open(
        path: &Path,
        publishers: &[Publisher],
        rank: Option<usize>,
    ) -> Result<(Self, wire::ModelIndex)> {
        #[cfg(unix)]
        let file = File::open(path).map_err(|e| e.to_string())?;
        #[cfg(windows)]
        let file = {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ)
                .open(path)
                .map_err(|e| e.to_string())?
        };
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
        let (protection, index) = Protection::verify(&bytes, publishers, payload, rank)?;
        Ok((
            Self {
                #[cfg(unix)]
                path: {
                    use std::os::fd::AsRawFd;
                    PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()))
                },
                #[cfg(windows)]
                path: path.to_owned(),
                start,
                payload,
                file: Arc::new(file),
                protection,
                accepted: false,
            },
            index,
        ))
    }

    /// File byte ranges, in offset order, that a host serving `rank` of `index` reads: the header,
    /// the signed index, that rank's bundle and stored data, and the assets and tables every rank
    /// shares.
    pub fn rank_ranges(&self, index: &wire::ModelIndex, rank: usize) -> Result<Vec<(u64, u64)>> {
        let position = self.protection.position(rank)?;
        let length = self.file.metadata().map_err(|e| e.to_string())?.len();
        let stored = self.protection.stored_ranges(position);
        rank_ranges(index, position, &stored, self.start, self.payload, length)
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
        crate::os::release_page_cache(&self.file);
    }
    fn reader(&self, package: &Package, key: &str) -> Result<ResourceReader> {
        if !self.accepted {
            return Err(invalid(
                "publisher acceptance required before resource access",
            ));
        }
        self.protection
            .reader(self.file.clone(), self.start, self.span(package, key)?)
    }
    pub fn read(&self, package: &Package, key: &str) -> Result<Vec<u8>> {
        let mut reader = self.reader(package, key)?;
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        Ok(bytes)
    }
    pub fn records(&self, package: &Package, key: &str) -> Result<Records> {
        let reader = self.reader(package, key)?;
        let nbytes = reader.len();
        Ok(Records {
            reader: BufReader::with_capacity(16 << 10, reader).take(nbytes),
            buffer: Vec::new(),
        })
    }
}

pub(super) struct Records {
    reader: Take<BufReader<ResourceReader>>,
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
    #[test]
    fn a_rank_reads_its_bundle_and_shared_spans_only() {
        let span = |offset, nbytes| Some(wire::Span { offset, nbytes });
        let bundle = |offset| wire::BundleIndex {
            resources: [(
                "storage.bin".to_owned(),
                wire::Span {
                    offset,
                    nbytes: 100,
                },
            )]
            .into(),
            ..Default::default()
        };
        let index = wire::ModelIndex {
            version: 4,
            ranks: vec![bundle(0), bundle(4096)],
            assets: vec![wire::Asset {
                name: "tokenizer".into(),
                payload: span(100, 50),
                sha256: String::new(),
            }],
            engram: Some(wire::Engram {
                tables: vec![wire::EngramTable {
                    weights: span(8192, 1000),
                    scales: span(9192, 8),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let (start, payload, length) = (4096, 10_000, 4096 + 10_000 + 300);
        let index_start = start + payload;
        // The tables' signed range, then each rank's, padding included.
        let shared = (8192, 1008);
        // Rank zero's bundle abuts the header and its tokenizer.
        assert_eq!(
            rank_ranges(&index, 0, &[shared, (0, 100)], start, payload, length).unwrap(),
            vec![
                (0, start + 150),
                (start + 8192, 1008),
                (index_start, length - index_start)
            ]
        );
        assert_eq!(
            rank_ranges(&index, 1, &[shared, (4096, 200)], start, payload, length).unwrap(),
            vec![
                (0, start),
                (start + 100, 50),
                (start + 4096, 200),
                (start + 8192, 1008),
                (index_start, length - index_start)
            ]
        );
        assert!(rank_ranges(&index, 2, &[shared], start, payload, length).is_err());
    }

    #[test]
    fn header_and_index_lengths_are_bounded_before_reading() {
        let path =
            std::env::temp_dir().join(format!("flare-header-{}.safetensors", std::process::id()));
        for size in [0, MAX_HEADER as u64 + 1, u64::MAX] {
            std::fs::write(&path, size.to_le_bytes()).unwrap();
            assert!(
                Archive::open(&path, &[], None)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("header size")
            );
        }
        let header = serde_json::json!({
            "__metadata__": {"format": "flare"},
            "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0,0]},
            "flare.metadata": {"dtype": "U8", "shape": [MAX_INDEX + 1], "data_offsets": [0,MAX_INDEX + 1]},
        });
        let mut bytes = serde_json::to_vec(&header).unwrap();
        bytes.resize(4088, b' ');
        let mut file = File::create(&path).unwrap();
        file.write_all(&4088u64.to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        drop(file);
        assert!(
            Archive::open(&path, &[], None)
                .err()
                .unwrap()
                .to_string()
                .contains("index size")
        );
        std::fs::remove_file(path).unwrap();
    }
}
