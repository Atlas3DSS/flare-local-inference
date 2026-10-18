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
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Take},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
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

/// One package file retained while a peer receives selected byte ranges.
#[derive(Clone)]
pub struct PackageFile {
    /// Empty for the primary; companions retain their signed sibling basenames.
    pub filename: String,
    pub file: Arc<File>,
    pub ranges: Vec<(u64, u64)>,
}

fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.retain(|&(_, nbytes)| nbytes > 0);
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
    merged
}

pub(super) fn validate_filename(name: &str) -> Result<()> {
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    if name.len() > 255
        || !name.ends_with(".safetensors")
        || name.contains("..")
        || !name
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        || matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
    {
        return Err(invalid("invalid package companion filename"));
    }
    Ok(())
}

pub(super) struct PayloadFile {
    pub file: Arc<File>,
    pub start: u64,
    pub length: u64,
    direct: Mutex<Option<Arc<File>>>,
}
impl PayloadFile {
    pub fn direct(&self) -> Result<Arc<File>> {
        let mut cached = self.direct.lock().unwrap();
        if cached.is_none() {
            *cached = Some(Arc::new(crate::os::direct_reopen(&self.file).map_err(
                |e| invalid(format!("open package payload for direct I/O: {e}")),
            )?));
        }
        Ok(cached.as_ref().unwrap().clone())
    }
    fn companion(file: File) -> Result<Self> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct DataHeader {
            #[serde(rename = "__metadata__")]
            format: Format,
            #[serde(rename = "flare.payload")]
            payload: ByteTensor,
        }
        let info = file.metadata().map_err(|e| e.to_string())?;
        if !info.is_file() {
            return Err(invalid("package companion must be a regular file"));
        }
        let mut prefix = [0; 8];
        file.read_exact_at(&mut prefix, 0)
            .map_err(|e| e.to_string())?;
        let size = u64::from_le_bytes(prefix);
        if size == 0 || size > MAX_HEADER as u64 || (size + 8) % 4096 != 0 {
            return Err(invalid("invalid companion header size"));
        }
        let mut bytes = vec![0; size as usize];
        file.read_exact_at(&mut bytes, 8)
            .map_err(|e| e.to_string())?;
        if bytes[0] != b'{' {
            return Err(invalid("invalid companion header"));
        }
        let header: DataHeader =
            serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        let length = header.payload.size()?;
        if header.format.format != "flare-data"
            || header.payload.data_offsets[0] != 0
            || info.len().checked_sub(size + 8) != Some(length)
        {
            return Err(invalid("invalid companion payload length or format"));
        }
        Ok(Self {
            file: Arc::new(file),
            start: size + 8,
            length,
            direct: Mutex::new(None),
        })
    }
}

/// Inspect bounded metadata for sibling files before authentication, without opening any sibling.
/// Callers may fetch these names from the primary's pinned source; signatures still require verification.
pub fn package_companions(path: &Path) -> Result<Vec<String>> {
    let (_, _, _, bytes) = Archive::envelope(path)?;
    let envelope =
        wire::ProtectedIndex::decode(bytes.as_slice()).map_err(|e| invalid(e.to_string()))?;
    if envelope.version != 4
        || envelope.groups.len() != 3
        || envelope.groups[0].kind != 1
        || envelope.ranks.is_empty()
    {
        return Err(invalid("invalid protected package envelope"));
    }
    let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())
        .map_err(|e| invalid(e.to_string()))?;
    let mut index = metadata
        .index
        .ok_or_else(|| invalid("missing model index"))?;
    if !matches!(index.version, 4 | 5) {
        return Err(invalid("unsupported model index"));
    }
    let mut parts = wire::StoredData::decode(envelope.groups[2].body.as_slice())
        .map_err(|e| invalid(e.to_string()))?
        .parts;
    for group in envelope.ranks {
        let rank = wire::RankProtection::decode(group.body.as_slice())
            .map_err(|e| invalid(e.to_string()))?;
        index
            .ranks
            .push(rank.bundle.ok_or_else(|| invalid("missing rank bundle"))?);
        parts.extend(rank.data.unwrap_or_default().parts);
    }
    let mut names = std::collections::BTreeSet::new();
    for span in index
        .ranks
        .iter()
        .flat_map(|rank| rank.resources.values())
        .chain(
            index
                .assets
                .iter()
                .filter_map(|asset| asset.payload.as_ref()),
        )
        .chain(
            index
                .engram
                .iter()
                .flat_map(|e| &e.tables)
                .flat_map(|t| [&t.weights, &t.scales])
                .filter_map(Option::as_ref),
        )
        .chain(&parts)
    {
        if !span.filename.is_empty() {
            validate_filename(&span.filename)?;
            if index.version != 5 {
                return Err(invalid("companion references require model index 5"));
            }
            names.insert(span.filename.clone());
            if names.len() > 99 {
                return Err(invalid("package references more than 99 data files"));
            }
        }
    }
    Ok(names.into_iter().collect())
}

pub(super) struct Archive {
    pub path: PathBuf,
    pub start: u64,
    /// Payload bytes after `start`; the signed index follows them to the end of the file.
    pub payload: u64,
    pub file: Arc<File>,
    pub files: BTreeMap<String, PayloadFile>,
    _directory: File,
    pub(super) protection: Protection,
}
impl Archive {
    /// Open the file and read its signed envelope: the file, the payload start and size, and the
    /// envelope bytes, none of them verified yet.
    pub(super) fn envelope(path: &Path) -> Result<(File, u64, u64, Vec<u8>)> {
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
        Ok((file, start, payload, bytes))
    }
    pub fn open(
        path: &Path,
        publishers: &[Publisher],
        rank: Option<usize>,
    ) -> Result<(Self, wire::ModelIndex)> {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        #[cfg(unix)]
        let directory = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_PATH | libc::O_DIRECTORY)
                .open(parent)
                .map_err(|e| e.to_string())?
        };
        #[cfg(windows)]
        let directory = {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ)
                .custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS)
                .open(parent)
                .map_err(|e| e.to_string())?
        };
        #[cfg(unix)]
        let primary = {
            use std::os::fd::AsRawFd;
            PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd())).join(
                path.file_name()
                    .ok_or_else(|| invalid("missing primary filename"))?,
            )
        };
        #[cfg(windows)]
        let primary = path.to_owned();
        let (file, start, payload, bytes) = Self::envelope(&primary)?;
        let (protection, index) = Protection::verify(&bytes, publishers, payload, rank)?;
        let file = Arc::new(file);
        let mut files = BTreeMap::new();
        files.insert(
            String::new(),
            PayloadFile {
                file: file.clone(),
                start,
                length: payload,
                direct: Mutex::new(None),
            },
        );
        for part in protection.data_parts() {
            if files.contains_key(&part.filename) {
                continue;
            }
            validate_filename(&part.filename)?;
            if files.len() > 99 {
                return Err(invalid("package references more than 99 data files"));
            }
            #[cfg(unix)]
            let companion = {
                use std::os::fd::{AsRawFd, FromRawFd};
                let name = std::ffi::CString::new(part.filename.as_str()).unwrap();
                // SAFETY: directory and name remain live; the returned descriptor has one owner.
                let fd = unsafe {
                    libc::openat(
                        directory.as_raw_fd(),
                        name.as_ptr(),
                        libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
                    )
                };
                if fd < 0 {
                    return Err(invalid(format!(
                        "open companion {}: {}",
                        part.filename,
                        std::io::Error::last_os_error()
                    )));
                }
                unsafe { File::from_raw_fd(fd) }
            };
            #[cfg(windows)]
            let companion = {
                use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
                use windows_sys::Win32::Storage::FileSystem::{
                    FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_READ,
                };
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .share_mode(FILE_SHARE_READ)
                    .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
                    .open(parent.join(&part.filename))
                    .map_err(|e| e.to_string())?;
                if file
                    .metadata()
                    .map_err(|e| e.to_string())?
                    .file_attributes()
                    & FILE_ATTRIBUTE_REPARSE_POINT
                    != 0
                {
                    return Err(invalid("package companion cannot be a reparse point"));
                }
                file
            };
            files.insert(part.filename.clone(), PayloadFile::companion(companion)?);
        }
        let result = Self {
            #[cfg(unix)]
            path: {
                use std::os::fd::AsRawFd;
                PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()))
            },
            #[cfg(windows)]
            path: path.to_owned(),
            start,
            payload,
            file,
            files,
            _directory: directory,
            protection,
        };
        for part in result.protection.data_parts() {
            result.source(part)?;
        }
        let spans = index
            .ranks
            .iter()
            .flat_map(|rank| rank.resources.values())
            .chain(
                index
                    .assets
                    .iter()
                    .filter_map(|asset| asset.payload.as_ref()),
            )
            .chain(
                index
                    .engram
                    .iter()
                    .flat_map(|e| &e.tables)
                    .flat_map(|t| [&t.weights, &t.scales])
                    .filter_map(Option::as_ref),
            );
        let mut ends = BTreeMap::new();
        for span in spans {
            result.source(span)?;
            let end = ends.entry(&span.filename).or_insert(0);
            *end = (*end).max(span.offset + span.nbytes);
        }
        if rank.is_none()
            && ends
                .iter()
                .any(|(name, end)| result.files[*name].length != *end)
        {
            return Err(invalid("unreferenced package payload bytes"));
        }
        Ok((result, index))
    }
    pub fn source(&self, span: &wire::Span) -> Result<&PayloadFile> {
        let file = self
            .files
            .get(&span.filename)
            .ok_or_else(|| invalid("resource names an unsigned companion"))?;
        if span
            .offset
            .checked_add(span.nbytes)
            .is_none_or(|end| end > file.length)
        {
            return Err(invalid("resource exceeds package payload"));
        }
        Ok(file)
    }
    pub fn file_index(&self, filename: &str) -> Result<usize> {
        self.files
            .keys()
            .position(|name| name == filename)
            .ok_or_else(|| invalid("unknown package file"))
    }

    /// Per-file ranges that a rank needs, including signed stored-data padding.
    pub fn rank_ranges(&self, index: &wire::ModelIndex, rank: usize) -> Result<Vec<PackageFile>> {
        let position = self.protection.position(rank)?;
        let bundle = index
            .ranks
            .get(position)
            .ok_or_else(|| invalid("rank outside package"))?;
        let stored = self.protection.stored_ranges(position);
        let mut selected: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
        let length = self.file.metadata().map_err(|e| e.to_string())?.len();
        selected.insert(
            String::new(),
            vec![
                (0, self.start),
                (
                    self.start + self.payload,
                    length - self.start - self.payload,
                ),
            ],
        );
        for span in bundle
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
            .chain(&stored)
        {
            let file = self.source(span)?;
            selected
                .entry(span.filename.clone())
                .or_insert_with(|| vec![(0, file.start)])
                .push((file.start + span.offset, span.nbytes));
        }
        Ok(selected
            .into_iter()
            .map(|(filename, ranges)| PackageFile {
                file: self.files[&filename].file.clone(),
                filename,
                ranges: merge_ranges(ranges),
            })
            .collect())
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
        for file in self.files.values() {
            crate::os::release_page_cache(&file.file);
        }
    }
    fn reader(&self, package: &Package, key: &str) -> Result<ResourceReader> {
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
    fn transfer_ranges_merge_adjacent_and_overlapping_extents() {
        assert_eq!(
            merge_ranges(vec![(20, 3), (0, 10), (5, 10), (15, 1), (30, 0)]),
            vec![(0, 16), (20, 3)]
        );
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
