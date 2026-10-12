//! Record files: one checkpoint per file, self-describing, aligned for direct I/O.
//!
//! A record lives at `<directory>/<consumed>-<digest>.ckpt`, with the consumed
//! count as 16 lowercase hexadecimal digits and the digest as 32. Its segments
//! are, in order and each starting at a [`BLOCK_BYTES`] boundary:
//!
//! 1. a header of [`HEADER_BYTES`], laid out as described below;
//! 2. a checksum segment of one 64-bit xxh3 per tail page, padded to a block;
//! 3. the page segment, page-major: page `i`, then rank `r`, then that rank's
//!    planes in geometry order, every plane padded to a block;
//! 4. the row segment, rank-major, every row padded to a block.
//!
//! The header occupies the first block. Every field is little-endian and the
//! fields are packed in this order with these widths:
//!
//! | offset | bytes | field |
//! |---|---|---|
//! | 0 | 8 | magic `FLARECKP` |
//! | 8 | 4 | format version, `u32` |
//! | 12 | 4 | flags, `u32` |
//! | 16 | 32 | model identity |
//! | 48 | 16 | layout fingerprint |
//! | 64 | 8 | key consumed count, `u64` |
//! | 72 | 16 | key digest |
//! | 88 | 1 | parent presence, `0` or `1` |
//! | 89 | 8 | parent consumed count, `u64`, zero when absent |
//! | 97 | 16 | parent digest, zero when absent |
//! | 113 | 16 | bucket digest |
//! | 129 | 8 | page token count, `u64` |
//! | 137 | 4 | tail page count, `u32` |
//! | 141 | 8 | page segment offset, `u64` |
//! | 149 | 8 | row segment offset, `u64` |
//! | 157 | 8 | row checksum, `u64` |
//! | 165 | 8 | creation time, seconds since the Unix epoch, `u64` |
//! | 173 | 4 | rank count, `u32` |
//! | 177 | variable | rank entries |
//!
//! Each rank entry is a plane count as `u32`, that many plane unit sizes as
//! `u64`, and the row unit size as `u64`. Unit sizes are the unpadded sizes
//! the GPU geometry declares; padding is derived. The bytes after the last
//! rank entry are zero. A geometry whose entries do not fit in the block is
//! rejected. The segment offsets are derived: the page segment starts after
//! the checksum segment and the row segment after `tail_pages` page records.
//!
//! The row exists when the file length covers the row segment; compaction
//! truncates the file at the row segment offset and changes nothing else. Writes
//! go to a temporary name in the same directory and rename into place.
//! Per-page checksums let a restore that reads only a suffix of the pages verify
//! what it read.

use std::{
    collections::HashSet,
    fmt::Write as _,
    fs,
    io::{self, Read as _},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{Error, ExecutionError, Failure, Result, cache::PrefixKey};

use super::geometry::{BLOCK_BYTES, CacheGeometry, Layout, RankLayout, invalid};

pub const MAGIC: [u8; 8] = *b"FLARECKP";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = BLOCK_BYTES as usize;
pub const EXTENSION: &str = "ckpt";
const SUFFIX: &str = ".ckpt";

/// Bytes of the fixed header fields that precede the rank entries.
const FIXED_FIELD_BYTES: usize = 177;
/// Length of a record file name: 16 hexadecimal digits, a hyphen, 32 digits, and `.ckpt`.
const FILE_NAME_LENGTH: usize = 16 + 1 + 32 + SUFFIX.len();

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankSizes {
    pub planes: Vec<u64>,
    pub row_bytes: u64,
}

impl RankSizes {
    fn page_record_bytes(&self) -> u64 {
        self.planes.iter().map(|size| padded(*size)).sum()
    }

    fn row_record_bytes(&self) -> u64 {
        padded(self.row_bytes)
    }

    fn encoded_bytes(&self) -> usize {
        4 + 8 * self.planes.len() + 8
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub identity: [u8; 32],
    pub fingerprint: [u8; 16],
    pub flags: u32,
    pub key: PrefixKey,
    pub parent: Option<PrefixKey>,
    pub bucket: [u8; 16],
    pub page_tokens: u64,
    pub tail_pages: u32,
    pub ranks: Vec<RankSizes>,
    pub page_segment: u64,
    pub row_segment: u64,
    pub row_checksum: u64,
    /// Seconds since the Unix epoch.
    pub created: u64,
}

impl Header {
    /// A header for a record under `layout`, with segment offsets derived from
    /// the tail page count and checksums left zero until the writer fills them.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        geometry: &[CacheGeometry],
        identity: [u8; 32],
        fingerprint: [u8; 16],
        key: PrefixKey,
        parent: Option<PrefixKey>,
        bucket: [u8; 16],
        tail_pages: u32,
        created: SystemTime,
    ) -> Result<Self> {
        let layout = Layout::new(geometry)?;
        let ranks = geometry
            .iter()
            .map(|rank| RankSizes {
                planes: rank.planes.iter().map(|plane| plane.unit_bytes).collect(),
                row_bytes: rank.row.unit_bytes,
            })
            .collect();
        let mut header = Self {
            identity,
            fingerprint,
            flags: 0,
            key,
            parent,
            bucket,
            page_tokens: layout.page_tokens,
            tail_pages,
            ranks,
            page_segment: 0,
            row_segment: 0,
            row_checksum: 0,
            created: created
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs()),
        };
        header.page_segment = header.checksum_segment() + header.checksum_bytes();
        header.row_segment = header
            .page_segment
            .checked_add(
                (header.tail_pages as u64)
                    .checked_mul(header.page_record_bytes())
                    .ok_or_else(|| invalid("record page segment size overflows"))?,
            )
            .ok_or_else(|| invalid("record row segment offset overflows"))?;
        header.validate()?;
        Ok(header)
    }

    pub fn encode(&self) -> Result<[u8; HEADER_BYTES]> {
        self.validate()?;
        let mut bytes = [0u8; HEADER_BYTES];
        let mut cursor = Cursor::new(&mut bytes);
        cursor.put(&MAGIC);
        cursor.put(&VERSION.to_le_bytes());
        cursor.put(&self.flags.to_le_bytes());
        cursor.put(&self.identity);
        cursor.put(&self.fingerprint);
        cursor.put(&self.key.consumed_tokens.to_le_bytes());
        cursor.put(&self.key.digest);
        cursor.put(&[self.parent.is_some() as u8]);
        let parent = self.parent.unwrap_or(PrefixKey {
            consumed_tokens: 0,
            digest: [0; 16],
        });
        cursor.put(&parent.consumed_tokens.to_le_bytes());
        cursor.put(&parent.digest);
        cursor.put(&self.bucket);
        cursor.put(&self.page_tokens.to_le_bytes());
        cursor.put(&self.tail_pages.to_le_bytes());
        cursor.put(&self.page_segment.to_le_bytes());
        cursor.put(&self.row_segment.to_le_bytes());
        cursor.put(&self.row_checksum.to_le_bytes());
        cursor.put(&self.created.to_le_bytes());
        cursor.put(&(self.ranks.len() as u32).to_le_bytes());
        for rank in &self.ranks {
            cursor.put(&(rank.planes.len() as u32).to_le_bytes());
            for size in &rank.planes {
                cursor.put(&size.to_le_bytes());
            }
            cursor.put(&rank.row_bytes.to_le_bytes());
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_BYTES {
            return Err(invalid(format!(
                "record header block is {} bytes; a header occupies {HEADER_BYTES}",
                bytes.len()
            )));
        }
        let mut cursor = Reader::new(&bytes[..HEADER_BYTES]);
        if cursor.array::<8>("magic")? != MAGIC {
            return Err(invalid("record header magic is not FLARECKP"));
        }
        let version = cursor.u32("version")?;
        if version != VERSION {
            return Err(invalid(format!(
                "record header version {version} is not the supported version {VERSION}"
            )));
        }
        let flags = cursor.u32("flags")?;
        let identity = cursor.array::<32>("identity")?;
        let fingerprint = cursor.array::<16>("fingerprint")?;
        let key = PrefixKey {
            consumed_tokens: cursor.u64("key consumed count")?,
            digest: cursor.array::<16>("key digest")?,
        };
        let parent_present = cursor.array::<1>("parent presence")?[0];
        let parent_key = PrefixKey {
            consumed_tokens: cursor.u64("parent consumed count")?,
            digest: cursor.array::<16>("parent digest")?,
        };
        let parent = match parent_present {
            0 => None,
            1 => Some(parent_key),
            other => {
                return Err(invalid(format!(
                    "record header parent presence flag is {other}; it must be 0 or 1"
                )));
            }
        };
        let bucket = cursor.array::<16>("bucket digest")?;
        let page_tokens = cursor.u64("page token count")?;
        let tail_pages = cursor.u32("tail page count")?;
        let page_segment = cursor.u64("page segment offset")?;
        let row_segment = cursor.u64("row segment offset")?;
        let row_checksum = cursor.u64("row checksum")?;
        let created = cursor.u64("creation time")?;
        let rank_count = cursor.u32("rank count")?;
        let mut ranks = Vec::new();
        for rank in 0..rank_count {
            let plane_count = cursor.u32("plane count")?;
            let mut planes = Vec::new();
            for _ in 0..plane_count {
                planes.push(cursor.u64("plane size")?);
            }
            let row_bytes = cursor.u64("row size")?;
            if planes.is_empty() {
                return Err(invalid(format!(
                    "record header rank {rank} declares no page planes"
                )));
            }
            ranks.push(RankSizes { planes, row_bytes });
        }
        let header = Self {
            identity,
            fingerprint,
            flags,
            key,
            parent,
            bucket,
            page_tokens,
            tail_pages,
            ranks,
            page_segment,
            row_segment,
            row_checksum,
            created,
        };
        header.validate()?;
        Ok(header)
    }

    /// Every field agrees with the layout rules: nonzero sizes, a key that
    /// extends its parent by exactly the tail pages, derived segment offsets,
    /// and rank entries that fit in the header block.
    fn validate(&self) -> Result<()> {
        if self.ranks.is_empty() {
            return Err(invalid("record header rank count is zero"));
        }
        let entry_bytes: usize = self.ranks.iter().map(RankSizes::encoded_bytes).sum();
        if FIXED_FIELD_BYTES + entry_bytes > HEADER_BYTES {
            return Err(invalid(format!(
                "record header for {} ranks and {} planes needs {} bytes, more than the \
                 {HEADER_BYTES}-byte header block",
                self.ranks.len(),
                self.ranks
                    .iter()
                    .map(|rank| rank.planes.len())
                    .sum::<usize>(),
                FIXED_FIELD_BYTES + entry_bytes
            )));
        }
        for (index, rank) in self.ranks.iter().enumerate() {
            if rank.planes.is_empty() {
                return Err(invalid(format!(
                    "record header rank {index} declares no page planes"
                )));
            }
            if rank.planes.contains(&0) {
                return Err(invalid(format!(
                    "record header rank {index} declares a zero-byte plane"
                )));
            }
            if rank.row_bytes == 0 {
                return Err(invalid(format!(
                    "record header rank {index} declares a zero-byte row"
                )));
            }
        }
        if self.page_tokens == 0 {
            return Err(invalid("record header page token count is zero"));
        }
        if self.key.consumed_tokens == 0 {
            return Err(invalid("record header key consumed count is zero"));
        }
        let parent_tokens = match self.parent {
            Some(parent) if parent.consumed_tokens >= self.key.consumed_tokens => {
                return Err(invalid(format!(
                    "record header parent consumed count {} is not below the key consumed count {}",
                    parent.consumed_tokens, self.key.consumed_tokens
                )));
            }
            Some(parent) => parent.consumed_tokens,
            None => 0,
        };
        let expected_tail =
            self.key.consumed_tokens / self.page_tokens - parent_tokens / self.page_tokens;
        if expected_tail != self.tail_pages as u64 {
            return Err(invalid(format!(
                "record header tail page count {} disagrees with the {expected_tail} complete \
                 pages between the parent and the key",
                self.tail_pages
            )));
        }
        let page_segment = self.checksum_segment() + self.checksum_bytes();
        if self.page_segment != page_segment {
            return Err(invalid(format!(
                "record header page segment offset {} is not the derived offset {page_segment}",
                self.page_segment
            )));
        }
        let row_segment = (self.tail_pages as u64)
            .checked_mul(self.page_record_bytes())
            .and_then(|pages| page_segment.checked_add(pages))
            .ok_or_else(|| invalid("record header segment sizes overflow"))?;
        if self.row_segment != row_segment {
            return Err(invalid(format!(
                "record header row segment offset {} is not the derived offset {row_segment}",
                self.row_segment
            )));
        }
        Ok(())
    }

    /// Offset of the per-page checksum segment.
    pub fn checksum_segment(&self) -> u64 {
        HEADER_BYTES as u64
    }

    /// Bytes of the checksum segment, padded to a block.
    pub fn checksum_bytes(&self) -> u64 {
        (self.tail_pages as u64 * 8).div_ceil(BLOCK_BYTES) * BLOCK_BYTES
    }

    /// Bytes of one page record across ranks under this header's sizes.
    pub fn page_record_bytes(&self) -> u64 {
        self.ranks.iter().map(RankSizes::page_record_bytes).sum()
    }

    pub fn row_record_bytes(&self) -> u64 {
        self.ranks.iter().map(RankSizes::row_record_bytes).sum()
    }

    /// File length when the row segment is present.
    pub fn length_with_row(&self) -> u64 {
        self.row_segment + self.row_record_bytes()
    }

    pub fn row_present(&self, file_length: u64) -> bool {
        file_length >= self.length_with_row()
    }

    /// The header's sizes equal the live geometry's.
    pub fn matches(&self, geometry: &[CacheGeometry]) -> bool {
        self.ranks.len() == geometry.len()
            && self.ranks.iter().zip(geometry).all(|(sizes, rank)| {
                rank.page_tokens == self.page_tokens
                    && rank.row.unit_bytes == sizes.row_bytes
                    && rank.planes.len() == sizes.planes.len()
                    && rank
                        .planes
                        .iter()
                        .zip(&sizes.planes)
                        .all(|(plane, size)| plane.unit_bytes == *size)
            })
    }

    /// The cross-rank layout the header describes.
    pub fn layout(&self) -> Result<Layout> {
        self.validate()?;
        let mut ranks = Vec::with_capacity(self.ranks.len());
        let (mut page_record_bytes, mut row_record_bytes) = (0u64, 0u64);
        for rank in &self.ranks {
            let mut plane_offsets = Vec::with_capacity(rank.planes.len());
            let mut offset = 0;
            for size in &rank.planes {
                plane_offsets.push(offset);
                offset += padded(*size);
            }
            ranks.push(RankLayout {
                page_offset: page_record_bytes,
                plane_offsets,
                row_offset: row_record_bytes,
            });
            page_record_bytes += offset;
            row_record_bytes += rank.row_record_bytes();
        }
        Ok(Layout {
            page_tokens: self.page_tokens,
            ranks,
            page_record_bytes,
            row_record_bytes,
        })
    }
}

fn padded(bytes: u64) -> u64 {
    bytes.div_ceil(BLOCK_BYTES) * BLOCK_BYTES
}

struct Cursor<'a> {
    bytes: &'a mut [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    // The header is validated to fit before encoding, so the block never overruns.
    fn put(&mut self, field: &[u8]) {
        self.bytes[self.position..self.position + field.len()].copy_from_slice(field);
        self.position += field.len();
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, count: usize, field: &str) -> Result<&'a [u8]> {
        let end = self.position + count;
        if end > self.bytes.len() {
            return Err(invalid(format!(
                "record header {field} extends past the header block"
            )));
        }
        let taken = &self.bytes[self.position..end];
        self.position = end;
        Ok(taken)
    }

    fn array<const N: usize>(&mut self, field: &str) -> Result<[u8; N]> {
        let mut array = [0u8; N];
        array.copy_from_slice(self.take(N, field)?);
        Ok(array)
    }

    fn u32(&mut self, field: &str) -> Result<u32> {
        self.array::<4>(field).map(u32::from_le_bytes)
    }

    fn u64(&mut self, field: &str) -> Result<u64> {
        self.array::<8>(field).map(u64::from_le_bytes)
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(text, "{byte:02x}").unwrap();
    }
    text
}

fn hex_digits(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2)
        || !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    text.as_bytes()
        .chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok())
        .collect()
}

/// `<identity prefix>-<fingerprint>`: 16 hexadecimal digits of the model
/// identity and 32 of the layout fingerprint.
pub fn directory_name(identity: &[u8; 32], fingerprint: &[u8; 16]) -> String {
    format!("{}-{}", hex(&identity[..8]), hex(fingerprint))
}

pub fn file_name(key: &PrefixKey) -> String {
    format!(
        "{:016x}-{}.{EXTENSION}",
        key.consumed_tokens,
        hex(&key.digest)
    )
}

pub fn temporary_name(key: &PrefixKey) -> String {
    format!(".{}.tmp", file_name(key))
}

pub fn parse_file_name(name: &str) -> Option<PrefixKey> {
    if name.len() != FILE_NAME_LENGTH || !name.is_ascii() {
        return None;
    }
    let (consumed, rest) = name.split_at(16);
    let rest = rest.strip_prefix('-')?;
    let (digest, suffix) = rest.split_at(32);
    if suffix != SUFFIX {
        return None;
    }
    let consumed = u64::from_be_bytes(hex_digits(consumed)?.try_into().ok()?);
    let digest: [u8; 16] = hex_digits(digest)?.try_into().ok()?;
    Some(PrefixKey {
        consumed_tokens: consumed,
        digest,
    })
}

/// A record found by the startup scan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scanned {
    pub header: Header,
    pub path: PathBuf,
    pub row: bool,
    pub bytes: u64,
    pub modified: SystemTime,
}

/// Read every record header in `directory`, delete temporary files, files
/// whose header or name is invalid or belongs to another identity, fingerprint,
/// or layout, and records whose parent is missing together with their
/// descendants. Returns the survivors ordered by consumed count ascending, so
/// parents precede children.
///
/// The directory is created when missing. Entries are inspected without
/// following symbolic links; symbolic links, subdirectories, and files whose
/// names are neither `.*.tmp` nor `*.ckpt` are left alone. A `.ckpt` file
/// shorter than its page segment cannot supply its pages and is deleted; a
/// file covering the page segment but not the row segment is a page-only
/// record. A file that vanishes while the scan runs is treated as absent.
pub fn scan(
    directory: &Path,
    identity: &[u8; 32],
    fingerprint: &[u8; 16],
    geometry: &[CacheGeometry],
) -> Result<Vec<Scanned>> {
    fs::create_dir_all(directory).map_err(|error| {
        io_failure(
            format!("create record directory {}", directory.display()),
            error,
        )
    })?;
    let metadata = fs::metadata(directory).map_err(|error| {
        io_failure(
            format!("inspect record directory {}", directory.display()),
            error,
        )
    })?;
    if !metadata.is_dir() {
        return Err(invalid(format!(
            "record directory {} is not a directory",
            directory.display()
        )));
    }
    let entries = fs::read_dir(directory).map_err(|error| {
        io_failure(
            format!("list record directory {}", directory.display()),
            error,
        )
    })?;
    let mut records = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(io_failure(
                    format!("list record directory {}", directory.display()),
                    error,
                ));
            }
        };
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let path = entry.path();
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(io_failure(
                    format!("inspect record file {}", path.display()),
                    error,
                ));
            }
        };
        if !metadata.is_file() {
            continue;
        }
        if name.starts_with('.') && name.ends_with(".tmp") {
            remove(&path)?;
            continue;
        }
        if !name.ends_with(SUFFIX) {
            continue;
        }
        match inspect(&path, name, &metadata, identity, fingerprint, geometry) {
            Ok(Some(scanned)) => records.push(scanned),
            Ok(None) => {}
            Err(_) => remove(&path)?,
        }
    }
    loop {
        let keys: HashSet<PrefixKey> = records.iter().map(|record| record.header.key).collect();
        let (kept, orphans): (Vec<Scanned>, Vec<Scanned>) =
            records.into_iter().partition(|record| {
                record
                    .header
                    .parent
                    .is_none_or(|parent| keys.contains(&parent))
            });
        records = kept;
        if orphans.is_empty() {
            break;
        }
        for orphan in &orphans {
            remove(&orphan.path)?;
        }
    }
    records.sort_by_key(|record| (record.header.key.consumed_tokens, record.header.key.digest));
    Ok(records)
}

/// `Ok(None)` when the file vanished; `Err` when it is not a valid record for
/// this directory and must be deleted.
fn inspect(
    path: &Path,
    name: &str,
    metadata: &fs::Metadata,
    identity: &[u8; 32],
    fingerprint: &[u8; 16],
    geometry: &[CacheGeometry],
) -> Result<Option<Scanned>> {
    let mut file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io_failure("open record", error)),
    };
    let mut block = [0u8; HEADER_BYTES];
    file.read_exact(&mut block)
        .map_err(|error| io_failure("read record header", error))?;
    let header = Header::decode(&block)?;
    if header.identity != *identity {
        return Err(invalid("record belongs to another model identity"));
    }
    if header.fingerprint != *fingerprint {
        return Err(invalid("record belongs to another layout fingerprint"));
    }
    if !header.matches(geometry) {
        return Err(invalid("record sizes differ from the cache geometry"));
    }
    if parse_file_name(name) != Some(header.key) {
        return Err(invalid("record file name differs from its header key"));
    }
    let bytes = metadata.len();
    if bytes < header.row_segment {
        return Err(invalid("record file is shorter than its page segment"));
    }
    Ok(Some(Scanned {
        row: header.row_present(bytes),
        header,
        path: path.to_path_buf(),
        bytes,
        modified: metadata.modified().unwrap_or(UNIX_EPOCH),
    }))
}

fn remove(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_failure(
            format!("delete record file {}", path.display()),
            error,
        )),
    }
}

/// A state error carrying the operating-system failure as its cause.
pub(crate) fn io_failure(message: impl Into<String>, error: io::Error) -> Error {
    let message = message.into();
    ExecutionError::State(Failure::caused_by(format!("{message}: {error}"), error)).into()
}

/// 64-bit xxh3 over `bytes`.
pub fn checksum(bytes: &[u8]) -> u64 {
    twox_hash::XxHash3_64::oneshot(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offload::{geometry::tests::geometry, index::Index};
    use std::{fs::File, io::Write as _, os::unix::fs::symlink};

    const IDENTITY: [u8; 32] = [0xab; 32];
    const FINGERPRINT: [u8; 16] = [0xcd; 16];

    fn key(consumed: u64, salt: u8) -> PrefixKey {
        PrefixKey {
            consumed_tokens: consumed,
            digest: [salt; 16],
        }
    }

    fn header(geometry: &[CacheGeometry], consumed: u64, parent: Option<u64>) -> Header {
        let page_tokens = geometry[0].page_tokens;
        let tail = consumed / page_tokens - parent.unwrap_or(0) / page_tokens;
        Header::new(
            geometry,
            IDENTITY,
            FINGERPRINT,
            key(consumed, consumed as u8),
            parent.map(|parent| key(parent, parent as u8)),
            [0x11; 16],
            tail as u32,
            UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000),
        )
        .unwrap()
    }

    fn geometries() -> Vec<Vec<CacheGeometry>> {
        vec![
            vec![geometry(
                64,
                &[("k", 8192), ("v", 8192), ("sel", 100)],
                10_000,
            )],
            vec![
                geometry(64, &[("k", 8192), ("v", 8192), ("sel", 100)], 10_000),
                geometry(64, &[("k", 8192), ("v", 8192), ("sel", 100)], 10_000),
            ],
            vec![
                geometry(128, &[("latent", 73_728), ("index", 16_384)], 1 << 20),
                geometry(128, &[("latent", 73_728), ("index", 16_384)], 1 << 20),
                geometry(128, &[("latent", 73_728), ("index", 16_384)], 1 << 20),
                geometry(128, &[("latent", 73_728), ("index", 16_384)], 1 << 20),
            ],
        ]
    }

    #[test]
    fn header_round_trips_and_derives_aligned_segments() {
        for geometry in geometries() {
            let page_tokens = geometry[0].page_tokens;
            let mut header = header(&geometry, 7 * page_tokens + 3, Some(2 * page_tokens + 5));
            header.row_checksum = 0x1234_5678_9abc_def0;
            header.flags = 3;
            let layout = Layout::new(&geometry).unwrap();
            assert_eq!(header.tail_pages, 5);
            assert_eq!(header.page_record_bytes(), layout.page_record_bytes);
            assert_eq!(header.row_record_bytes(), layout.row_record_bytes);
            assert_eq!(header.checksum_bytes(), BLOCK_BYTES);
            assert_eq!(header.page_segment, 2 * BLOCK_BYTES);
            assert_eq!(
                header.row_segment,
                header.page_segment + 5 * layout.page_record_bytes
            );
            assert_eq!(
                header.length_with_row(),
                header.page_segment + layout.record_bytes(5)
            );
            assert!(header.matches(&geometry));
            assert_eq!(header.layout().unwrap(), layout);
            let encoded = header.encode().unwrap();
            assert_eq!(&encoded[..8], &MAGIC);
            let decoded = Header::decode(&encoded).unwrap();
            assert_eq!(decoded, header);
            assert_eq!(decoded.encode().unwrap(), encoded);
            for rank in &decoded.layout().unwrap().ranks {
                assert!(rank.page_offset.is_multiple_of(BLOCK_BYTES));
                assert!(rank.row_offset.is_multiple_of(BLOCK_BYTES));
                for offset in &rank.plane_offsets {
                    assert!(offset.is_multiple_of(BLOCK_BYTES));
                }
            }
        }
    }

    #[test]
    fn row_presence_follows_the_file_length() {
        let geometry = geometries().remove(0);
        let paged = header(&geometry, 64 * 3, None);
        assert_eq!(paged.row_record_bytes(), 3 * BLOCK_BYTES);
        assert!(!paged.row_present(paged.row_segment));
        assert!(!paged.row_present(paged.length_with_row() - 1));
        assert!(paged.row_present(paged.length_with_row()));
        assert!(paged.row_present(paged.length_with_row() + BLOCK_BYTES));
        let rootless = header(&geometry, 30, None);
        assert_eq!(rootless.tail_pages, 0);
        assert_eq!(rootless.checksum_bytes(), 0);
        assert_eq!(rootless.page_segment, BLOCK_BYTES);
        assert_eq!(rootless.row_segment, BLOCK_BYTES);
        let decoded = Header::decode(&rootless.encode().unwrap()).unwrap();
        assert_eq!(decoded, rootless);
    }

    #[test]
    fn matches_compares_sizes_not_addresses() {
        let geometry = geometries().remove(1);
        let header = header(&geometry, 640, None);
        let mut moved = geometry.clone();
        moved[1].planes[0].address += 0x10_0000;
        moved[0].page_capacity *= 3;
        assert!(header.matches(&moved));
        let mut widened = geometry.clone();
        widened[1].planes[2].unit_bytes = 101;
        assert!(!header.matches(&widened));
        let mut extra = geometry.clone();
        let duplicate = extra[0].planes[0].clone();
        extra[0].planes.push(duplicate);
        assert!(!header.matches(&extra));
        assert!(!header.matches(&geometry[..1]));
        let mut repaged = geometry.clone();
        repaged[0].page_tokens = 128;
        assert!(!header.matches(&repaged));
    }

    #[test]
    fn decode_rejects_corrupt_headers() {
        let geometry = geometries().remove(0);
        let header = header(&geometry, 640, Some(100));
        let encoded = header.encode().unwrap();
        let corrupt = |offset: usize, value: u8| {
            let mut bytes = encoded;
            bytes[offset] = value;
            Header::decode(&bytes).unwrap_err().to_string()
        };
        assert!(corrupt(0, b'X').contains("magic"));
        assert!(corrupt(8, 2).contains("version"));
        assert!(corrupt(88, 2).contains("parent presence"));
        assert!(corrupt(142, 0).contains("page segment"));
        assert!(corrupt(150, 0).contains("row segment"));
        assert!(corrupt(137, 8).contains("tail page count"));
        assert!(corrupt(173, 0).contains("rank count"));
        assert!(corrupt(173, 2).contains("rank 1"));
        assert!(corrupt(129, 0).contains("page token count"));
        assert!(corrupt(177, 0).contains("no page planes"));
        assert!(corrupt(182, 0).contains("zero-byte plane"));
        assert!(
            Header::decode(&encoded[..HEADER_BYTES - 1])
                .unwrap_err()
                .to_string()
                .contains("header block")
        );
        let mut wide = header.clone();
        wide.ranks = vec![
            RankSizes {
                planes: vec![4096; 500],
                row_bytes: 4096,
            };
            1
        ];
        assert!(
            wide.encode()
                .unwrap_err()
                .to_string()
                .contains("header block")
        );
        let mut too_many = header.clone();
        too_many.ranks = vec![
            RankSizes {
                planes: vec![4096; 100],
                row_bytes: 4096,
            };
            5
        ];
        assert!(
            too_many
                .encode()
                .unwrap_err()
                .to_string()
                .contains("header block")
        );
        assert!(
            Header::new(
                &geometry,
                IDENTITY,
                FINGERPRINT,
                key(640, 1),
                Some(key(700, 2)),
                [0; 16],
                0,
                UNIX_EPOCH
            )
            .is_err()
        );
        assert!(
            Header::new(
                &geometry,
                IDENTITY,
                FINGERPRINT,
                key(640, 1),
                Some(key(100, 2)),
                [0; 16],
                7,
                UNIX_EPOCH
            )
            .is_err()
        );
    }

    #[test]
    fn names_round_trip() {
        let key = PrefixKey {
            consumed_tokens: 0x1234,
            digest: [
                0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
                0xee, 0xff,
            ],
        };
        let name = file_name(&key);
        assert_eq!(
            name,
            "0000000000001234-00112233445566778899aabbccddeeff.ckpt"
        );
        assert_eq!(temporary_name(&key), format!(".{name}.tmp"));
        assert_eq!(parse_file_name(&name), Some(key));
        assert_eq!(parse_file_name(&temporary_name(&key)), None);
        assert_eq!(parse_file_name(&name.to_uppercase()), None);
        assert_eq!(parse_file_name(&name[1..]), None);
        assert_eq!(parse_file_name(&name.replace('-', "+")), None);
        assert_eq!(parse_file_name(&name.replace(".ckpt", ".ckpx")), None);
        assert_eq!(
            parse_file_name("+000000000001234-00112233445566778899aabbccddeeff.ckpt"),
            None
        );
        let extreme = PrefixKey {
            consumed_tokens: u64::MAX,
            digest: [0xff; 16],
        };
        assert_eq!(parse_file_name(&file_name(&extreme)), Some(extreme));
        let mut identity = [0u8; 32];
        identity[..8].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04]);
        identity[8] = 0xff;
        assert_eq!(
            directory_name(&identity, &key.digest),
            "deadbeef01020304-00112233445566778899aabbccddeeff"
        );
    }

    fn write_record(directory: &Path, header: &Header, name: &str, length: u64) -> PathBuf {
        let path = directory.join(name);
        let mut file = File::create(&path).unwrap();
        file.write_all(&header.encode().unwrap()).unwrap();
        file.set_len(length).unwrap();
        path
    }

    fn names(records: &[Scanned]) -> Vec<String> {
        records
            .iter()
            .map(|record| {
                record
                    .path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_owned()
            })
            .collect()
    }

    #[test]
    fn scan_keeps_the_valid_tree_and_deletes_everything_else() {
        let directory = std::env::temp_dir().join(format!(
            "flare-record-scan-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&directory);
        let geometry = geometries().remove(0);
        let root = header(&geometry, 200, None);
        let child = header(&geometry, 400, Some(200));
        let grandchild = header(&geometry, 600, Some(400));
        let sibling = header(&geometry, 330, Some(200));
        let orphan = header(&geometry, 1000, Some(900));
        let orphan_child = header(&geometry, 1100, Some(1000));
        let mut other_identity = header(&geometry, 250, None);
        other_identity.identity = [1; 32];
        let mut other_fingerprint = header(&geometry, 260, None);
        other_fingerprint.fingerprint = [2; 16];
        let misnamed = header(&geometry, 270, None);
        let other_geometry = Header::new(
            &geometries().remove(1),
            IDENTITY,
            FINGERPRINT,
            key(280, 28),
            None,
            [0; 16],
            4,
            UNIX_EPOCH,
        )
        .unwrap();

        // The directory does not exist yet; the scan creates it.
        assert!(
            scan(&directory, &IDENTITY, &FINGERPRINT, &geometry)
                .unwrap()
                .is_empty()
        );
        let keep = |header: &Header| {
            write_record(
                &directory,
                header,
                &file_name(&header.key),
                header.length_with_row(),
            )
        };
        keep(&root);
        keep(&child);
        let compacted = write_record(
            &directory,
            &grandchild,
            &file_name(&grandchild.key),
            grandchild.row_segment,
        );
        keep(&sibling);
        let deleted = vec![
            keep(&orphan),
            keep(&orphan_child),
            keep(&other_identity),
            keep(&other_fingerprint),
            keep(&other_geometry),
            write_record(
                &directory,
                &misnamed,
                &file_name(&key(271, 71)),
                misnamed.length_with_row(),
            ),
            write_record(&directory, &root, &temporary_name(&root.key), 0),
            write_record(
                &directory,
                &child,
                &file_name(&key(401, 42)),
                child.row_segment - 1,
            ),
            directory.join("junk.ckpt"),
        ];
        fs::write(&deleted[8], b"not a record").unwrap();
        let foreign = directory.join("README");
        fs::write(&foreign, b"ignored").unwrap();
        let subdirectory = directory.join("nested.ckpt");
        fs::create_dir(&subdirectory).unwrap();
        let link = directory.join(file_name(&key(999, 9)));
        symlink("/nonexistent/target.ckpt", &link).unwrap();

        let records = scan(&directory, &IDENTITY, &FINGERPRINT, &geometry).unwrap();
        assert_eq!(
            names(&records),
            [&root, &sibling, &child, &grandchild]
                .iter()
                .map(|header| file_name(&header.key))
                .collect::<Vec<_>>()
        );
        assert_eq!(records[0].header, root);
        assert!(records[0].row);
        assert_eq!(records[0].bytes, root.length_with_row());
        assert!(!records[3].row);
        assert_eq!(records[3].path, compacted);
        assert_eq!(records[3].bytes, grandchild.row_segment);
        for path in &deleted {
            assert!(!path.exists(), "{} survived", path.display());
        }
        assert!(foreign.exists());
        assert!(subdirectory.is_dir());
        assert!(fs::symlink_metadata(&link).unwrap().is_symlink());

        let again = scan(&directory, &IDENTITY, &FINGERPRINT, &geometry).unwrap();
        assert_eq!(again, records);

        let mut index = Index::new(64);
        for record in &records {
            index
                .insert(
                    record.header.key,
                    record.header.parent,
                    record.header.bucket,
                    record.header.tail_pages as usize,
                    record.row,
                    true,
                    (),
                )
                .unwrap();
        }
        let mut tree: Vec<_> = index
            .nodes()
            .map(|node| {
                (
                    node.key,
                    node.parent,
                    node.tail_pages,
                    node.row,
                    node.children(),
                )
            })
            .collect();
        tree.sort_by_key(|node| node.0.consumed_tokens);
        assert_eq!(
            tree,
            vec![
                (root.key, None, 3, true, 2),
                (sibling.key, Some(root.key), 2, true, 0),
                (child.key, Some(root.key), 3, true, 1),
                (grandchild.key, Some(child.key), 3, false, 0),
            ]
        );

        // Deleting the child orphans the grandchild on the next scan.
        fs::remove_file(directory.join(file_name(&child.key))).unwrap();
        let pruned = scan(&directory, &IDENTITY, &FINGERPRINT, &geometry).unwrap();
        assert_eq!(
            names(&pruned),
            [file_name(&root.key), file_name(&sibling.key)]
        );
        assert!(!compacted.exists());

        let file = directory.join("file");
        fs::write(&file, b"").unwrap();
        assert!(scan(&file, &IDENTITY, &FINGERPRINT, &geometry).is_err());
        fs::remove_dir_all(&directory).unwrap();
    }
}
