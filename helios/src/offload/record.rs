//! Record files: one checkpoint per file, self-describing, aligned for direct I/O.
//!
//! A record lives at `<directory>/<consumed>-<digest>.ckpt`, with the consumed
//! count as 16 hexadecimal digits and the digest as 32. Its segments are, in
//! order and each starting at a [`BLOCK_BYTES`] boundary:
//!
//! 1. a header of [`HEADER_BYTES`], little-endian: magic `FLARECKP`, format
//!    version, flags, model identity, layout fingerprint, the key, the parent
//!    key, the bucket digest, the page token count, the tail page count, the
//!    rank count, per-rank plane sizes and row bytes, the segment offsets, the
//!    row checksum, and the creation time;
//! 2. a checksum segment of one 64-bit xxh3 per tail page, padded to a block;
//! 3. the page segment, page-major: page `i`, then rank `r`, then that rank's
//!    planes in geometry order, every plane padded to a block;
//! 4. the row segment, rank-major, every row padded to a block.
//!
//! The row exists when the file length covers the row segment; compaction
//! truncates the file at the row segment offset and changes nothing else. Writes
//! go to a temporary name in the same directory and rename into place.
//! Per-page checksums let a restore that reads only a suffix of the pages verify
//! what it read.

use std::{
    path::{Path, PathBuf},
    time::SystemTime,
};

use crate::{Result, cache::PrefixKey};

use super::geometry::{BLOCK_BYTES, Layout};

pub const MAGIC: [u8; 8] = *b"FLARECKP";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = BLOCK_BYTES as usize;
pub const EXTENSION: &str = "ckpt";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankSizes {
    pub planes: Vec<u64>,
    pub row_bytes: u64,
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
    pub fn new(
        layout: &Layout,
        geometry_planes: &[Vec<u64>],
        identity: [u8; 32],
        fingerprint: [u8; 16],
        key: PrefixKey,
        parent: Option<PrefixKey>,
        bucket: [u8; 16],
        tail_pages: u32,
        created: SystemTime,
    ) -> Result<Self> {
        let _ = (
            layout,
            geometry_planes,
            identity,
            fingerprint,
            key,
            parent,
            bucket,
            tail_pages,
            created,
        );
        todo!("derive segment offsets and sizes")
    }

    pub fn encode(&self) -> Result<[u8; HEADER_BYTES]> {
        todo!("serialize little-endian fields into one block")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let _ = bytes;
        todo!("parse and validate magic, version, and field bounds")
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
        todo!("sum of padded plane sizes across ranks")
    }

    pub fn row_record_bytes(&self) -> u64 {
        todo!("sum of padded row bytes across ranks")
    }

    /// File length when the row segment is present.
    pub fn length_with_row(&self) -> u64 {
        self.row_segment + self.row_record_bytes()
    }

    pub fn row_present(&self, file_length: u64) -> bool {
        file_length >= self.length_with_row()
    }

    /// The header's sizes equal the live layout's.
    pub fn matches(&self, layout: &Layout, planes: &[Vec<u64>]) -> bool {
        let _ = (layout, planes);
        todo!("compare page tokens, rank count, plane sizes, row bytes")
    }
}

/// `<identity prefix>-<fingerprint>`: 16 hexadecimal digits of the model
/// identity and 32 of the layout fingerprint.
pub fn directory_name(identity: &[u8; 32], fingerprint: &[u8; 16]) -> String {
    let _ = (identity, fingerprint);
    todo!()
}

pub fn file_name(key: &PrefixKey) -> String {
    let _ = key;
    todo!("<consumed 16 hex>-<digest 32 hex>.ckpt")
}

pub fn temporary_name(key: &PrefixKey) -> String {
    let _ = key;
    todo!(".<file name>.tmp")
}

pub fn parse_file_name(name: &str) -> Option<PrefixKey> {
    let _ = name;
    todo!()
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
pub fn scan(
    directory: &Path,
    identity: &[u8; 32],
    fingerprint: &[u8; 16],
    layout: &Layout,
    planes: &[Vec<u64>],
) -> Result<Vec<Scanned>> {
    let _ = (directory, identity, fingerprint, layout, planes);
    todo!()
}

/// 64-bit xxh3 over `bytes`.
pub fn checksum(bytes: &[u8]) -> u64 {
    twox_hash::XxHash3_64::oneshot(bytes)
}
