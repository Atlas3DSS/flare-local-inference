//! The agent's cache of executables and package files, under `~/.cache/flare`.
//!
//! Executables live in `bin/` under their SHA-256 names and run only from there. Package files
//! live in `packages/<directory>/<filename>`: one directory per package and rank, holding each
//! file at its logical length with only the transferred byte ranges written, so a rank's copy is
//! a sparse file. Entries are pinned by the sessions transferring them and by the groups running
//! them; receiving an entry evicts the unpinned entries of its kind.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

use super::{Failure, home};

pub fn default_root() -> PathBuf {
    home().join(".cache/flare")
}

/// The SHA-256 name under which an executable is cached and verified.
pub fn binary_name(bytes: &[u8]) -> String {
    binary_name_from_digest(&Sha256::digest(bytes))
}

fn binary_name_from_digest(digest: &[u8]) -> String {
    format!("flare-{}", super::hex(&digest[..12]))
}

pub fn kind_directory(kind: &str) -> Result<&'static str, Failure> {
    match kind {
        "binary" => Ok("bin"),
        "package" => Ok("packages"),
        _ => Err(format!("unknown cache kind {kind:?}").into()),
    }
}

fn component_valid(part: &str) -> bool {
    !part.is_empty()
        && !part.starts_with('.')
        && !part.contains("..")
        && part
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
}

/// A cache entry path: `bin/NAME` or `packages/DIRECTORY/FILENAME`, with no traversal.
pub fn entry(cache: &Path, directory: &str, name: &str) -> Result<PathBuf, Failure> {
    let components: Vec<_> = name.split('/').collect();
    let expected = if directory == "packages" { 2 } else { 1 };
    if components.len() != expected || !components.iter().all(|part| component_valid(part)) {
        return Err(format!("invalid cache entry name {name:?}").into());
    }
    Ok(cache.join(directory).join(name))
}

/// The unit eviction removes: a package directory, or an executable file.
pub fn unit(cache: &Path, path: &Path) -> PathBuf {
    let packages = cache.join("packages");
    if path.parent().and_then(Path::parent) == Some(packages.as_path()) {
        path.parent().unwrap().to_owned()
    } else {
        path.to_owned()
    }
}

/// Pinned units with their pin counts.
pub type Pins = Arc<Mutex<HashMap<PathBuf, usize>>>;

/// One holder's pins, released when dropped.
pub struct Pinned {
    pins: Pins,
    paths: Vec<PathBuf>,
}

impl Pinned {
    pub fn new(pins: &Pins) -> Self {
        Self {
            pins: pins.clone(),
            paths: Vec::new(),
        }
    }

    pub fn pin(&mut self, unit: PathBuf) {
        *self.pins.lock().unwrap().entry(unit.clone()).or_default() += 1;
        self.paths.push(unit);
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        let mut pins = self.pins.lock().unwrap();
        for path in &self.paths {
            if let Some(count) = pins.get_mut(path) {
                *count -= 1;
                if *count == 0 {
                    pins.remove(path);
                }
            }
        }
    }
}

/// Remove a kind directory's units that nothing pins, leaving transfers in progress.
pub fn evict(directory: &Path, pins: &Pins) -> Result<Vec<PathBuf>, Failure> {
    let pins = pins.lock().unwrap();
    let mut removed = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        let hidden = path
            .file_name()
            .is_none_or(|name| name.to_string_lossy().starts_with('.'));
        if !hidden && !pins.contains_key(&path) {
            if path.is_dir() {
                fs::remove_dir_all(&path)?;
            } else {
                fs::remove_file(&path)?;
            }
            removed.push(path);
        }
    }
    Ok(removed)
}

/// The entry's path when the cache holds it at `size` bytes.
pub fn have(cache: &Path, kind: &str, name: &str, size: u64) -> Result<Option<PathBuf>, Failure> {
    let path = entry(cache, kind_directory(kind)?, name)?;
    Ok(fs::metadata(&path)
        .is_ok_and(|m| m.len() == size)
        .then_some(path))
}

/// Validate byte ranges: increasing, disjoint, non-empty, within `size`.
pub fn validate_ranges(size: u64, ranges: &[(u64, u64)]) -> Result<(), Failure> {
    let mut end = 0;
    for &(offset, nbytes) in ranges {
        if offset < end || nbytes == 0 || offset.checked_add(nbytes).is_none_or(|e| e > size) {
            return Err("ranges must increase, stay disjoint and fit the file".into());
        }
        end = offset + nbytes;
    }
    Ok(())
}

pub fn ranges_from_json(
    size: u64,
    listed: Option<&serde_json::Value>,
) -> Result<Vec<(u64, u64)>, Failure> {
    let Some(listed) = listed.filter(|r| !r.is_null()) else {
        return Ok(vec![(0, size)]);
    };
    let mut ranges = Vec::new();
    for range in listed.as_array().ok_or("ranges must be a list")? {
        let (Some(offset), Some(nbytes)) = (range[0].as_u64(), range[1].as_u64()) else {
            return Err("each range is [offset, bytes]".into());
        };
        ranges.push((offset, nbytes));
    }
    validate_ranges(size, &ranges)?;
    Ok(ranges)
}

/// Total bytes the ranges cover.
pub fn range_bytes(ranges: &[(u64, u64)]) -> u64 {
    ranges.iter().map(|&(_, n)| n).sum()
}

/// Store `ranges` of an entry of `size` bytes, read in order from `source`, as `kind`/`name`.
/// Bytes outside the ranges stay holes. An executable must hash to its name.
pub fn store(
    cache: &Path,
    kind: &str,
    name: &str,
    size: u64,
    ranges: &[(u64, u64)],
    source: &mut dyn Read,
) -> Result<PathBuf, Failure> {
    let path = entry(cache, kind_directory(kind)?, name)?;
    if kind == "binary" && ranges != [(0, size)] {
        return Err("only packages arrive as byte ranges".into());
    }
    fs::create_dir_all(path.parent().unwrap())?;
    let partial = path.with_file_name(format!(
        ".{}.partial",
        path.file_name().unwrap().to_string_lossy()
    ));
    let result = store_partial(&partial, kind, name, size, ranges, source);
    if result.is_err() {
        let _ = fs::remove_file(&partial);
        return result.map(|_| path);
    }
    fs::rename(&partial, &path)?;
    Ok(path)
}

fn store_partial(
    partial: &Path,
    kind: &str,
    name: &str,
    size: u64,
    ranges: &[(u64, u64)],
    source: &mut dyn Read,
) -> Result<(), Failure> {
    let mut output = File::create(partial)?;
    output.set_len(size)?;
    let mut digest = Sha256::new();
    let mut block = vec![0u8; 16 << 20];
    for &(offset, nbytes) in ranges {
        output.seek(SeekFrom::Start(offset))?;
        let mut remaining = nbytes;
        while remaining > 0 {
            let want = remaining.min(block.len() as u64) as usize;
            let count = source.read(&mut block[..want])?;
            if count == 0 {
                return Err(format!("{name}: transfer ended {remaining} bytes early").into());
            }
            if kind == "binary" {
                digest.update(&block[..count]);
            }
            output.write_all(&block[..count])?;
            remaining -= count as u64;
        }
    }
    output.sync_all()?;
    if kind == "binary" {
        let found = binary_name_from_digest(&digest.finalize());
        if found != name {
            return Err(format!("{name}: received bytes hash to {found}").into());
        }
        fs::set_permissions(partial, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Read `ranges` of `file` in order, as a transfer sends them.
pub struct RangeReader<'a> {
    file: &'a File,
    ranges: std::slice::Iter<'a, (u64, u64)>,
    position: u64,
    remaining: u64,
}

impl<'a> RangeReader<'a> {
    pub fn new(file: &'a File, ranges: &'a [(u64, u64)]) -> Self {
        Self {
            file,
            ranges: ranges.iter(),
            position: 0,
            remaining: 0,
        }
    }
}

impl Read for RangeReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        use std::os::unix::fs::FileExt;
        while self.remaining == 0 {
            match self.ranges.next() {
                Some(&(offset, nbytes)) => {
                    self.position = offset;
                    self.remaining = nbytes;
                }
                None => return Ok(0),
            }
        }
        let want = buffer.len().min(self.remaining as usize);
        let count = self.file.read_at(&mut buffer[..want], self.position)?;
        if count == 0 {
            return Err(std::io::Error::other("file ended inside a requested range"));
        }
        self.position += count as u64;
        self.remaining -= count as u64;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_bounded_components() {
        let cache = Path::new("/c");
        assert!(entry(cache, "bin", "flare-00").is_ok());
        assert!(entry(cache, "packages", "export-1-rank1/model.safetensors").is_ok());
        for name in ["", "../x", "a/b", ".hidden"] {
            assert!(entry(cache, "bin", name).is_err(), "{name}");
        }
        for name in [
            "a",
            "a/../escape",
            "a/b/c",
            "a/",
            "a/hidden\\name",
            "a/.hidden",
        ] {
            assert!(entry(cache, "packages", name).is_err(), "{name}");
        }
        assert_eq!(
            unit(cache, Path::new("/c/packages/d/f")),
            PathBuf::from("/c/packages/d")
        );
        assert_eq!(
            unit(cache, Path::new("/c/bin/f")),
            PathBuf::from("/c/bin/f")
        );
    }

    #[test]
    fn ranges_store_sparsely_and_executables_must_hash_to_their_names() {
        use std::os::unix::fs::MetadataExt;
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path();
        let bytes: Vec<u8> = (0..(8u32 << 20)).map(|i| (i % 251) as u8).collect();
        let ranges = [(0, 4096), (5 << 20, 1 << 20)];
        validate_ranges(bytes.len() as u64, &ranges).unwrap();
        assert!(validate_ranges(100, &[(0, 50), (40, 10)]).is_err());
        assert!(validate_ranges(100, &[(0, 0)]).is_err());
        let source = temp.path().join("source");
        fs::write(&source, &bytes).unwrap();
        let file = File::open(&source).unwrap();
        let mut reader = RangeReader::new(&file, &ranges);
        let path = store(
            cache,
            "package",
            "e-8-rank1/m.safetensors",
            bytes.len() as u64,
            &ranges,
            &mut reader,
        )
        .unwrap();
        let copy = fs::read(&path).unwrap();
        assert_eq!(copy.len(), bytes.len());
        for &(offset, nbytes) in &ranges {
            let span = offset as usize..(offset + nbytes) as usize;
            assert_eq!(copy[span.clone()], bytes[span]);
        }
        assert!(copy[8192..1 << 20].iter().all(|&b| b == 0));
        assert!(fs::metadata(&path).unwrap().blocks() * 512 <= 4 << 20);
        assert_eq!(
            have(
                cache,
                "package",
                "e-8-rank1/m.safetensors",
                bytes.len() as u64
            )
            .unwrap(),
            Some(path.clone())
        );
        assert_eq!(
            have(cache, "package", "e-8-rank1/m.safetensors", 1).unwrap(),
            None
        );

        let script = b"#!/bin/sh\necho hi\n";
        let name = binary_name(script);
        let stored = store(
            cache,
            "binary",
            &name,
            script.len() as u64,
            &[(0, script.len() as u64)],
            &mut &script[..],
        )
        .unwrap();
        assert_eq!(
            fs::metadata(&stored).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert!(
            store(
                cache,
                "binary",
                "flare-000000000000000000000000",
                script.len() as u64,
                &[(0, script.len() as u64)],
                &mut &script[..]
            )
            .is_err()
        );
        assert!(
            !cache
                .join("bin/.flare-000000000000000000000000.partial")
                .exists()
        );

        // Eviction removes unpinned units of a kind and keeps pinned ones.
        let pins = Pins::default();
        let mut pinned = Pinned::new(&pins);
        pinned.pin(unit(cache, &path));
        let removed = evict(&cache.join("packages"), &pins).unwrap();
        assert!(removed.is_empty() && path.exists());
        drop(pinned);
        let removed = evict(&cache.join("packages"), &pins).unwrap();
        assert_eq!(removed, vec![cache.join("packages/e-8-rank1")]);
    }
}
