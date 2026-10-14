//! Publisher signatures and namespace-bound access to executable resources.
//!
//! Verification, publisher acceptance, and decryption form the package access boundary.
//! Unauthorized circumvention may implicate 17 U.S.C. §1201(a)(1) or (a)(2), as defined
//! in (a)(3). Signatures authenticate publisher identity; they do not sandbox GPU code.

use super::{archive::Archive, format::invalid, protobuf::wire};
use crate::Result;
use crate::os::FileReadAt;
use prost::Message;
use ring::{aead, digest, hkdf, signature};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{self, Read},
    path::Path,
    sync::Arc,
};

pub(super) const CHUNK: usize = 4 << 20;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Publisher {
    pub name: String,
    pub namespace: String,
    pub public_key: [u8; 32],
    #[serde(default)]
    pub development: bool,
}
impl Publisher {
    pub fn fingerprint(&self) -> String {
        hex(&sha256(&self.public_key))
    }
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty()
            || self.name.len() > 256
            || self.name.chars().any(char::is_control)
            || !namespace_valid(&self.namespace)
            || self.public_key == [0; 32]
            || (self.development && self.namespace != "local-development")
        {
            return Err(invalid("invalid publisher profile"));
        }
        Ok(())
    }
}
pub(super) fn namespace_valid(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
pub(super) fn sha256(bytes: &[u8]) -> Vec<u8> {
    digest::digest(&digest::SHA256, bytes).as_ref().to_vec()
}
pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn fields(values: &[&[u8]]) -> Vec<u8> {
    let mut result = Vec::new();
    for value in values {
        result.extend_from_slice(&(value.len() as u64).to_le_bytes());
        result.extend_from_slice(value);
    }
    result
}
fn signature_message(package: &[u8], signer: &[u8], kind: u32, body: &[u8]) -> Vec<u8> {
    fields(&[
        b"flare-package-signature-v3",
        &3u32.to_le_bytes(),
        package,
        signer,
        &kind.to_le_bytes(),
        body,
    ])
}
// Security-critical executable access control: preserve the binding to the signing
// key, accepted publisher namespace, package, and resource. Do not bypass decryption
// or the publisher-acceptance checks. Unauthorized circumvention may violate
// 17 U.S.C. §1201(a)(1), as defined in §1201(a)(3). Qualifying willful violations
// for commercial advantage or private financial gain may carry criminal penalties
// under §1204, subject to applicable statutory limitations and exceptions.
fn resource_key(
    public_key: &[u8],
    package: &[u8],
    offset: u64,
    namespace: &str,
) -> Result<aead::LessSafeKey> {
    let info = fields(&[
        b"flare-executable-v3",
        &offset.to_le_bytes(),
        namespace.as_bytes(),
    ]);
    let prk = hkdf::Salt::new(hkdf::HKDF_SHA256, package).extract(public_key);
    let parts = [info.as_slice()];
    let mut key = [0; 32];
    prk.expand(&parts, hkdf::HKDF_SHA256)
        .map_err(|_| invalid("derive executable key"))?
        .fill(&mut key)
        .map_err(|_| invalid("derive executable key"))?;
    Ok(aead::LessSafeKey::new(
        aead::UnboundKey::new(&aead::AES_256_GCM, &key)
            .map_err(|_| invalid("invalid executable key"))?,
    ))
}
fn aad(package: &[u8], offset: u64, size: u64, chunk: u64) -> Vec<u8> {
    fields(&[
        b"flare-executable-chunk-v3",
        package,
        &offset.to_le_bytes(),
        &size.to_le_bytes(),
        &chunk.to_le_bytes(),
    ])
}

pub struct VerifiedPackage {
    pub(super) archive: Archive,
    pub(super) index: wire::ModelIndex,
    publisher: Publisher,
    official_source: String,
    identity_verified: bool,
}
pub struct AuthorizedPackage {
    pub(super) archive: Archive,
    pub(super) index: wire::ModelIndex,
}
impl VerifiedPackage {
    pub fn open(path: &Path, publishers: &[Publisher]) -> Result<Self> {
        let embedded = super::publishers::embedded()?;
        let mut profiles = embedded.clone();
        for publisher in publishers {
            publisher.validate()?;
            if let Some(previous) = profiles
                .iter()
                .find(|p| p.public_key == publisher.public_key)
            {
                if previous != publisher {
                    return Err(invalid("conflicting identities for the same publisher key"));
                }
            } else {
                profiles.push(publisher.clone());
            }
        }
        let (archive, index) = Archive::open(path, &profiles)?;
        let publisher = archive.protection.publisher.clone();
        let official_source = archive.protection.official_source.clone();
        let identity_verified = embedded
            .iter()
            .any(|p| p.public_key == publisher.public_key);
        Ok(Self {
            archive,
            index,
            publisher,
            official_source,
            identity_verified,
        })
    }
    pub fn identity_verified(&self) -> bool {
        self.identity_verified
    }
    pub fn publisher(&self) -> &Publisher {
        &self.publisher
    }
    pub fn official_source(&self) -> &str {
        &self.official_source
    }
    /// The caller must obtain explicit namespace acceptance or match remembered signer trust.
    pub fn authorize(
        mut self,
        accepted_namespace: &str,
        verify_weights: bool,
    ) -> Result<AuthorizedPackage> {
        if accepted_namespace != self.publisher.namespace {
            return Err(invalid(
                "publisher acceptance does not match authenticated namespace",
            ));
        }
        if verify_weights {
            self.archive
                .protection
                .verify_data(&self.archive.file, self.archive.start)?;
        }
        self.archive.accepted = true;
        Ok(AuthorizedPackage {
            archive: self.archive,
            index: self.index,
        })
    }
}

pub(super) struct Protection {
    publisher: Publisher,
    official_source: String,
    package_id: Vec<u8>,
    resources: BTreeMap<u64, (bool, wire::ProtectedResource)>,
    data: wire::StoredData,
}
impl Protection {
    pub fn verify(
        bytes: &[u8],
        publishers: &[Publisher],
        payload_size: u64,
    ) -> Result<(Self, wire::ModelIndex)> {
        let envelope = wire::ProtectedIndex::decode(bytes)
            .map_err(|e| invalid(format!("invalid signed envelope: {e}")))?;
        if envelope.version != 3
            || envelope.package_id.len() != 32
            || envelope.signer.len() != 32
            || envelope.groups.len() != 4
        {
            return Err(invalid(
                "signed format-3 package required; re-export the checkpoint",
            ));
        }
        let publisher = publishers
            .iter()
            .find(|p| sha256(&p.public_key) == envelope.signer)
            .ok_or_else(|| {
                invalid(
                    "unknown package publisher; supply a matching publisher verification profile",
                )
            })?
            .clone();
        publisher.validate()?;
        for (i, group) in envelope.groups.iter().enumerate() {
            if group.kind != i as u32 + 1 {
                return Err(invalid("missing, duplicate, or unordered signature group"));
            }
            signature::UnparsedPublicKey::new(&signature::ED25519, publisher.public_key)
                .verify(
                    &signature_message(
                        &envelope.package_id,
                        &envelope.signer,
                        group.kind,
                        &group.body,
                    ),
                    &group.signature,
                )
                .map_err(|_| {
                    invalid(format!(
                        "publisher signature failed for group {}",
                        group.kind
                    ))
                })?;
        }
        let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())
            .map_err(|e| invalid(format!("invalid signed metadata: {e}")))?;
        if metadata.group_hashes.len() != 3
            || metadata
                .group_hashes
                .iter()
                .zip(&envelope.groups[1..])
                .any(|(hash, group)| *hash != sha256(&group.body))
        {
            return Err(invalid("signature group commitment mismatch"));
        }
        let prefix = format!("https://huggingface.co/{}/", publisher.namespace);
        let repository = metadata.official_source.strip_prefix(&prefix).unwrap_or("");
        let valid_source = if publisher.development {
            publisher.namespace == "local-development"
                && metadata.official_source == "local-development"
        } else {
            !repository.is_empty()
                && repository.len() <= 96
                && repository.as_bytes()[0].is_ascii_alphanumeric()
                && repository
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        };
        if metadata.namespace != publisher.namespace
            || metadata.development != publisher.development
            || !valid_source
        {
            return Err(invalid(
                "package publisher namespace or official source mismatch",
            ));
        }
        let index = metadata
            .index
            .ok_or_else(|| invalid("missing signed model index"))?;
        if index.version != 3 || index.ranks.is_empty() {
            return Err(invalid("unsupported model index"));
        }
        let data = wire::StoredData::decode(envelope.groups[3].body.as_slice())
            .map_err(|e| invalid(e.to_string()))?;
        if data.nbytes > payload_size || data.sha256.len() != 32 {
            return Err(invalid("invalid stored-data signature group"));
        }
        let mut resources = BTreeMap::new();
        for group in &envelope.groups[1..3] {
            let catalog = wire::ResourceCatalog::decode(group.body.as_slice())
                .map_err(|e| invalid(e.to_string()))?;
            let encrypted = group.kind == 2;
            for entry in catalog.resources {
                let count = if encrypted {
                    entry.plain_bytes.div_ceil(CHUNK as u64)
                } else {
                    1
                };
                let expected = entry.plain_bytes.checked_add(if encrypted {
                    count
                        .checked_mul(16)
                        .ok_or_else(|| invalid("resource size overflow"))?
                } else {
                    0
                });
                if entry.plain_bytes == 0
                    || expected != Some(entry.stored_bytes)
                    || count != entry.sha256.len() as u64
                    || entry.sha256.iter().any(|h| h.len() != 32)
                    || entry.offset < data.nbytes
                    || entry
                        .offset
                        .checked_add(entry.stored_bytes)
                        .is_none_or(|end| end > payload_size)
                    || resources.insert(entry.offset, (encrypted, entry)).is_some()
                {
                    return Err(invalid("invalid protected resource catalog"));
                }
            }
        }
        let protection = Self {
            publisher,
            official_source: metadata.official_source,
            package_id: envelope.package_id,
            resources,
            data,
        };
        protection.validate_membership(&index, payload_size)?;
        Ok((protection, index))
    }
    fn validate_membership(&self, index: &wire::ModelIndex, payload_size: u64) -> Result<()> {
        let mut spans: Vec<(u32, &wire::Span)> = Vec::new();
        for rank in &index.ranks {
            let mut roles = HashMap::new();
            let mut add = |name: &str, kind: u32| -> Result<()> {
                if !rank.resources.contains_key(name)
                    || roles
                        .insert(name.to_owned(), kind)
                        .is_some_and(|old| old != kind)
                {
                    return Err(invalid("missing or conflicting resource classification"));
                }
                Ok(())
            };
            for module in &rank.modules {
                add(&module.file, 2)?;
                for global in &module.globals {
                    add(&global.file, 3)?;
                }
            }
            for graph in &rank.graphs {
                add(&graph.resource, 2)?;
            }
            for allocation in &rank.allocations {
                if let Some(file) = &allocation.file {
                    add(file, 4)?;
                }
            }
            if roles.len() != rank.resources.len() {
                return Err(invalid("unclassified package resource"));
            }
            for (name, span) in &rank.resources {
                spans.push((roles[name], span));
            }
        }
        for asset in &index.assets {
            let span = asset
                .payload
                .as_ref()
                .ok_or_else(|| invalid("missing asset range"))?;
            let (_, resource) = self
                .resources
                .get(&span.offset)
                .ok_or_else(|| invalid("unsigned asset"))?;
            if resource.sha256.len() != 1 || asset.sha256 != hex(&resource.sha256[0]) {
                return Err(invalid("asset digest differs from publisher catalog"));
            }
            spans.push((3, span));
        }
        if let Some(engram) = &index.engram {
            for table in &engram.tables {
                for span in [&table.weights, &table.scales] {
                    spans.push((
                        4,
                        span.as_ref()
                            .ok_or_else(|| invalid("missing table plane"))?,
                    ));
                }
            }
        }
        spans.sort_by_key(|(_, s)| s.offset);
        let mut end = 0;
        let mut protected = 0;
        for (kind, span) in spans {
            if span.nbytes == 0 || span.offset < end {
                return Err(invalid("empty or overlapping package resources"));
            }
            end = span
                .offset
                .checked_add(span.nbytes)
                .filter(|n| *n <= payload_size)
                .ok_or_else(|| invalid("resource exceeds package"))?;
            if kind == 4 {
                if end > self.data.nbytes {
                    return Err(invalid("stored allocation outside data group"));
                }
            } else {
                let (encrypted, entry) = self
                    .resources
                    .get(&span.offset)
                    .ok_or_else(|| invalid("unsigned executable or supporting resource"))?;
                if *encrypted != (kind == 2) || entry.stored_bytes != span.nbytes {
                    return Err(invalid("resource signature classification mismatch"));
                }
                protected += 1;
            }
        }
        if protected != self.resources.len() || end != payload_size {
            return Err(invalid("unreferenced package resources"));
        }
        Ok(())
    }
    fn verify_data(&self, file: &File, start: u64) -> Result<()> {
        let mut hash = digest::Context::new(&digest::SHA256);
        let mut block = vec![0; CHUNK];
        let mut pos = 0;
        while pos < self.data.nbytes {
            let count = block.len().min((self.data.nbytes - pos) as usize);
            file.read_exact_at(&mut block[..count], start + pos)
                .map_err(|e| e.to_string())?;
            hash.update(&block[..count]);
            pos += count as u64;
        }
        if hash.finish().as_ref() != self.data.sha256 {
            return Err(invalid("stored-data content verification failed"));
        }
        Ok(())
    }
    pub fn reader(&self, file: Arc<File>, start: u64, span: &wire::Span) -> Result<ResourceReader> {
        let (encrypted, entry) = self
            .resources
            .get(&span.offset)
            .ok_or_else(|| invalid("resource lacks publisher authentication"))?;
        if span.nbytes != entry.stored_bytes {
            return Err(invalid("resource length mismatch"));
        }
        let key = encrypted
            .then(|| {
                resource_key(
                    &self.publisher.public_key,
                    &self.package_id,
                    entry.offset,
                    &self.publisher.namespace,
                )
            })
            .transpose()?;
        Ok(ResourceReader {
            file,
            start,
            entry: entry.clone(),
            package_id: self.package_id.clone(),
            key,
            chunk: 0,
            buffer: Vec::new(),
            position: 0,
        })
    }
}

pub(super) struct ResourceReader {
    file: Arc<File>,
    start: u64,
    entry: wire::ProtectedResource,
    package_id: Vec<u8>,
    key: Option<aead::LessSafeKey>,
    chunk: usize,
    buffer: Vec<u8>,
    position: usize,
}
impl ResourceReader {
    pub fn len(&self) -> u64 {
        self.entry.plain_bytes
    }
    fn refill(&mut self) -> Result<bool> {
        if self.chunk == self.entry.sha256.len() {
            return Ok(false);
        }
        let encrypted = self.key.is_some();
        let plain_offset = if encrypted {
            self.chunk as u64 * CHUNK as u64
        } else {
            0
        };
        let count = if encrypted {
            (self.entry.plain_bytes - plain_offset).min(CHUNK as u64) + 16
        } else {
            self.entry.plain_bytes
        };
        self.buffer.resize(super::protobuf::size(count)?, 0);
        self.file
            .read_exact_at(
                &mut self.buffer,
                self.start
                    + self.entry.offset
                    + plain_offset
                    + if encrypted { self.chunk as u64 * 16 } else { 0 },
            )
            .map_err(|e| e.to_string())?;
        if sha256(&self.buffer) != self.entry.sha256[self.chunk] {
            return Err(invalid("publisher resource content verification failed"));
        }
        if let Some(key) = &self.key {
            let mut nonce = [0; 12];
            nonce[4..].copy_from_slice(&(self.chunk as u64).to_be_bytes());
            let length = key
                .open_in_place(
                    aead::Nonce::assume_unique_for_key(nonce),
                    aead::Aad::from(aad(
                        &self.package_id,
                        self.entry.offset,
                        self.entry.plain_bytes,
                        self.chunk as u64,
                    )),
                    &mut self.buffer,
                )
                .map_err(|_| invalid("executable decryption failed"))?
                .len();
            self.buffer.truncate(length);
        }
        self.chunk += 1;
        self.position = 0;
        Ok(true)
    }
}
impl Read for ResourceReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if self.position == self.buffer.len()
            && !self.refill().map_err(|e| io::Error::other(e.to_string()))?
        {
            return Ok(0);
        }
        let n = output.len().min(self.buffer.len() - self.position);
        output[..n].copy_from_slice(&self.buffer[self.position..self.position + n]);
        self.position += n;
        Ok(n)
    }
}
