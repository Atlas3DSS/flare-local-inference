use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

use crate::execution::{ExecutionManifest, FrameCompletion, OwnedUpload};
use crate::{Error, ExecutionError, Failure, Result};

pub(super) const HELLO: u8 = 1;
pub(super) const PREPARE: u8 = 2;
pub(super) const READY: u8 = 3;
pub(super) const COMMIT: u8 = 4;
pub(super) const COMPLETE: u8 = 5;
pub(super) const DRAIN: u8 = 6;
pub(super) const DRAINED: u8 = 7;
pub(super) const RELEASE: u8 = 8;
pub(super) const FAILED: u8 = 9;
const MAX_PACKET: usize = 64 * 1024 * 1024;
pub(super) const TIMEOUT: Duration = Duration::from_secs(120);

pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Worker, message)
}

fn io(error: std::io::Error) -> Error {
    ExecutionError::Worker(Failure::caused_by(
        "tensor-parallel control socket failed",
        error,
    ))
    .into()
}

pub(super) struct Channel {
    pub stream: UnixStream,
    received: Vec<u8>,
}

impl Channel {
    pub fn new(stream: UnixStream) -> Result<Self> {
        stream.set_read_timeout(Some(TIMEOUT)).map_err(io)?;
        stream.set_write_timeout(Some(TIMEOUT)).map_err(io)?;
        stream.set_nonblocking(false).map_err(io)?;
        Ok(Self {
            stream,
            received: Vec::new(),
        })
    }

    pub fn nonblocking(&self, enabled: bool) -> Result<()> {
        self.stream.set_nonblocking(enabled).map_err(io)
    }

    pub fn idle(&self, enabled: bool) -> Result<()> {
        self.stream
            .set_read_timeout(if enabled { None } else { Some(TIMEOUT) })
            .map_err(io)
    }

    pub fn send(&mut self, tag: u8, data: &[u8]) -> Result<()> {
        if data.len() > MAX_PACKET {
            return Err(invalid(
                "tensor-parallel control packet exceeds its size limit",
            ));
        }
        let mut header = [0; 5];
        header[0] = tag;
        header[1..].copy_from_slice(&(data.len() as u32).to_le_bytes());
        self.stream.write_all(&header).map_err(io)?;
        self.stream.write_all(data).map_err(io)
    }

    pub fn poll(&mut self) -> Result<Option<(u8, Vec<u8>)>> {
        loop {
            let needed = if self.received.len() < 5 {
                5
            } else {
                let size = u32::from_le_bytes(self.received[1..5].try_into().unwrap()) as usize;
                if size > MAX_PACKET {
                    return Err(invalid("oversized tensor-parallel control packet"));
                }
                5 + size
            };
            if self.received.len() == needed && needed >= 5 {
                let bytes = std::mem::take(&mut self.received);
                if bytes[0] == FAILED {
                    return Err(invalid(format!(
                        "tensor-parallel peer failed: {}",
                        String::from_utf8_lossy(&bytes[5..])
                    )));
                }
                return Ok(Some((bytes[0], bytes[5..].to_vec())));
            }
            let mut bytes = [0; 8192];
            let count = bytes.len().min(needed - self.received.len());
            match self.stream.read(&mut bytes[..count]) {
                Ok(0) => return Err(invalid("tensor-parallel peer disconnected")),
                Ok(count) => self.received.extend_from_slice(&bytes[..count]),
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(None),
                Err(error) => return Err(io(error)),
            }
        }
    }

    pub fn receive(&mut self, expected: u8) -> Result<Vec<u8>> {
        let (tag, data) = self
            .poll()?
            .ok_or_else(|| invalid("tensor-parallel control deadline expired"))?;
        if tag != expected {
            return Err(invalid(format!(
                "expected rank message {expected}, received {tag}"
            )));
        }
        Ok(data)
    }
}

pub(super) fn number(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn string(bytes: &mut Vec<u8>, value: &str) {
    number(bytes, value.len());
    bytes.extend_from_slice(value.as_bytes());
}

pub(super) struct Reader<'a>(pub &'a [u8]);
impl<'a> Reader<'a> {
    pub fn bytes(&mut self, count: usize) -> Result<&'a [u8]> {
        if count > self.0.len() {
            return Err(invalid("truncated tensor-parallel control packet"));
        }
        let (value, rest) = self.0.split_at(count);
        self.0 = rest;
        Ok(value)
    }
    pub fn number(&mut self) -> Result<usize> {
        usize::try_from(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
            .map_err(|_| invalid("unaddressable tensor-parallel control value"))
    }
    pub fn end(self) -> Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(invalid("trailing tensor-parallel control bytes"))
        }
    }
}

/// Logical I/O schema only; CUDA addresses and handles never cross processes.
pub(super) fn layout(manifest: &ExecutionManifest) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in [
        2,
        manifest.abi_version as usize,
        manifest.max_active,
        manifest.input_frame_bytes,
        manifest.buffers.len(),
        manifest.graphs.len(),
    ] {
        number(&mut bytes, value);
    }
    for buffer in &manifest.buffers {
        string(&mut bytes, &buffer.name);
        string(&mut bytes, &format!("{:?}", buffer.dtype));
        number(&mut bytes, buffer.nbytes);
        number(&mut bytes, buffer.shape.len());
        for value in buffer.shape.iter().chain(&buffer.strides) {
            number(&mut bytes, *value);
        }
    }
    for graph in &manifest.graphs {
        string(&mut bytes, &graph.name);
        string(
            &mut bytes,
            &format!("{:?}/{:?}", graph.kind, graph.sampling_mode),
        );
        for value in [
            graph.batch_capacity,
            graph.token_capacity,
            graph.input_bindings.len(),
        ] {
            number(&mut bytes, value);
        }
        for binding in &graph.input_bindings {
            string(&mut bytes, &binding.buffer_name);
            number(&mut bytes, binding.host_offset);
            number(&mut bytes, binding.nbytes);
        }
    }
    bytes
}

pub(super) fn uploads(
    manifest: &ExecutionManifest,
    data: &mut Reader<'_>,
) -> Result<Vec<OwnedUpload>> {
    let count = data.number()?;
    if count > manifest.input_frame_bytes || count > data.0.len() / 24 {
        return Err(invalid("invalid rank upload count"));
    }
    let mut uploads = Vec::with_capacity(count);
    for _ in 0..count {
        let buffer = manifest
            .buffers
            .get(data.number()?)
            .ok_or_else(|| invalid("rank upload buffer index out of range"))?;
        let offset = data.number()?;
        let count = data.number()?;
        uploads.push(OwnedUpload {
            buffer: buffer.name.clone(),
            offset,
            bytes: data.bytes(count)?.to_vec(),
        });
    }
    Ok(uploads)
}

pub(super) fn completion(value: &FrameCompletion) -> Vec<u8> {
    let mut bytes = Vec::new();
    number(&mut bytes, value.fence.0 as usize);
    number(&mut bytes, value.graph_index);
    number(&mut bytes, usize::from(value.timing.gpu_ns[0].is_some()));
    number(&mut bytes, value.timing.gpu_ns[0].unwrap_or(0) as usize);
    number(&mut bytes, value.timing.wall_ns as usize);
    number(&mut bytes, usize::from(value.timing.queued));
    for record in value.records.iter() {
        for field in record.fields() {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
    }
    bytes
}

pub(super) fn timing(reader: &mut Reader<'_>) -> Result<super::super::ExecutionTiming> {
    let present = reader.number()?;
    let gpu_ns = reader.number()? as u64;
    let wall_ns = reader.number()? as u64;
    let queued = reader.number()?;
    if present > 1 || queued > 1 || (present == 0 && gpu_ns != 0) {
        return Err(invalid("invalid rank timing interval"));
    }
    Ok(super::super::ExecutionTiming {
        gpu_ns: [
            (present == 1).then_some(gpu_ns),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        wall_ns,
        queued: queued == 1,
    })
}
