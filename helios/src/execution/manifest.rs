use std::collections::{HashMap, HashSet};

use crate::{BufferSpan, Error, ExecutionError, GraphBundle, Result};

pub const ABI_VERSION: u32 = 2;
pub const SPECULATIVE_ABI_VERSION: u32 = 3;
pub const VECTOR_RECORD_FIELDS: usize = 24;
pub const MAX_OUTPUT_TOKENS: usize = 8;
pub const REQUEST_LANES: usize = 4;
pub const RECORD_FIELDS: usize = 8;
pub const COMPLETION_BYTES: usize = REQUEST_LANES * RECORD_FIELDS * 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dtype {
    Bool,
    Uint8,
    Int32,
    Int64,
    Float32,
    Float16,
    Bfloat16,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Self::Bool | Self::Uint8 => 1,
            Self::Float16 | Self::Bfloat16 => 2,
            Self::Int32 | Self::Float32 => 4,
            Self::Int64 => 8,
        }
    }
}

impl TryFrom<&str> for Dtype {
    type Error = crate::Error;
    fn try_from(value: &str) -> Result<Self> {
        match value {
            "bool" => Ok(Self::Bool),
            "uint8" => Ok(Self::Uint8),
            "int32" => Ok(Self::Int32),
            "int64" => Ok(Self::Int64),
            "float32" => Ok(Self::Float32),
            "float16" => Ok(Self::Float16),
            "bfloat16" => Ok(Self::Bfloat16),
            _ => Err(Error::execution(
                ExecutionError::Manifest,
                format!("unsupported buffer dtype {value:?}"),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphKind {
    Init,
    Restore,
    Snapshot,
    Prefill,
    Decode,
    Speculative,
    Resume,
    Vision,
}

impl GraphKind {
    pub fn samples(self) -> bool {
        matches!(
            self,
            Self::Prefill | Self::Decode | Self::Speculative | Self::Resume
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplingMode {
    General,
}

impl TryFrom<&str> for SamplingMode {
    type Error = crate::Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "general" => Ok(Self::General),
            _ => Err(Error::execution(
                ExecutionError::Manifest,
                format!("unsupported graph sampling mode {value:?}"),
            )),
        }
    }
}

impl TryFrom<&str> for GraphKind {
    type Error = crate::Error;
    fn try_from(value: &str) -> Result<Self> {
        match value {
            "init" => Ok(Self::Init),
            "restore" => Ok(Self::Restore),
            "snapshot" => Ok(Self::Snapshot),
            "prefill" => Ok(Self::Prefill),
            "decode" => Ok(Self::Decode),
            "speculative" => Ok(Self::Speculative),
            "resume" => Ok(Self::Resume),
            "vision" => Ok(Self::Vision),
            _ => Err(Error::execution(
                ExecutionError::Manifest,
                format!("unsupported graph kind {value:?}"),
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BufferDescriptor {
    pub name: String,
    pub address: usize,
    pub nbytes: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct InputBinding {
    pub buffer_name: String,
    pub host_offset: usize,
    pub nbytes: usize,
}

#[derive(Clone, Debug)]
pub struct GraphDescriptor {
    pub name: String,
    pub kind: GraphKind,
    pub sampling_mode: SamplingMode,
    pub handle: usize,
    pub batch_capacity: usize,
    pub token_capacity: usize,
    pub input_bindings: Vec<InputBinding>,
}

#[derive(Clone, Debug)]
pub struct ExecutionManifest {
    pub abi_version: u32,
    pub context: usize,
    pub device: i32,
    pub max_active: usize,
    pub input_frame_bytes: usize,
    pub buffers: Vec<BufferDescriptor>,
    pub graphs: Vec<GraphDescriptor>,
}

impl ExecutionManifest {
    pub fn completion_fields(&self) -> usize {
        if self.abi_version == SPECULATIVE_ABI_VERSION {
            VECTOR_RECORD_FIELDS
        } else {
            RECORD_FIELDS
        }
    }
    pub fn completion_bytes(&self) -> usize {
        REQUEST_LANES * self.completion_fields() * 8
    }

    pub fn buffer(&self, name: &str) -> Result<&BufferDescriptor> {
        self.buffers
            .iter()
            .find(|buffer| buffer.name == name)
            .ok_or_else(|| {
                Error::execution(ExecutionError::Manifest, format!("unknown buffer {name:?}"))
            })
    }

    pub fn validate(&self) -> Result<()> {
        if ![ABI_VERSION, SPECULATIVE_ABI_VERSION].contains(&self.abi_version) {
            return Err(Error::execution(
                ExecutionError::Manifest,
                "unsupported execution ABI version",
            ));
        }
        if !(1..=REQUEST_LANES).contains(&self.max_active) {
            return Err(Error::execution(
                ExecutionError::Manifest,
                "max_active must be 1 through 4",
            ));
        }
        if self.input_frame_bytes == 0
            || !self.input_frame_bytes.is_multiple_of(8)
            || self.input_frame_bytes > isize::MAX as usize
        {
            return Err(Error::execution(
                ExecutionError::Manifest,
                "input_frame_bytes must be a positive, addressable multiple of eight",
            ));
        }
        GraphBundle {
            context: self.context,
            device: self.device,
            graph_execs: self.graphs.iter().map(|graph| graph.handle).collect(),
            buffers: self
                .buffers
                .iter()
                .map(|buffer| BufferSpan {
                    address: buffer.address,
                    nbytes: buffer.nbytes,
                })
                .collect(),
        }
        .validate()?;
        let mut names = HashSet::new();
        for buffer in &self.buffers {
            if buffer.name.is_empty() || !names.insert(&buffer.name) {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    "buffer names must be nonempty and unique",
                ));
            }
            if buffer.shape.is_empty()
                || buffer.shape.contains(&0)
                || buffer.strides.len() != buffer.shape.len()
            {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    format!("invalid shape/strides for {}", buffer.name),
                ));
            }
            let mut stride = 1usize;
            for (extent, actual) in buffer.shape.iter().zip(&buffer.strides).rev() {
                if *actual != stride {
                    return Err(Error::execution(
                        ExecutionError::Manifest,
                        format!("buffer {} must be contiguous row-major", buffer.name),
                    ));
                }
                stride = stride.checked_mul(*extent).ok_or_else(|| {
                    Error::execution(ExecutionError::Manifest, "buffer shape overflow")
                })?;
            }
            if stride.checked_mul(buffer.dtype.size()) != Some(buffer.nbytes)
                || buffer.address % buffer.dtype.size() != 0
            {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    format!(
                        "buffer {} byte extent or alignment does not match dtype/shape",
                        buffer.name
                    ),
                ));
            }
        }
        let mut spans: Vec<_> = self
            .buffers
            .iter()
            .map(|buffer| (buffer.address, buffer.address + buffer.nbytes))
            .collect();
        spans.sort_unstable();
        if spans.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(Error::execution(
                ExecutionError::Manifest,
                "registered device buffer spans overlap",
            ));
        }
        for (name, dtype, shape) in [
            ("frame_index", Dtype::Int32, vec![1]),
            ("submission_sequence", Dtype::Int64, vec![1]),
            (
                "completion_records",
                Dtype::Int64,
                vec![2, 4, self.completion_fields()],
            ),
        ] {
            let buffer = self.buffer(name)?;
            if buffer.dtype != dtype || buffer.shape != shape {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    format!("invalid required buffer layout: {name}"),
                ));
            }
        }
        let mut names = HashSet::new();
        let mut host_spans = HashMap::<&str, (usize, usize)>::new();
        for graph in &self.graphs {
            if graph.name.is_empty() || !names.insert(&graph.name) {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    "graph names must be nonempty and unique",
                ));
            }
            let valid = match graph.kind {
                GraphKind::Speculative => {
                    self.abi_version == SPECULATIVE_ABI_VERSION
                        && [1, 2, 4].contains(&graph.batch_capacity)
                        && (1..=8).contains(&graph.token_capacity)
                }
                GraphKind::Prefill => {
                    graph.batch_capacity == 1
                        && ([1, 16].contains(&graph.token_capacity)
                            || (graph.token_capacity >= 64
                                && graph.token_capacity.is_multiple_of(64)))
                }
                GraphKind::Vision => graph.batch_capacity == 1 && graph.token_capacity > 0,
                _ => [1, 2, 4].contains(&graph.batch_capacity) && graph.token_capacity == 1,
            };
            if !valid {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    format!("unsupported graph capacities: {}", graph.name),
                ));
            }
            let mut bound = HashSet::new();
            for binding in &graph.input_bindings {
                if !bound.insert(binding.buffer_name.as_str())
                    || binding.buffer_name == "completion_records"
                {
                    return Err(Error::execution(
                        ExecutionError::Manifest,
                        format!("duplicate or output-only input binding in {}", graph.name),
                    ));
                }
                let buffer = self.buffer(&binding.buffer_name)?;
                let end = binding
                    .host_offset
                    .checked_add(binding.nbytes)
                    .ok_or_else(|| {
                        Error::execution(ExecutionError::Manifest, "host input span overflow")
                    })?;
                if binding.nbytes != buffer.nbytes
                    || binding.host_offset % buffer.dtype.size() != 0
                    || end > self.input_frame_bytes
                {
                    return Err(Error::execution(
                        ExecutionError::Manifest,
                        format!("invalid host input binding: {}", binding.buffer_name),
                    ));
                }
                let span = (binding.host_offset, end);
                if let Some(previous) = host_spans.insert(&binding.buffer_name, span) {
                    if previous != span {
                        return Err(Error::execution(
                            ExecutionError::Manifest,
                            "one buffer has different host offsets across graphs",
                        ));
                    }
                }
            }
            if !bound.contains("frame_index") || !bound.contains("submission_sequence") {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    format!("graph {} lacks reserved input bindings", graph.name),
                ));
            }
        }
        let mut spans: Vec<_> = host_spans.values().copied().collect();
        spans.sort_unstable();
        if spans.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(Error::execution(
                ExecutionError::Manifest,
                "different input buffers overlap in host frame",
            ));
        }
        Ok(())
    }
}
