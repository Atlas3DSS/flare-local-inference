use serde::Deserialize;

use super::{
    NativeResources,
    format::{Reference, invalid},
};
use crate::{
    Result,
    execution::{BufferDescriptor, ExecutionManifest, GraphDescriptor, InputBinding},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Execution {
    abi_version: u32,
    max_active: usize,
    input_frame_bytes: usize,
    buffers: Vec<Buffer>,
    graphs: Vec<Graph>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Buffer {
    name: String,
    reference: Reference,
    nbytes: usize,
    dtype: String,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Graph {
    name: String,
    kind: String,
    sampling_mode: String,
    batch_capacity: usize,
    token_capacity: usize,
    input_bindings: Vec<Binding>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    buffer_name: String,
    host_offset: usize,
    nbytes: usize,
}

impl NativeResources {
    pub fn execution_manifest(&self) -> Result<ExecutionManifest> {
        let value = self
            .execution
            .as_ref()
            .ok_or_else(|| invalid("package has no serving contract"))?;
        let execution: Execution =
            serde_json::from_value(value.clone()).map_err(|e| invalid(e.to_string()))?;
        let buffers = execution
            .buffers
            .into_iter()
            .map(|b| {
                let allocation = self.allocation(&b.reference.allocation)?;
                if b.reference
                    .offset
                    .checked_add(b.nbytes)
                    .is_none_or(|end| end > allocation.nbytes)
                {
                    return Err(invalid("execution buffer exceeds its allocation"));
                }
                Ok(BufferDescriptor {
                    name: b.name,
                    address: allocation.address + b.reference.offset,
                    nbytes: b.nbytes,
                    dtype: b.dtype.as_str().try_into()?,
                    shape: b.shape,
                    strides: b.strides,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let graphs = execution
            .graphs
            .into_iter()
            .map(|g| {
                let index = self
                    .names
                    .iter()
                    .position(|name| *name == g.name)
                    .ok_or_else(|| invalid("execution references an unknown graph"))?;
                Ok(GraphDescriptor {
                    name: g.name,
                    kind: g.kind.as_str().try_into()?,
                    sampling_mode: g.sampling_mode.as_str().try_into()?,
                    handle: self.executables[index] as usize,
                    batch_capacity: g.batch_capacity,
                    token_capacity: g.token_capacity,
                    input_bindings: g
                        .input_bindings
                        .into_iter()
                        .map(|b| InputBinding {
                            buffer_name: b.buffer_name,
                            host_offset: b.host_offset,
                            nbytes: b.nbytes,
                        })
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let manifest = ExecutionManifest {
            abi_version: execution.abi_version,
            context: self.context as usize,
            device: self.device,
            max_active: execution.max_active,
            input_frame_bytes: execution.input_frame_bytes,
            buffers,
            graphs,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}
