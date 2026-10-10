//! Fixed-capacity graph execution over two pinned transport frames.

mod manifest;
mod parallel;
mod ranks;
mod transport;
mod worker;
pub use parallel::{TensorParallelExecutor, serve_rank};
pub(crate) use ranks::RankExecutor;
pub use transport::{CompletionRecord, FrameCompletion, GraphExecutor, TokenVector, Upload};
pub use worker::{ExecutionWorker, OwnedUpload};

#[cfg(test)]
mod tests;

pub use manifest::{
    ABI_VERSION, BufferDescriptor, COMPLETION_BYTES, Dtype, ExecutionManifest, GraphDescriptor,
    GraphKind, InputBinding, MAX_OUTPUT_TOKENS, RECORD_FIELDS, REQUEST_LANES,
    SPECULATIVE_ABI_VERSION, SamplingMode, VECTOR_RECORD_FIELDS,
};
