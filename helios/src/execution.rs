//! Fixed-capacity graph execution over two pinned transport frames.

mod manifest;
mod parallel;
mod rank_stream;
mod ranks;
pub mod rendezvous;
#[cfg(feature = "roce")]
pub mod roce;
pub mod roce_fabric;
mod threaded;
pub use rank_stream::RankStream;
pub use ranks::RankPeers;
mod transport;
mod worker;
pub use parallel::{TensorParallelExecutor, serve_rank};
pub(crate) use ranks::RankExecutor;
pub use transport::{
    CompletionRecord, ExecutionTiming, FrameCompletion, GraphExecutor, TokenVector, Upload,
};
pub use worker::{ExecutionWorker, OwnedUpload};

#[cfg(test)]
mod tests;

pub use manifest::{
    ABI_VERSION, BATCH_CAPACITIES, BufferDescriptor, CollectiveHealth, Dtype, ExecutionManifest,
    GraphDescriptor, GraphKind, InputBinding, MAX_OUTPUT_TOKENS, RECORD_FIELDS, REQUEST_LANES,
    SPECULATIVE_ABI_VERSION, SamplingMode, VECTOR_RECORD_FIELDS,
};
