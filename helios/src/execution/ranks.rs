//! Direct scheduler execution on one CUDA rank or a coordinated rank group.

use std::os::unix::net::UnixStream;

use super::{ExecutionManifest, FrameCompletion, GraphExecutor, TensorParallelExecutor, Upload};
use crate::{Result, Shutdown, cache::Fence};

pub(crate) enum RankExecutor {
    Local(Box<GraphExecutor>),
    Parallel(Box<TensorParallelExecutor>),
}

impl RankExecutor {
    /// # Safety
    /// Prepared graphs, contexts, buffers, and b12x IPC allocations must remain
    /// retained until coordinated drain. Peer sockets identify ranks 1..world.
    pub(crate) unsafe fn new(
        manifest: ExecutionManifest,
        peers: Option<Vec<UnixStream>>,
    ) -> Result<Self> {
        match peers {
            Some(peers) => unsafe { TensorParallelExecutor::new(manifest, peers) }
                .map(|executor| Self::Parallel(Box::new(executor))),
            None => unsafe { GraphExecutor::new(manifest) }
                .map(|executor| Self::Local(Box::new(executor))),
        }
    }

    pub(crate) fn submit(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        match self {
            Self::Local(executor) => executor.submit(graph, uploads, grammar),
            Self::Parallel(executor) => executor.submit(graph, uploads, grammar),
        }
    }

    pub(crate) fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        match self {
            Self::Local(executor) => executor.poll(),
            Self::Parallel(executor) => executor.poll(),
        }
    }

    pub(crate) fn outstanding(&self) -> usize {
        match self {
            Self::Local(executor) => executor.outstanding(),
            Self::Parallel(executor) => executor.outstanding(),
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn manifest(&self) -> &ExecutionManifest {
        match self {
            Self::Local(executor) => executor.manifest(),
            Self::Parallel(executor) => executor.manifest(),
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn can_submit(&self, grammar: bool) -> bool {
        match self {
            Self::Local(executor) => executor.can_submit(grammar),
            Self::Parallel(executor) => executor.can_submit(grammar),
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn next_fence(&self) -> Result<Fence> {
        match self {
            Self::Local(executor) => executor.next_fence(),
            Self::Parallel(executor) => executor.next_fence(),
        }
    }

    pub(crate) fn close(&mut self) -> Shutdown {
        match self {
            Self::Local(executor) => executor.close(),
            Self::Parallel(executor) => executor.close(),
        }
    }
}
