//! Coordinated CUDA ranks sharing one process address space.

use super::{ExecutionManifest, FrameCompletion, GraphExecutor, OwnedUpload, Upload};
use crate::{Error, ExecutionError, Result, Shutdown, cache::Fence};
use std::{
    sync::{Arc, mpsc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const TIMEOUT: Duration = Duration::from_secs(120);
fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Worker, message)
}

struct Step {
    fence: Fence,
    graph: usize,
    grammar: bool,
    uploads: Vec<OwnedUpload>,
}
impl Step {
    fn uploads(&self) -> Vec<Upload<'_>> {
        self.uploads
            .iter()
            .map(|u| Upload {
                buffer: &u.buffer,
                offset: u.offset,
                bytes: &u.bytes,
            })
            .collect()
    }
}
enum Command {
    Prepare(Arc<Step>),
    Commit,
    Drain,
}
enum Reply {
    Ready,
    Complete(FrameCompletion),
    Drained,
}
struct Peer {
    sender: mpsc::Sender<Command>,
    receiver: mpsc::Receiver<Result<Reply>>,
    thread: Option<JoinHandle<Shutdown>>,
    completion: Option<FrameCompletion>,
}
impl Peer {
    fn send(&self, command: Command) -> Result<()> {
        self.sender
            .send(command)
            .map_err(|_| invalid("rank thread disconnected"))
    }
    fn receive(&self) -> Result<Reply> {
        self.receiver
            .recv_timeout(TIMEOUT)
            .map_err(|_| invalid("rank thread response deadline expired"))?
    }
}

pub(crate) struct ThreadedExecutor {
    local: GraphExecutor,
    peers: Vec<Peer>,
    pending: Option<Instant>,
    completion: Option<FrameCompletion>,
    poison: Option<Error>,
    shutdown: Option<Shutdown>,
}
impl ThreadedExecutor {
    pub unsafe fn new(manifest: ExecutionManifest, ranks: Vec<ExecutionManifest>) -> Result<Self> {
        if ![2, 4].contains(&(ranks.len() + 1)) {
            return Err(invalid("tensor parallelism requires two or four ranks"));
        }
        for rank in &ranks {
            rank.validate()?;
            if !same_layout(&manifest, rank) {
                return Err(invalid("rank graph I/O schemas differ"));
            }
        }
        let local = unsafe { GraphExecutor::new(manifest)? };
        let mut result = Self {
            local,
            peers: Vec::new(),
            pending: None,
            completion: None,
            poison: None,
            shutdown: None,
        };
        for (index, manifest) in ranks.into_iter().enumerate() {
            let (sender, commands) = mpsc::channel();
            let (replies, receiver) = mpsc::channel();
            let thread = thread::Builder::new()
                .name(format!("flare-rank-{}", index + 1))
                .spawn(move || {
                    let mut executor = match unsafe { GraphExecutor::new(manifest) } {
                        Ok(executor) => executor,
                        Err(error) => {
                            let _ = replies.send(Err(error.clone()));
                            return Shutdown {
                                drained: true,
                                error: Some(error),
                            };
                        }
                    };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        rank_loop(&mut executor, commands, &replies)
                    }))
                    .unwrap_or_else(|_| Err(invalid("rank execution thread panicked")));
                    match result {
                        Ok(()) => executor.close(),
                        Err(error) => {
                            let _ = replies.send(Err(error.clone()));
                            executor.abandon(error)
                        }
                    }
                })
                .map_err(|e| invalid(format!("cannot start rank thread: {e}")))?;
            let mut peer = Peer {
                sender,
                receiver,
                thread: Some(thread),
                completion: None,
            };
            let ready = match peer.receiver.recv_timeout(TIMEOUT) {
                Ok(reply) => reply,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    Err(invalid("rank initialization thread exited"))
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // A thread inside CUDA may still use resources supplied by the caller.
                    eprintln!("rank initialization deadline expired; terminating the CUDA process");
                    std::process::abort();
                }
            };
            if !matches!(ready, Ok(Reply::Ready)) {
                drop(peer.sender);
                let _ = peer.thread.take().unwrap().join();
                return Err(ready
                    .err()
                    .unwrap_or_else(|| invalid("rank startup did not report readiness")));
            }
            result.peers.push(peer);
        }
        Ok(result)
    }
    pub fn manifest(&self) -> &ExecutionManifest {
        self.local.manifest()
    }
    pub fn outstanding(&self) -> usize {
        usize::from(self.pending.is_some())
    }
    pub fn can_submit(&self, grammar: bool) -> bool {
        self.poison.is_none()
            && self.shutdown.is_none()
            && self.pending.is_none()
            && self.local.can_submit(grammar)
    }
    pub fn next_fence(&self) -> Result<Fence> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.shutdown.is_some() {
            return Err(invalid("rank executor is closed"));
        }
        self.local.next_fence()
    }
    pub fn submit(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        self.next_fence()?;
        if self.pending.is_some() {
            return Ok(None);
        }
        let Some(fence) = self.local.validate_submission(graph, uploads, grammar)? else {
            return Ok(None);
        };
        let result = (|| {
            let step = Arc::new(Step {
                fence,
                graph,
                grammar,
                uploads: uploads
                    .iter()
                    .map(|u| OwnedUpload {
                        buffer: u.buffer.to_owned(),
                        offset: u.offset,
                        bytes: u.bytes.to_vec(),
                    })
                    .collect(),
            });
            for peer in &self.peers {
                peer.send(Command::Prepare(step.clone()))?;
            }
            for peer in &self.peers {
                if !matches!(peer.receive()?, Reply::Ready) {
                    return Err(invalid("rank did not validate the step"));
                }
            }
            self.pending = Some(Instant::now());
            for peer in &self.peers {
                peer.send(Command::Commit)?;
            }
            if self.local.submit(graph, uploads, grammar)? != Some(fence) {
                return Err(invalid("rank-zero fence changed after validation"));
            }
            Ok(Some(fence))
        })();
        if let Err(error) = &result {
            self.poison = Some(error.clone());
        }
        result
    }
    pub fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        let result = self.poll_ranks();
        if let Err(error) = &result {
            self.poison = Some(error.clone());
        }
        result
    }
    fn poll_ranks(&mut self) -> Result<Option<FrameCompletion>> {
        let Some(started) = self.pending else {
            return Ok(None);
        };
        if started.elapsed() > TIMEOUT {
            return Err(invalid("tensor-parallel graph completion deadline expired"));
        }
        if self.completion.is_none() {
            self.completion = self.local.poll()?;
        }
        for peer in &mut self.peers {
            if peer.completion.is_some() {
                continue;
            }
            match peer.receiver.try_recv() {
                Ok(Ok(Reply::Complete(completion))) => peer.completion = Some(completion),
                Ok(Err(error)) => return Err(error),
                Err(mpsc::TryRecvError::Empty) => {}
                _ => return Err(invalid("rank thread failed to return graph completion")),
            }
        }
        if self.completion.is_none() || self.peers.iter().any(|p| p.completion.is_none()) {
            return Ok(None);
        }
        for peer in &mut self.peers {
            super::parallel::agree(
                self.completion.as_ref().unwrap(),
                peer.completion.as_ref().unwrap(),
            )?;
            peer.completion = None;
        }
        self.pending = None;
        Ok(self.completion.take())
    }
    pub fn close(&mut self) -> Shutdown {
        if let Some(shutdown) = &self.shutdown {
            return shutdown.clone();
        }
        let result = (|| {
            while self.pending.is_some() {
                self.poll()?;
                thread::sleep(Duration::from_micros(50));
            }
            if let Some(error) = &self.poison {
                return Err(error.clone());
            }
            for peer in &self.peers {
                peer.send(Command::Drain)?;
            }
            for peer in &mut self.peers {
                if !matches!(peer.receive()?, Reply::Drained) {
                    return Err(invalid("rank thread did not drain"));
                }
                let shutdown = peer
                    .thread
                    .take()
                    .unwrap()
                    .join()
                    .map_err(|_| invalid("rank thread panicked"))?;
                if !shutdown.drained {
                    return Err(invalid("rank thread did not confirm drain"));
                }
                if let Some(error) = shutdown.error {
                    return Err(error);
                }
            }
            Ok(())
        })();
        let shutdown = match result {
            Ok(()) => self.local.close(),
            Err(error) => self.local.abandon(error),
        };
        self.shutdown = Some(shutdown.clone());
        shutdown
    }
}
impl Drop for ThreadedExecutor {
    fn drop(&mut self) {
        self.close();
    }
}

fn rank_loop(
    executor: &mut GraphExecutor,
    commands: mpsc::Receiver<Command>,
    replies: &mpsc::Sender<Result<Reply>>,
) -> Result<()> {
    let send = |reply| {
        replies
            .send(Ok(reply))
            .map_err(|_| invalid("rank coordinator disconnected"))
    };
    send(Reply::Ready)?;
    while let Ok(command) = commands.recv() {
        match command {
            Command::Prepare(step) => {
                let uploads = step.uploads();
                if executor.validate_submission(step.graph, &uploads, step.grammar)?
                    != Some(step.fence)
                {
                    return Err(invalid("rank submission fence differs"));
                }
                send(Reply::Ready)?;
                if !matches!(commands.recv_timeout(TIMEOUT), Ok(Command::Commit)) {
                    return Err(invalid("rank commit missing after preparation"));
                }
                if executor.submit(step.graph, &uploads, step.grammar)? != Some(step.fence) {
                    return Err(invalid("rank fence changed after validation"));
                }
                let started = Instant::now();
                loop {
                    if let Some(completion) = executor.poll()? {
                        send(Reply::Complete(completion))?;
                        break;
                    }
                    if started.elapsed() > TIMEOUT {
                        return Err(invalid("rank graph completion deadline expired"));
                    }
                    thread::yield_now();
                }
            }
            Command::Drain => {
                let shutdown = executor.close();
                if !shutdown.drained {
                    return Err(invalid("rank streams did not drain"));
                }
                if let Some(error) = shutdown.error {
                    return Err(error);
                }
                send(Reply::Drained)?;
                return Ok(());
            }
            Command::Commit => return Err(invalid("rank commit arrived without preparation")),
        }
    }
    Ok(())
}

fn same_layout(a: &ExecutionManifest, b: &ExecutionManifest) -> bool {
    match (&a.engram, &b.engram) {
        (Some(a), Some(b)) if !Arc::ptr_eq(&a.shared, &b.shared) || a.rank == b.rank => {
            return false;
        }
        (Some(_), None) | (None, Some(_)) => return false,
        _ => {}
    }
    a.abi_version == b.abi_version
        && a.max_active == b.max_active
        && a.input_frame_bytes == b.input_frame_bytes
        && a.buffers.len() == b.buffers.len()
        && a.graphs.len() == b.graphs.len()
        && a.buffers.iter().zip(&b.buffers).all(|(a, b)| {
            a.name == b.name
                && a.dtype == b.dtype
                && a.nbytes == b.nbytes
                && a.shape == b.shape
                && a.strides == b.strides
        })
        && a.graphs.iter().zip(&b.graphs).all(|(a, b)| {
            a.name == b.name
                && a.kind == b.kind
                && a.sampling_mode == b.sampling_mode
                && a.batch_capacity == b.batch_capacity
                && a.token_capacity == b.token_capacity
                && a.input_bindings.len() == b.input_bindings.len()
                && a.input_bindings
                    .iter()
                    .zip(&b.input_bindings)
                    .all(|(a, b)| {
                        a.buffer_name == b.buffer_name
                            && a.host_offset == b.host_offset
                            && a.nbytes == b.nbytes
                    })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_layout_compares_bindings_but_not_cuda_addresses() {
        let root = super::super::tests::manifest();
        let mut peer = root.clone();
        peer.context += 1;
        peer.device += 1;
        for buffer in &mut peer.buffers {
            buffer.address += 0x10000;
        }
        for graph in &mut peer.graphs {
            graph.handle += 0x10000;
        }
        assert!(same_layout(&root, &peer));
        peer.graphs[0].input_bindings[0].host_offset += 8;
        assert!(!same_layout(&root, &peer));
        peer = root.clone();
        peer.buffers.pop();
        assert!(!same_layout(&root, &peer));
        peer = root.clone();
        peer.graphs[0].token_capacity += 1;
        assert!(!same_layout(&root, &peer));
    }
}
