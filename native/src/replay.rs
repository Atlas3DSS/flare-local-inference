use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::{Result, cuda::Stream};

#[derive(Clone, Debug)]
pub struct BufferSpan {
    pub address: usize,
    pub nbytes: usize,
}

#[derive(Clone, Debug)]
pub struct GraphBundle {
    pub context: usize,
    pub device: i32,
    pub graph_execs: Vec<usize>,
    pub buffers: Vec<BufferSpan>,
}

impl GraphBundle {
    pub fn validate(&self) -> Result<()> {
        if self.context == 0 || self.device < 0 {
            return Err(
                "graph bundle requires a nonnull context and nonnegative device ordinal".into(),
            );
        }
        if self.graph_execs.is_empty() || self.graph_execs.contains(&0) {
            return Err("graph bundle requires at least one nonnull graph executable".into());
        }
        if self.graph_execs.iter().collect::<HashSet<_>>().len() != self.graph_execs.len() {
            return Err("graph executable handles must be unique".into());
        }
        for buffer in &self.buffers {
            if buffer.address == 0
                || buffer.nbytes == 0
                || buffer.address.checked_add(buffer.nbytes).is_none()
            {
                return Err(
                    "buffer spans require nonzero addresses and lengths without address overflow"
                        .into(),
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Completion {
    pub ticket: u64,
    pub graph_index: usize,
    pub repetitions: u32,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Shutdown {
    pub drained: bool,
    pub error: Option<String>,
}

struct Submission {
    ticket: u64,
    graph_index: usize,
    repetitions: u32,
}

struct Control {
    sender: Option<mpsc::SyncSender<Submission>>,
    thread: Option<JoinHandle<Shutdown>>,
    next_ticket: u64,
    shutdown: Option<Shutdown>,
}

pub struct ReplayWorker {
    control: Mutex<Control>,
    completions: Mutex<mpsc::Receiver<Completion>>,
    outstanding: Arc<AtomicUsize>,
    capacity: usize,
    graph_count: usize,
}

impl ReplayWorker {
    /// Register borrowed executable handles and storage on their existing CUDA context.
    ///
    /// # Safety
    /// The caller must retain the context, every graph and every captured allocation
    /// until `close` reports `drained = true`. Graphs must be ready for raw replay,
    /// including RNG state, and must not be reinstantiated or mutated during use.
    /// Captured memory-pool lifetimes must permit the submitted variant order;
    /// outputs that survive another replay require independent persistent storage.
    pub unsafe fn new(bundle: GraphBundle, capacity: usize) -> Result<Self> {
        bundle.validate()?;
        Self::spawn(
            bundle.graph_execs.len(),
            capacity,
            move |receiver, sender, ready| {
                let stream = match Stream::open(bundle.context, bundle.device) {
                    Ok(stream) => stream,
                    Err(error) => {
                        let _ = ready.send(Err(error.clone()));
                        return Shutdown {
                            drained: true,
                            error: Some(error),
                        };
                    }
                };
                let _ = ready.send(Ok(()));
                let result = catch_unwind(AssertUnwindSafe(|| {
                    command_loop(receiver, sender, |index, repetitions| {
                        let launch = (0..repetitions)
                            .try_for_each(|_| stream.launch(bundle.graph_execs[index]));
                        let synchronize = stream.synchronize();
                        launch.and(synchronize)
                    })
                }));
                let drain = stream.synchronize();
                let drained = drain.is_ok();
                let destroy = stream.destroy();
                let error = match result {
                    Ok(error) => error,
                    Err(_) => Some("native replay worker panicked".into()),
                };
                Shutdown {
                    drained,
                    error: drain.err().or(error).or(destroy.err()),
                }
            },
        )
    }

    fn spawn(
        graph_count: usize,
        capacity: usize,
        run: impl FnOnce(
            mpsc::Receiver<Submission>,
            mpsc::Sender<Completion>,
            mpsc::SyncSender<Result<()>>,
        ) -> Shutdown
        + Send
        + 'static,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err("submission capacity must be positive".into());
        }
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let (completion_sender, completions) = mpsc::channel();
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("sparkgap-cuda".into())
            .spawn(move || run(receiver, completion_sender, ready_sender))
            .map_err(|e| format!("cannot start native replay worker: {e}"))?;
        match ready_receiver.recv() {
            Ok(Ok(())) => {}
            result => {
                let _ = thread.join();
                return Err(match result {
                    Ok(Err(error)) => error,
                    _ => "native replay worker exited during initialization".into(),
                });
            }
        }
        Ok(Self {
            control: Mutex::new(Control {
                sender: Some(sender),
                thread: Some(thread),
                next_ticket: 1,
                shutdown: None,
            }),
            completions: Mutex::new(completions),
            outstanding: Arc::new(AtomicUsize::new(0)),
            capacity,
            graph_count,
        })
    }

    pub fn submit(&self, graph_index: usize, repetitions: u32) -> Result<u64> {
        if graph_index >= self.graph_count {
            return Err(format!("graph index {graph_index} is out of range"));
        }
        if repetitions == 0 {
            return Err("repetitions must be positive".into());
        }
        let mut control = self
            .control
            .lock()
            .map_err(|_| "replay control lock poisoned")?;
        let sender = control.sender.as_ref().ok_or("replay worker is closed")?;
        if self.outstanding.load(Ordering::Acquire) >= self.capacity {
            return Err("replay submission capacity exhausted; consume completion events".into());
        }
        let ticket = control.next_ticket;
        let next_ticket = ticket
            .checked_add(1)
            .ok_or("replay ticket counter exhausted")?;
        self.outstanding.fetch_add(1, Ordering::AcqRel);
        if let Err(error) = sender.try_send(Submission {
            ticket,
            graph_index,
            repetitions,
        }) {
            self.outstanding.fetch_sub(1, Ordering::AcqRel);
            return Err(format!("cannot submit graph replay: {error}"));
        }
        control.next_ticket = next_ticket;
        Ok(ticket)
    }

    pub fn poll(&self, timeout: Duration) -> Result<Option<Completion>> {
        let receiver = self
            .completions
            .lock()
            .map_err(|_| "replay completion lock poisoned")?;
        let received = if timeout.is_zero() {
            match receiver.try_recv() {
                Ok(value) => Ok(Some(value)),
                Err(mpsc::TryRecvError::Empty) => Ok(None),
                Err(mpsc::TryRecvError::Disconnected) => Err(()),
            }
        } else {
            match receiver.recv_timeout(timeout) {
                Ok(value) => Ok(Some(value)),
                Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
                Err(mpsc::RecvTimeoutError::Disconnected) => Err(()),
            }
        };
        let completion = match received {
            Ok(completion) => completion,
            Err(()) => {
                let outstanding = self.outstanding.load(Ordering::Acquire);
                if outstanding != 0 {
                    return Err(format!(
                        "native replay worker disconnected with {outstanding} outstanding submissions; close the worker to confirm stream drain"
                    ));
                }
                None
            }
        };
        if completion.is_some() {
            self.outstanding.fetch_sub(1, Ordering::AcqRel);
        }
        Ok(completion)
    }

    pub fn close(&self) -> Shutdown {
        let mut control = self.control.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(shutdown) = &control.shutdown {
            return shutdown.clone();
        }
        control.sender.take();
        let shutdown = match control
            .thread
            .take()
            .expect("worker thread exists until close")
            .join()
        {
            Ok(shutdown) => shutdown,
            Err(_) => Shutdown {
                drained: false,
                error: Some("native replay worker panicked before confirming stream drain".into()),
            },
        };
        control.shutdown = Some(shutdown.clone());
        shutdown
    }

    pub fn outstanding(&self) -> usize {
        self.outstanding.load(Ordering::Acquire)
    }
}

impl Drop for ReplayWorker {
    fn drop(&mut self) {
        self.close();
    }
}

fn command_loop(
    receiver: mpsc::Receiver<Submission>,
    sender: mpsc::Sender<Completion>,
    mut execute: impl FnMut(usize, u32) -> Result<()>,
) -> Option<String> {
    let mut failure = None;
    for submission in receiver {
        if failure.is_none() {
            failure = execute(submission.graph_index, submission.repetitions).err();
        }
        let _ = sender.send(Completion {
            ticket: submission.ticket,
            graph_index: submission.graph_index,
            repetitions: submission.repetitions,
            error: failure.clone(),
        });
    }
    failure
}

#[cfg(test)]
mod tests;
