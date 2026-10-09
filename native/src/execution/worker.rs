use std::sync::{Mutex, mpsc};
use std::thread::{self, JoinHandle};

use super::{ExecutionManifest, FrameCompletion, GraphExecutor, Upload};
use crate::{Result, Shutdown, cache::Fence};

pub struct OwnedUpload {
    pub buffer: String,
    pub offset: usize,
    pub bytes: Vec<u8>,
}

enum Command {
    Submit {
        graph: usize,
        uploads: Vec<OwnedUpload>,
        grammar: bool,
        reply: mpsc::SyncSender<Result<Option<Fence>>>,
    },
    Poll(mpsc::SyncSender<Result<Option<FrameCompletion>>>),
    Outstanding(mpsc::SyncSender<usize>),
}

struct Control {
    sender: Option<mpsc::SyncSender<Command>>,
    thread: Option<JoinHandle<Shutdown>>,
    shutdown: Option<Shutdown>,
}

/// Synchronous control surface for bootstrap/qualification callers on other threads.
/// A native scheduler can own GraphExecutor directly without this channel boundary.
pub struct ExecutionWorker {
    control: Mutex<Control>,
}

impl ExecutionWorker {
    /// # Safety
    /// The owner must satisfy GraphExecutor::new's resource lifetime contract.
    pub unsafe fn new(manifest: ExecutionManifest) -> Result<Self> {
        manifest.validate()?;
        let (sender, receiver) = mpsc::sync_channel(1);
        let (ready_sender, ready) = mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("sparkgap-execution".into())
            .spawn(move || {
                // SAFETY: The caller retains the manifest's graphs and allocations
                // until this worker reports drain. The executor binds its streams
                // to the declared context on this thread and has exclusive access.
                let mut executor = match unsafe { GraphExecutor::new(manifest) } {
                    Ok(executor) => executor,
                    Err(error) => {
                        let _ = ready_sender.send(Err(error.clone()));
                        return Shutdown {
                            drained: true,
                            error: Some(error),
                        };
                    }
                };
                let _ = ready_sender.send(Ok(()));
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    while let Ok(command) = receiver.recv() {
                        match command {
                            Command::Submit {
                                graph,
                                uploads,
                                grammar,
                                reply,
                            } => {
                                let borrowed: Vec<_> = uploads
                                    .iter()
                                    .map(|upload| Upload {
                                        buffer: &upload.buffer,
                                        offset: upload.offset,
                                        bytes: &upload.bytes,
                                    })
                                    .collect();
                                let _ = reply.send(executor.submit(graph, &borrowed, grammar));
                            }
                            Command::Poll(reply) => {
                                let _ = reply.send(executor.poll());
                            }
                            Command::Outstanding(reply) => {
                                let _ = reply.send(executor.outstanding());
                            }
                        }
                    }
                }));
                let mut shutdown = executor.close();
                if result.is_err() {
                    shutdown
                        .error
                        .get_or_insert("native execution worker panicked".into());
                }
                shutdown
            })
            .map_err(|error| format!("cannot start execution worker: {error}"))?;
        match ready.recv() {
            Ok(Ok(())) => {}
            result => {
                let _ = thread.join();
                return Err(match result {
                    Ok(Err(error)) => error,
                    _ => "execution worker exited during startup".into(),
                });
            }
        }
        Ok(Self {
            control: Mutex::new(Control {
                sender: Some(sender),
                thread: Some(thread),
                shutdown: None,
            }),
        })
    }

    fn request<T>(&self, command: impl FnOnce(mpsc::SyncSender<T>) -> Command) -> Result<T> {
        let control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let sender = control
            .sender
            .as_ref()
            .ok_or("execution worker is closed")?;
        let (reply, receiver) = mpsc::sync_channel(1);
        sender
            .send(command(reply))
            .map_err(|_| "execution worker disconnected")?;
        receiver
            .recv()
            .map_err(|_| "execution worker exited without a response".into())
    }

    pub fn submit(
        &self,
        graph: usize,
        uploads: Vec<OwnedUpload>,
        grammar: bool,
    ) -> Result<Option<Fence>> {
        self.request(|reply| Command::Submit {
            graph,
            uploads,
            grammar,
            reply,
        })?
    }
    pub fn poll(&self) -> Result<Option<FrameCompletion>> {
        self.request(Command::Poll)?
    }
    pub fn outstanding(&self) -> Result<usize> {
        self.request(Command::Outstanding)
    }
    pub fn close(&self) -> Shutdown {
        let mut control = self
            .control
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(shutdown) = &control.shutdown {
            return shutdown.clone();
        }
        control.sender.take();
        let shutdown = match control.thread.take().unwrap().join() {
            Ok(shutdown) => shutdown,
            Err(_) => Shutdown {
                drained: false,
                error: Some("execution worker panicked before reporting drain".into()),
            },
        };
        control.shutdown = Some(shutdown.clone());
        shutdown
    }
}

impl Drop for ExecutionWorker {
    fn drop(&mut self) {
        self.close();
    }
}
