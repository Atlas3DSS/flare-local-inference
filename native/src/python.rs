use std::sync::Mutex;
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::execution::{
    BufferDescriptor, Dtype, ExecutionManifest, ExecutionWorker, GraphDescriptor, GraphKind,
    InputBinding, OwnedUpload, SamplingMode,
};
use crate::lifetime::OwnerLease;
#[cfg(feature = "qualification")]
use crate::{BufferSpan, GraphBundle, ReplayWorker};

#[cfg(feature = "qualification")]
#[pyclass(module = "flare_native")]
struct ReplayCompletion {
    #[pyo3(get)]
    ticket: u64,
    #[pyo3(get)]
    graph_index: usize,
    #[pyo3(get)]
    repetitions: u32,
    #[pyo3(get)]
    error: Option<String>,
}

/// Borrow a prepared model's executable graphs and replay them on one native stream.
///
/// The owner must retain every captured allocation, executable, and CUDA context.
/// Graph handles must not be reinstantiated and tensor storage must not be resized
/// or mutated concurrently. Captured PyTorch generator-backed RNG is unsupported.
#[cfg(feature = "qualification")]
#[pyclass(module = "flare_native")]
struct GraphReplayWorker {
    worker: ReplayWorker,
    owner: Mutex<OwnerLease<Py<PyAny>>>,
}

#[cfg(feature = "qualification")]
#[pymethods]
impl GraphReplayWorker {
    #[new]
    #[pyo3(signature = (owner, context, device, graph_execs, buffers, queue_capacity=8))]
    fn new(
        py: Python<'_>,
        owner: Py<PyAny>,
        context: usize,
        device: i32,
        graph_execs: Vec<usize>,
        buffers: Vec<(usize, usize)>,
        queue_capacity: usize,
    ) -> PyResult<Self> {
        if owner.bind(py).is_none() {
            return Err(PyValueError::new_err(
                "owner must retain the prepared graph resources",
            ));
        }
        let bundle = GraphBundle {
            context,
            device,
            graph_execs,
            buffers: buffers
                .into_iter()
                .map(|(address, nbytes)| BufferSpan { address, nbytes })
                .collect(),
        };
        bundle
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let worker = py
            .detach(|| {
                // SAFETY: Preparation supplies ready handles and retains their
                // captured allocations in owner. This reference stays live during
                // startup; OwnerLease retains it afterward until confirmed drain.
                unsafe { ReplayWorker::new(bundle, queue_capacity) }
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            worker,
            owner: Mutex::new(OwnerLease(Some(owner))),
        })
    }

    #[pyo3(signature = (graph_index, repetitions=1))]
    fn submit(&self, graph_index: usize, repetitions: u32) -> PyResult<u64> {
        self.worker
            .submit(graph_index, repetitions)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[pyo3(signature = (timeout_ms=0))]
    fn poll(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<Option<ReplayCompletion>> {
        py.detach(|| self.worker.poll(Duration::from_millis(timeout_ms)))
            .map(|completion| {
                completion.map(|value| ReplayCompletion {
                    ticket: value.ticket,
                    graph_index: value.graph_index,
                    repetitions: value.repetitions,
                    error: value.error.map(|error| error.to_string()),
                })
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[getter]
    fn outstanding(&self) -> usize {
        self.worker.outstanding()
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let shutdown = py.detach(|| self.worker.close());
        if shutdown.drained {
            let owner = self
                .owner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
        match shutdown.error {
            Some(error) => Err(PyRuntimeError::new_err(error.to_string())),
            None => Ok(()),
        }
    }
}

#[cfg(feature = "qualification")]
impl Drop for GraphReplayWorker {
    fn drop(&mut self) {
        let shutdown = self.worker.close();
        if shutdown.drained {
            let owner = self
                .owner
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
    }
}

#[pyfunction]
fn current_context() -> PyResult<(usize, i32)> {
    crate::current_context().map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

fn execution_manifest(value: &Bound<'_, PyAny>) -> PyResult<ExecutionManifest> {
    macro_rules! field {
        ($value:expr, $name:literal, $kind:ty) => {
            $value.get_item($name)?.extract::<$kind>()?
        };
    }
    let mut buffers = Vec::new();
    for buffer in value.get_item("buffers")?.try_iter()? {
        let buffer = buffer?;
        buffers.push(BufferDescriptor {
            name: field!(buffer, "name", String),
            address: field!(buffer, "address", usize),
            nbytes: field!(buffer, "nbytes", usize),
            dtype: Dtype::try_from(field!(buffer, "dtype", String).as_str())
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            shape: field!(buffer, "shape", Vec<usize>),
            strides: field!(buffer, "strides", Vec<usize>),
        });
    }
    let mut graphs = Vec::new();
    for graph in value.get_item("graphs")?.try_iter()? {
        let graph = graph?;
        let mut input_bindings = Vec::new();
        for binding in graph.get_item("input_bindings")?.try_iter()? {
            let binding = binding?;
            input_bindings.push(InputBinding {
                buffer_name: field!(binding, "buffer_name", String),
                host_offset: field!(binding, "host_offset", usize),
                nbytes: field!(binding, "nbytes", usize),
            });
        }
        graphs.push(GraphDescriptor {
            name: field!(graph, "name", String),
            kind: GraphKind::try_from(field!(graph, "kind", String).as_str())
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            sampling_mode: SamplingMode::try_from(field!(graph, "sampling_mode", String).as_str())
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            handle: field!(graph, "handle", usize),
            batch_capacity: field!(graph, "batch_capacity", usize),
            token_capacity: field!(graph, "token_capacity", usize),
            input_bindings,
        });
    }
    Ok(ExecutionManifest {
        abi_version: field!(value, "abi_version", u32),
        context: field!(value, "context", usize),
        device: field!(value, "device", i32),
        max_active: field!(value, "max_active", usize),
        input_frame_bytes: field!(value, "input_frame_bytes", usize),
        buffers,
        graphs,
    })
}

#[pyclass(module = "flare_native")]
struct ExecutionCompletion {
    #[pyo3(get)]
    fence: u64,
    #[pyo3(get)]
    graph_index: usize,
    #[pyo3(get)]
    records: Vec<Vec<i64>>,
    #[pyo3(get)]
    elapsed_ns: Option<u64>,
}

#[pyclass(name = "GraphExecutor", module = "flare_native")]
struct PreparedGraphExecutor {
    worker: ExecutionWorker,
    owner: Mutex<OwnerLease<Py<PyAny>>>,
    drained: std::sync::atomic::AtomicBool,
}

#[pymethods]
impl PreparedGraphExecutor {
    #[new]
    fn new(py: Python<'_>, owner: Py<PyAny>, manifest: &Bound<'_, PyAny>) -> PyResult<Self> {
        if owner.bind(py).is_none() {
            return Err(PyValueError::new_err(
                "owner must retain prepared execution resources",
            ));
        }
        let manifest = execution_manifest(manifest)?;
        manifest
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let worker = py
            .detach(|| {
                // SAFETY: Preparation's owner retains the validated buffers,
                // executable graphs, and context with exclusive native access.
                // Startup holds owner locally; OwnerLease releases it only after
                // close confirms both streams drained, including failed submits.
                unsafe { ExecutionWorker::new(manifest) }
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            worker,
            owner: Mutex::new(OwnerLease(Some(owner))),
            drained: std::sync::atomic::AtomicBool::new(false),
        })
    }

    #[pyo3(signature = (graph_index, uploads, grammar_constrained=false))]
    fn submit(
        &self,
        py: Python<'_>,
        graph_index: usize,
        uploads: Vec<(String, usize, Vec<u8>)>,
        grammar_constrained: bool,
    ) -> PyResult<Option<u64>> {
        let uploads = uploads
            .into_iter()
            .map(|(buffer, offset, bytes)| OwnedUpload {
                buffer,
                offset,
                bytes,
            })
            .collect();
        py.detach(|| {
            self.worker
                .submit(graph_index, uploads, grammar_constrained)
        })
        .map(|fence| fence.map(|fence| fence.0))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn poll(&self, py: Python<'_>) -> PyResult<Option<ExecutionCompletion>> {
        py.detach(|| self.worker.poll())
            .map(|completion| {
                completion.map(|completion| ExecutionCompletion {
                    fence: completion.fence.0,
                    graph_index: completion.graph_index,
                    elapsed_ns: completion.elapsed_ns,
                    records: completion
                        .records
                        .into_iter()
                        .map(|record| record.fields())
                        .collect(),
                })
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[getter]
    fn outstanding(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.worker.outstanding())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
    #[getter]
    fn drained(&self) -> bool {
        self.drained.load(std::sync::atomic::Ordering::Acquire)
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let shutdown = py.detach(|| self.worker.close());
        if shutdown.drained {
            self.drained
                .store(true, std::sync::atomic::Ordering::Release);
            let owner = self
                .owner
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take_after_drain();
            drop(owner);
        }
        match shutdown.error {
            Some(error) => Err(PyRuntimeError::new_err(error.to_string())),
            None => Ok(()),
        }
    }
}

impl Drop for PreparedGraphExecutor {
    fn drop(&mut self) {
        let shutdown = self.worker.close();
        if shutdown.drained {
            let owner = self
                .owner
                .get_mut()
                .unwrap_or_else(|error| error.into_inner())
                .take_after_drain();
            drop(owner);
        }
    }
}

#[pymodule]
fn flare_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "qualification")]
    {
        module.add_class::<GraphReplayWorker>()?;
        module.add_class::<ReplayCompletion>()?;
    }
    module.add_class::<PreparedGraphExecutor>()?;
    module.add_class::<ExecutionCompletion>()?;
    #[cfg(feature = "server")]
    module.add_class::<NativeServer>()?;
    module.add_function(wrap_pyfunction!(current_context, module)?)?;
    Ok(())
}

#[cfg(test)]
mod binding_tests {
    use super::*;

    #[test]
    fn raw_replay_exports_require_qualification_feature() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "flare_native").unwrap();
            flare_native(&module).unwrap();
            for name in ["GraphReplayWorker", "ReplayCompletion"] {
                assert_eq!(
                    module.hasattr(name).unwrap(),
                    cfg!(feature = "qualification")
                );
            }
            assert!(module.hasattr("GraphExecutor").unwrap());
            assert!(module.hasattr("current_context").unwrap());
            assert_eq!(module.hasattr("Server").unwrap(), cfg!(feature = "server"));
        });
    }
}

#[cfg(feature = "server")]
#[pyclass(name = "Server", module = "flare_native")]
struct NativeServer {
    server: crate::server::Server<crate::model::qwen4exp::Qwen4Exp>,
    owner: Mutex<OwnerLease<Py<PyAny>>>,
    drained: std::sync::atomic::AtomicBool,
}
#[cfg(feature = "server")]
#[pymethods]
impl NativeServer {
    #[new]
    fn new(
        py: Python<'_>,
        owner: Py<PyAny>,
        manifest: &Bound<'_, PyAny>,
        config_json: &str,
    ) -> PyResult<Self> {
        if owner.bind(py).is_none() {
            return Err(PyValueError::new_err(
                "owner must retain the complete prepared model",
            ));
        }
        let manifest = execution_manifest(manifest)?;
        let bootstrap = py
            .detach(|| {
                crate::server::Bootstrap::load::<crate::model::qwen4exp::Qwen4Exp>(
                    config_json,
                    &manifest,
                )
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let server = match py.detach(|| {
            // SAFETY: The prepared owner retains every captured allocation and
            // handle and relinquishes external mutation during native execution.
            // It stays live across startup, is retained on an undrained failure,
            // and is otherwise transferred into OwnerLease until confirmed drain.
            unsafe {
                crate::server::Server::<crate::model::qwen4exp::Qwen4Exp>::start(
                    manifest,
                    bootstrap.model,
                    bootstrap.frontend,
                    bootstrap.images,
                    bootstrap.server,
                )
            }
        }) {
            Ok(server) => server,
            Err(error) => {
                if !error.drained {
                    std::mem::forget(owner);
                }
                let exception = PyRuntimeError::new_err(error.message.to_string());
                exception
                    .value(py)
                    .setattr("resources_retained", !error.drained)?;
                return Err(exception);
            }
        };
        Ok(Self {
            server,
            owner: Mutex::new(OwnerLease(Some(owner))),
            drained: std::sync::atomic::AtomicBool::new(false),
        })
    }
    #[getter]
    fn address(&self) -> String {
        self.server.address().to_string()
    }
    #[getter]
    fn healthy(&self) -> bool {
        self.server.healthy()
    }
    #[pyo3(signature = (timeout_ms=1000))]
    fn wait(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<bool> {
        if timeout_ms > 60_000 {
            return Err(PyValueError::new_err(
                "wait timeout must not exceed 60000 ms",
            ));
        }
        let healthy = py.detach(|| self.server.wait(Duration::from_millis(timeout_ms)));
        py.check_signals()?;
        Ok(healthy)
    }
    #[getter]
    fn drained(&self) -> bool {
        self.drained.load(std::sync::atomic::Ordering::Acquire)
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let shutdown = py.detach(|| self.server.close());
        if shutdown.drained {
            self.drained
                .store(true, std::sync::atomic::Ordering::Release);
            let owner = self
                .owner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
        shutdown.error.map_or(Ok(()), |error| {
            Err(PyRuntimeError::new_err(error.to_string()))
        })
    }
}
#[cfg(feature = "server")]
impl Drop for NativeServer {
    fn drop(&mut self) {
        let shutdown = self.server.close();
        if shutdown.drained {
            let owner = self
                .owner
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .take_after_drain();
            drop(owner);
        }
    }
}
