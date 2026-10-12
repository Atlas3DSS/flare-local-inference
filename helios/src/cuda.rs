use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::Arc;

use libloading::Library;

use crate::{CudaError, Error, ExecutionError, Failure, Result};

type Handle = *mut c_void;
type GetContext = unsafe extern "C" fn(*mut Handle) -> c_int;
type SetContext = unsafe extern "C" fn(Handle) -> c_int;
type GetDevice = unsafe extern "C" fn(*mut c_int) -> c_int;
type StreamCreate = unsafe extern "C" fn(*mut Handle, c_uint) -> c_int;
type StreamOperation = unsafe extern "C" fn(Handle) -> c_int;
type GraphLaunch = unsafe extern "C" fn(Handle, Handle) -> c_int;
type ErrorText = unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int;
type HostAlloc = unsafe extern "C" fn(*mut Handle, usize, c_uint) -> c_int;
type HostFree = unsafe extern "C" fn(Handle) -> c_int;
type Upload = unsafe extern "C" fn(u64, *const c_void, usize, Handle) -> c_int;
type Download = unsafe extern "C" fn(Handle, u64, usize, Handle) -> c_int;
type EventCreate = unsafe extern "C" fn(*mut Handle, c_uint) -> c_int;
type EventRecord = unsafe extern "C" fn(Handle, Handle) -> c_int;
type EventElapsed = unsafe extern "C" fn(*mut f32, Handle, Handle) -> c_int;
type StreamWait = unsafe extern "C" fn(Handle, Handle, c_uint) -> c_int;
type WaitValue = unsafe extern "C" fn(Handle, u64, u32, c_uint) -> c_int;
type HostPointer = unsafe extern "C" fn(*mut u64, Handle, c_uint) -> c_int;
type StreamCreatePriority = unsafe extern "C" fn(*mut Handle, c_uint, c_int) -> c_int;
type PriorityRange = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;
type HostRegister = unsafe extern "C" fn(Handle, usize, c_uint) -> c_int;
type HostUnregister = unsafe extern "C" fn(Handle) -> c_int;
type DeviceAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
type MemcpyBatch = unsafe extern "C" fn(
    *const u64,
    *const u64,
    *const usize,
    usize,
    *const CopyAttributes,
    *const usize,
    usize,
    Handle,
) -> c_int;

/// `CUmemcpyAttributes`: source access order, source and destination location
/// hints as `CUmemLocation {type, id}`, and flags.
#[repr(C)]
struct CopyAttributes {
    source_order: c_int,
    source_location: [c_int; 2],
    destination_location: [c_int; 2],
    flags: c_uint,
}

/// `CU_MEMCPY_SRC_ACCESS_ORDER_STREAM`: the source is read in stream order.
const SOURCE_ORDER_STREAM: c_int = 1;
/// `CU_MEMHOSTREGISTER_PORTABLE`: the registration is visible to every context.
const HOST_REGISTER_PORTABLE: c_uint = 0x1;
/// `CU_DEVICE_ATTRIBUTE_INTEGRATED`.
const ATTRIBUTE_INTEGRATED: c_int = 18;

/// Most bytes one batched copy may carry.
pub const BATCH_BYTES: usize = 16 << 20;
/// Most entries one batched copy may carry.
pub const BATCH_ENTRIES: usize = 4096;

struct Driver {
    _library: Library,
    get_context: GetContext,
    set_context: SetContext,
    get_device: GetDevice,
    stream_create: StreamCreate,
    stream_synchronize: StreamOperation,
    stream_destroy: StreamOperation,
    graph_launch: GraphLaunch,
    error_name: ErrorText,
    error_string: ErrorText,
    host_alloc: HostAlloc,
    host_free: HostFree,
    upload: Upload,
    download: Download,
    event_create: EventCreate,
    event_record: EventRecord,
    event_elapsed: EventElapsed,
    event_query: StreamOperation,
    event_destroy: StreamOperation,
    stream_wait: StreamWait,
    wait_value: WaitValue,
    host_pointer: HostPointer,
    stream_create_priority: StreamCreatePriority,
    priority_range: PriorityRange,
    host_register: HostRegister,
    host_unregister: HostUnregister,
    device_attribute: DeviceAttribute,
    event_synchronize: StreamOperation,
    memcpy_batch: MemcpyBatch,
}

impl Driver {
    fn load() -> Result<Arc<Self>> {
        // SAFETY: The CUDA driver exports these ABI signatures. The retained
        // Library keeps every resolved function pointer valid for this Driver.
        unsafe {
            let library = Library::new("libcuda.so.1").map_err(|e| {
                ExecutionError::State(Failure::caused_by(
                    format!("cannot load NVIDIA CUDA driver libcuda.so.1: {e}"),
                    e,
                ))
            })?;
            macro_rules! symbol {
                ($name:literal) => {
                    *library.get(concat!($name, "\0").as_bytes()).map_err(|e| {
                        ExecutionError::State(Failure::caused_by(
                            format!("cannot load CUDA entrypoint {}: {e}", $name),
                            e,
                        ))
                    })?
                };
            }
            Ok(Arc::new(Self {
                get_context: symbol!("cuCtxGetCurrent"),
                set_context: symbol!("cuCtxSetCurrent"),
                get_device: symbol!("cuCtxGetDevice"),
                stream_create: symbol!("cuStreamCreate"),
                stream_synchronize: symbol!("cuStreamSynchronize"),
                stream_destroy: symbol!("cuStreamDestroy_v2"),
                graph_launch: symbol!("cuGraphLaunch"),
                error_name: symbol!("cuGetErrorName"),
                error_string: symbol!("cuGetErrorString"),
                host_alloc: symbol!("cuMemHostAlloc"),
                host_free: symbol!("cuMemFreeHost"),
                upload: symbol!("cuMemcpyHtoDAsync_v2"),
                download: symbol!("cuMemcpyDtoHAsync_v2"),
                event_create: symbol!("cuEventCreate"),
                event_record: symbol!("cuEventRecord"),
                event_elapsed: symbol!("cuEventElapsedTime"),
                event_query: symbol!("cuEventQuery"),
                event_destroy: symbol!("cuEventDestroy_v2"),
                stream_wait: symbol!("cuStreamWaitEvent"),
                wait_value: symbol!("cuStreamWaitValue32_v2"),
                host_pointer: symbol!("cuMemHostGetDevicePointer_v2"),
                stream_create_priority: symbol!("cuStreamCreateWithPriority"),
                priority_range: symbol!("cuCtxGetStreamPriorityRange"),
                host_register: symbol!("cuMemHostRegister_v2"),
                host_unregister: symbol!("cuMemHostUnregister"),
                device_attribute: symbol!("cuDeviceGetAttribute"),
                event_synchronize: symbol!("cuEventSynchronize"),
                memcpy_batch: symbol!("cuMemcpyBatchAsync_v2"),
                _library: library,
            }))
        }
    }

    fn check(&self, operation: &str, status: c_int) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let describe = |function: ErrorText| {
            let mut text = std::ptr::null();
            // SAFETY: Driver retains the CUDA library and its error-text functions.
            // CUDA owns the returned NUL-terminated string; it is copied immediately.
            unsafe {
                if function(status, &mut text) == 0 && !text.is_null() {
                    CStr::from_ptr(text).to_string_lossy().into_owned()
                } else {
                    "unknown CUDA error".to_owned()
                }
            }
        };
        Err(ExecutionError::Driver(CudaError {
            operation: operation.to_owned(),
            code: status,
            name: describe(self.error_name),
            description: describe(self.error_string),
        })
        .into())
    }

    fn current(&self) -> Result<(usize, i32)> {
        let mut context = std::ptr::null_mut();
        // SAFETY: The driver owns the ABI and the output pointer is valid for the call.
        self.check("cuCtxGetCurrent", unsafe {
            (self.get_context)(&mut context)
        })?;
        if context.is_null() {
            return Err(Error::execution(
                ExecutionError::State,
                "no current CUDA context; initialize the model's device before registration",
            ));
        }
        let mut device = -1;
        // SAFETY: A current context was verified and device is a valid output pointer.
        self.check("cuCtxGetDevice", unsafe { (self.get_device)(&mut device) })?;
        Ok((context as usize, device))
    }
}

pub fn current_context() -> Result<(usize, i32)> {
    Driver::load()?.current()
}

/// Whether the device owning `context` shares physical memory with the host.
pub(crate) fn device_integrated(context: usize) -> Result<bool> {
    let _ = (context, ATTRIBUTE_INTEGRATED);
    todo!("set the context current and query CU_DEVICE_ATTRIBUTE_INTEGRATED")
}

/// Page-locked registration of process memory, released on drop.
pub(crate) struct HostRegistration {
    driver: Arc<Driver>,
    address: usize,
    nbytes: usize,
}

impl HostRegistration {
    pub(crate) fn address(&self) -> usize {
        self.address
    }
    pub(crate) fn nbytes(&self) -> usize {
        self.nbytes
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        // SAFETY: The registration was made by `register_host` on this range.
        unsafe {
            (self.driver.host_unregister)(self.address as Handle);
        }
    }
}

/// Register `nbytes` at `address`, which must be page aligned, as portable
/// page-locked memory with `context` current. The range stays mapped and
/// unchanged in size until the registration is dropped.
pub(crate) unsafe fn register_host(
    context: usize,
    address: usize,
    nbytes: usize,
) -> Result<HostRegistration> {
    let _ = (context, address, nbytes, HOST_REGISTER_PORTABLE);
    todo!("cuMemHostRegister_v2 with the portable flag")
}

pub(crate) struct Stream {
    driver: Arc<Driver>,
    handle: usize,
}

impl Stream {
    pub(crate) fn open(context: usize, expected_device: i32) -> Result<Self> {
        let driver = Driver::load()?;
        driver.check("cuCtxSetCurrent", unsafe {
            (driver.set_context)(context as Handle)
        })?;
        let (_, device) = driver.current()?;
        if device != expected_device {
            return Err(Error::execution(
                ExecutionError::State,
                format!(
                    "CUDA context belongs to device {device}, expected device {expected_device}"
                ),
            ));
        }
        let mut handle = std::ptr::null_mut();
        driver.check("cuStreamCreate", unsafe {
            (driver.stream_create)(&mut handle, 1)
        })?;
        Ok(Self {
            driver,
            handle: handle as usize,
        })
    }

    /// A stream at the lowest priority the context supports, for copies that
    /// must yield to model work.
    pub(crate) fn open_lowest_priority(context: usize, expected_device: i32) -> Result<Self> {
        let _ = (context, expected_device, SOURCE_ORDER_STREAM);
        todo!("cuCtxGetStreamPriorityRange then cuStreamCreateWithPriority with the least priority")
    }

    /// Issue one batched copy of `destinations[i] <- sources[i]` for `sizes[i]`
    /// bytes each, in stream order on both sides. Every address is a device
    /// address or a registered host address. The batch carries at most
    /// [`BATCH_BYTES`] and [`BATCH_ENTRIES`]; larger batches are rejected.
    ///
    /// # Safety
    /// Every range stays valid and, for host sources, unchanged until the
    /// stream passes the copy.
    pub(crate) unsafe fn copy_batch(
        &self,
        destinations: &[u64],
        sources: &[u64],
        sizes: &[usize],
    ) -> Result<()> {
        let _ = (destinations, sources, sizes);
        todo!("cuMemcpyBatchAsync_v2 with one stream-ordered attribute")
    }

    pub(crate) fn launch(&self, graph: usize) -> Result<()> {
        self.driver.check("cuGraphLaunch", unsafe {
            (self.driver.graph_launch)(graph as Handle, self.handle as Handle)
        })
    }

    pub(crate) fn host_buffer(&self, nbytes: usize) -> Result<PinnedHost> {
        self.allocate_host(nbytes, 0)
    }

    pub(crate) fn mapped_host_buffer(&self, nbytes: usize) -> Result<PinnedHost> {
        self.allocate_host(nbytes, 3)
    }

    fn allocate_host(&self, nbytes: usize, flags: u32) -> Result<PinnedHost> {
        let mut handle = std::ptr::null_mut();
        self.driver.check("cuMemHostAlloc", unsafe {
            (self.driver.host_alloc)(&mut handle, nbytes, flags)
        })?;
        unsafe {
            std::ptr::write_bytes(handle.cast::<u8>(), 0, nbytes);
        }
        Ok(PinnedHost {
            driver: self.driver.clone(),
            handle: handle as usize,
            nbytes,
        })
    }

    pub(crate) fn event(&self) -> Result<Event> {
        self.event_with_timing(false)
    }
    pub(crate) fn event_with_timing(&self, timing: bool) -> Result<Event> {
        let mut handle = std::ptr::null_mut();
        self.driver.check("cuEventCreate", unsafe {
            (self.driver.event_create)(&mut handle, if timing { 0 } else { 2 })
        })?;
        Ok(Event {
            driver: self.driver.clone(),
            handle: handle as usize,
        })
    }

    pub(crate) fn record(&self, event: &Event) -> Result<()> {
        unsafe { self.record_borrowed(event.handle) }
    }

    /// The event belongs to this context and outlives all queued work.
    pub(crate) unsafe fn record_borrowed(&self, event: usize) -> Result<()> {
        self.driver.check("cuEventRecord", unsafe {
            (self.driver.event_record)(event as Handle, self.handle as Handle)
        })
    }

    /// The mapped flag remains allocated until this stream drains.
    pub(crate) unsafe fn wait_value(&self, flag: usize, generation: u32) -> Result<()> {
        self.driver.check("cuStreamWaitValue32_v2", unsafe {
            (self.driver.wait_value)(self.handle as Handle, flag as u64, generation, 1)
        })
    }

    pub(crate) fn mapped_address(&self, host: &PinnedHost) -> Result<usize> {
        let mut address = 0;
        self.driver.check("cuMemHostGetDevicePointer", unsafe {
            (self.driver.host_pointer)(&mut address, host.handle as Handle, 0)
        })?;
        Ok(address as usize)
    }

    pub(crate) fn wait(&self, event: &Event) -> Result<()> {
        self.driver.check("cuStreamWaitEvent", unsafe {
            (self.driver.stream_wait)(self.handle as Handle, event.handle as Handle, 0)
        })
    }

    /// The pinned source remains unchanged until this stream passes the transfer.
    pub(crate) unsafe fn upload(
        &self,
        device: usize,
        host: &PinnedHost,
        offset: usize,
        nbytes: usize,
    ) -> Result<()> {
        self.driver.check("cuMemcpyHtoDAsync_v2", unsafe {
            (self.driver.upload)(
                device as u64,
                (host.handle + offset) as *const c_void,
                nbytes,
                self.handle as Handle,
            )
        })
    }

    /// The pinned destination is not read or reused until this stream passes the transfer.
    pub(crate) unsafe fn download(
        &self,
        host: &PinnedHost,
        device: usize,
        nbytes: usize,
    ) -> Result<()> {
        self.driver.check("cuMemcpyDtoHAsync_v2", unsafe {
            (self.driver.download)(
                host.handle as Handle,
                device as u64,
                nbytes,
                self.handle as Handle,
            )
        })
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        self.driver.check("cuStreamSynchronize", unsafe {
            (self.driver.stream_synchronize)(self.handle as Handle)
        })
    }

    pub(crate) fn destroy(mut self) -> Result<()> {
        let handle = std::mem::take(&mut self.handle);
        self.driver.check("cuStreamDestroy_v2", unsafe {
            (self.driver.stream_destroy)(handle as Handle)
        })
    }
}

pub(crate) struct Event {
    driver: Arc<Driver>,
    handle: usize,
}

impl Event {
    pub(crate) fn elapsed_ns(&self, start: &Event) -> Result<u64> {
        let mut millis = 0.0;
        self.driver.check("cuEventElapsedTime", unsafe {
            (self.driver.event_elapsed)(&mut millis, start.handle as Handle, self.handle as Handle)
        })?;
        if !millis.is_finite() || millis <= 0.0 {
            return Err(Error::execution(
                ExecutionError::Completion,
                "invalid speculative CUDA event duration",
            ));
        }
        Ok((f64::from(millis) * 1_000_000.0).round() as u64)
    }

    /// Block the calling thread until the event's work completes.
    pub(crate) fn synchronize(&self) -> Result<()> {
        self.driver.check("cuEventSynchronize", unsafe {
            (self.driver.event_synchronize)(self.handle as Handle)
        })
    }

    pub(crate) fn completed(&self) -> Result<bool> {
        let status = unsafe { (self.driver.event_query)(self.handle as Handle) };
        if status == 600 {
            return Ok(false);
        }
        self.driver.check("cuEventQuery", status)?;
        Ok(true)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            (self.driver.event_destroy)(self.handle as Handle);
        }
    }
}

pub(crate) struct PinnedHost {
    driver: Arc<Driver>,
    handle: usize,
    nbytes: usize,
}

impl PinnedHost {
    pub(crate) fn address(&self) -> usize {
        self.handle
    }
    pub(crate) fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.handle as *const u8, self.nbytes) }
    }
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.handle as *mut u8, self.nbytes) }
    }
}

impl Drop for PinnedHost {
    fn drop(&mut self) {
        unsafe {
            (self.driver.host_free)(self.handle as Handle);
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.handle != 0 {
            // Explicit synchronization reports errors; Drop only releases the stream handle.
            unsafe {
                (self.driver.stream_destroy)(self.handle as Handle);
            }
        }
    }
}
