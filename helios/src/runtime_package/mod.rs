//! Native ownership and reconstruction of prepared CUDA graph packages.

pub mod driver;
mod format;
mod serving;

use std::collections::HashMap;
use std::ffi::CString;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::Arc;

use crate::{BufferSpan, GraphBundle, Result, startup::StartupProgress};
use driver::{
    Access, AllocationProperties, CopyEndpoint, CopyParams, Driver, Handle, KernelParams, Location,
    MemsetParams, TensorMap,
};
use format::{Blob, Memory, Node, Package, Reference, Relocation, RelocationKind, invalid};

struct Storage {
    base: u64,
    address: u64,
    nbytes: usize,
    memory: Memory,
    fixed: bool,
}

/// Owns a CUDA context and every allocation, module, and graph loaded into it.
/// Borrowed handles remain valid until this owner is dropped, after replay drains.
pub struct NativeResources {
    driver: Arc<Driver>,
    context: Handle,
    device: i32,
    allocations: HashMap<String, Storage>,
    modules: Vec<Handle>,
    templates: Vec<Handle>,
    executables: Vec<Handle>,
    names: Vec<String>,
    reservations: Vec<(u64, usize)>,
    execution: Option<serde_json::Value>,
}

impl NativeResources {
    /// Load compiled code and persistent storage, then bind and instantiate graphs.
    ///
    /// # Safety
    /// The package must be trusted. Its kernels and declared bindings must obey
    /// CUDA's memory and execution contracts, including any mapped host access.
    /// This method makes the owner's context current on the calling thread.
    pub unsafe fn load(path: impl AsRef<Path>, ordinal: i32) -> Result<Self> {
        // SAFETY: The caller supplies the same trusted-package contract.
        unsafe { Self::load_with_progress(path, ordinal, &StartupProgress::silent()) }
    }

    /// Load a model while reporting memory, kernel, and execution preparation.
    ///
    /// # Safety
    /// The package must satisfy the trust and lifetime requirements of [`Self::load`].
    pub unsafe fn load_with_progress(
        path: impl AsRef<Path>,
        ordinal: i32,
        progress: &StartupProgress,
    ) -> Result<Self> {
        let phase = progress.phase("Read model", None);
        let path = path.as_ref();
        let package: Package = serde_json::from_reader(
            File::open(path.join("package.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| invalid(format!("invalid runtime package: {e}")))?;
        package.validate()?;
        phase.finish("validated");
        let phase = progress.phase("Initialize GPU", None);
        let driver = Arc::new(Driver::load()?);
        // SAFETY: Driver signatures are checked against cuda.h; output storage is
        // valid, and CUDA initialization precedes all context/resource operations.
        unsafe {
            driver.check("cuInit", (driver.init)(0))?;
            let mut device = 0;
            driver.check("cuDeviceGet", (driver.device)(&mut device, ordinal))?;
            for (attribute, expected) in [75, 76].into_iter().zip(package.architecture) {
                let mut value = 0;
                driver.check(
                    "cuDeviceGetAttribute",
                    (driver.device_attribute)(&mut value, attribute, device),
                )?;
                if value != expected {
                    return Err(invalid("package GPU architecture does not match device"));
                }
            }
            let mut context = null_mut();
            driver.check(
                "cuCtxCreate_v2",
                (driver.context_create)(&mut context, package.context.flags, device),
            )?;
            let mut owner = Self {
                driver,
                context,
                device,
                allocations: HashMap::new(),
                modules: vec![],
                templates: vec![],
                executables: vec![],
                names: vec![],
                reservations: vec![],
                execution: package.execution.clone(),
            };
            phase.finish(format!(
                "GPU {ordinal} · SM{}{}",
                package.architecture[0], package.architecture[1]
            ));
            owner.populate(path, &package, progress)?;
            Ok(owner)
        }
    }

    fn address(&self, reference: &Reference) -> u64 {
        self.allocations[&reference.allocation].address + reference.offset as u64
    }

    fn relocate(&self, data: &mut [u8], relocation: &Relocation, offset: usize) -> Result<()> {
        let address = self.address(&relocation.target);
        match relocation.kind {
            RelocationKind::Pointer => {
                data[offset..offset + 8].copy_from_slice(&address.to_le_bytes())
            }
            RelocationKind::TensorMap => {
                let mut descriptor = TensorMap(data[offset..offset + 128].try_into().unwrap());
                // SAFETY: TensorMap has CUDA's required 64-byte alignment. The
                // descriptor and replacement allocation were validated before use.
                self.driver.check("cuTensorMapReplaceAddress", unsafe {
                    (self.driver.tensor_map_replace)(&mut descriptor, address as Handle)
                })?;
                data[offset..offset + 128].copy_from_slice(&descriptor.0);
            }
        }
        Ok(())
    }

    fn bind(&self, blob: &Blob) -> Result<Vec<u8>> {
        let mut data = blob.data.clone();
        for relocation in &blob.relocations {
            self.relocate(&mut data, relocation, relocation.offset)?;
        }
        Ok(data)
    }

    unsafe fn populate(
        &mut self,
        root: &Path,
        package: &Package,
        progress: &StartupProgress,
    ) -> Result<()> {
        let driver = self.driver.clone();
        let bytes = package.allocations.iter().fold(0u64, |total, allocation| {
            total.saturating_add(allocation.nbytes as u64)
        });
        let phase = progress.phase("Load weights", Some(bytes));
        let mut loaded = 0u64;
        phase.detail(format!(
            "0.0 / {:.1} GiB",
            bytes as f64 / (1u64 << 30) as f64
        ));
        // SAFETY: Package validation bounds all indexes and relocation fields.
        // Every CUDA resource is immediately retained by self for error cleanup.
        unsafe {
            let footprint = package
                .allocations
                .iter()
                .filter(|allocation| matches!(allocation.memory, Memory::Device))
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.nbytes)
                })
                .ok_or_else(|| invalid("package device footprint overflows address space"))?;
            let (mut free, mut total) = (0, 0);
            driver.check(
                "cuMemGetInfo_v2",
                (driver.memory_info)(&mut free, &mut total),
            )?;
            if footprint > free {
                return Err(invalid(format!(
                    "package needs {footprint} device bytes; only {free} are free"
                )));
            }
            for [limit, value] in &package.context.limits {
                driver.check(
                    "cuCtxSetLimit",
                    (driver.context_limit)(*limit as i32, *value),
                )?;
            }
            driver.check(
                "cuCtxSetCacheConfig",
                (driver.context_cache)(package.context.cache),
            )?;
            for reservation in &package.reservations {
                let mut address = 0;
                driver.check(
                    "cuMemAddressReserve",
                    (driver.reserve)(&mut address, reservation.nbytes, 0, reservation.address, 0),
                )?;
                self.reservations.push((address, reservation.nbytes));
                if address != reservation.address {
                    return Err(invalid(format!(
                        "CUDA address collision: requested {:#x}, reserved {address:#x}",
                        reservation.address
                    )));
                }
            }
            for allocation in &package.allocations {
                let mut base = 0;
                let size = allocation.nbytes + allocation.alignment - 1;
                let address = if let Some(address) = allocation.address {
                    match allocation.memory {
                        Memory::Device => {
                            let properties = AllocationProperties {
                                kind: 1,
                                handles: 0,
                                location: Location {
                                    kind: 1,
                                    device: self.device,
                                },
                                windows: null_mut(),
                                flags: [0; 8],
                            };
                            let mut physical = 0;
                            driver.check(
                                "cuMemCreate",
                                (driver.physical_create)(
                                    &mut physical,
                                    allocation.nbytes,
                                    &properties,
                                    0,
                                ),
                            )?;
                            let mapped = (driver.map)(address, allocation.nbytes, 0, physical, 0);
                            (driver.physical_release)(physical);
                            driver.check("cuMemMap", mapped)?;
                        }
                        Memory::MappedHost => {
                            let mapped = libc::mmap(
                                address as Handle,
                                allocation.nbytes,
                                libc::PROT_READ | libc::PROT_WRITE,
                                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE,
                                -1,
                                0,
                            );
                            if mapped != address as Handle {
                                return Err(invalid(format!(
                                    "host address collision at {address:#x}: {}",
                                    std::io::Error::last_os_error()
                                )));
                            }
                            let status = (driver.host_register)(mapped, allocation.nbytes, 2);
                            if status != 0 {
                                libc::munmap(mapped, allocation.nbytes);
                            }
                            driver.check("cuMemHostRegister_v2", status)?;
                        }
                    }
                    address
                } else {
                    driver.check("cuMemAlloc_v2", (driver.allocate)(&mut base, size))?;
                    (base + allocation.alignment as u64 - 1) & !(allocation.alignment as u64 - 1)
                };
                self.allocations.insert(
                    allocation.name.clone(),
                    Storage {
                        base,
                        address,
                        nbytes: allocation.nbytes,
                        memory: allocation.memory,
                        fixed: allocation.address.is_some(),
                    },
                );
                if allocation.address.is_some() {
                    match allocation.memory {
                        Memory::Device => {
                            let access = Access {
                                location: Location {
                                    kind: 1,
                                    device: self.device,
                                },
                                flags: 3,
                            };
                            driver.check(
                                "cuMemSetAccess",
                                (driver.access)(address, allocation.nbytes, &access, 1),
                            )?;
                        }
                        Memory::MappedHost => {
                            let mut gpu = 0;
                            driver.check(
                                "cuMemHostGetDevicePointer_v2",
                                (driver.host_device)(&mut gpu, address as Handle, 0),
                            )?;
                            if gpu != address {
                                return Err(invalid(
                                    "mapped host address differs from recorded CUDA address",
                                ));
                            }
                        }
                    }
                }
                if let Some(file) = &allocation.file {
                    let mut file = File::open(root.join(file)).map_err(|e| e.to_string())?;
                    if file.metadata().map_err(|e| e.to_string())?.len() != allocation.nbytes as u64
                    {
                        return Err(invalid(format!(
                            "storage size mismatch for {}",
                            allocation.name
                        )));
                    }
                    let mut buffer = vec![0u8; allocation.nbytes.min(8 << 20)];
                    let mut offset = 0;
                    while offset < allocation.nbytes {
                        let length = buffer.len().min(allocation.nbytes - offset);
                        file.read_exact(&mut buffer[..length])
                            .map_err(|e| e.to_string())?;
                        match allocation.memory {
                            Memory::Device => driver.check(
                                "cuMemcpyHtoD_v2",
                                (driver.upload)(
                                    address + offset as u64,
                                    buffer.as_ptr().cast(),
                                    length,
                                ),
                            )?,
                            Memory::MappedHost => std::ptr::copy_nonoverlapping(
                                buffer.as_ptr(),
                                (address as usize + offset) as *mut u8,
                                length,
                            ),
                        }
                        offset += length;
                        loaded += length as u64;
                        phase.advance(length as u64);
                        phase.detail(format!(
                            "{:.1} / {:.1} GiB",
                            loaded as f64 / (1u64 << 30) as f64,
                            bytes as f64 / (1u64 << 30) as f64
                        ));
                    }
                } else if matches!(allocation.memory, Memory::MappedHost) {
                    std::ptr::write_bytes(address as *mut u8, 0, allocation.nbytes);
                } else {
                    driver.check(
                        "cuMemsetD8_v2",
                        (driver.zero)(address, 0, allocation.nbytes),
                    )?;
                }
                if allocation.file.is_none() {
                    loaded += allocation.nbytes as u64;
                    phase.advance(allocation.nbytes as u64);
                    phase.detail(format!(
                        "{:.1} / {:.1} GiB",
                        loaded as f64 / (1u64 << 30) as f64,
                        bytes as f64 / (1u64 << 30) as f64
                    ));
                }
            }
            for allocation in &package.allocations {
                for relocation in &allocation.relocations {
                    let address =
                        self.allocations[&allocation.name].address + relocation.offset as u64;
                    let mut bytes = vec![0u8; relocation.width()];
                    driver.check(
                        "cuMemcpyDtoH_v2",
                        (driver.download)(bytes.as_mut_ptr().cast(), address, bytes.len()),
                    )?;
                    self.relocate(&mut bytes, relocation, 0)?;
                    driver.check(
                        "cuMemcpyHtoD_v2",
                        (driver.upload)(address, bytes.as_ptr().cast(), bytes.len()),
                    )?;
                }
            }
            phase.finish(format!(
                "{:.1} GiB GPU · {:.1} GiB host",
                footprint as f64 / (1u64 << 30) as f64,
                (bytes - footprint as u64) as f64 / (1u64 << 30) as f64
            ));
            let phase = progress.phase(
                "Load kernels",
                Some((package.modules.len() + package.functions.len()) as u64),
            );
            let mut modules = HashMap::new();
            for module in &package.modules {
                let image = fs::read(root.join(&module.file)).map_err(|e| e.to_string())?;
                if !image.starts_with(b"\x7fELF") {
                    return Err(invalid("module image is not a compiled cubin"));
                }
                let mut handle = null_mut();
                driver.check(
                    "cuModuleLoadData",
                    (driver.module_load)(&mut handle, image.as_ptr().cast()),
                )?;
                self.modules.push(handle);
                modules.insert(&module.name, handle);
                for global in &module.globals {
                    let symbol = CString::new(global.symbol.as_str()).unwrap();
                    let (mut address, mut size) = (0, 0);
                    driver.check(
                        "cuModuleGetGlobal_v2",
                        (driver.module_global)(&mut address, &mut size, handle, symbol.as_ptr()),
                    )?;
                    let mut contents =
                        fs::read(root.join(&global.file)).map_err(|e| e.to_string())?;
                    if size != global.nbytes || contents.len() != size {
                        return Err(invalid(format!(
                            "module global size mismatch for {}",
                            global.symbol
                        )));
                    }
                    for relocation in &global.relocations {
                        self.relocate(&mut contents, relocation, relocation.offset)?;
                    }
                    driver.check(
                        "cuMemcpyHtoD_v2",
                        (driver.upload)(address, contents.as_ptr().cast(), size),
                    )?;
                }
                phase.advance(1);
            }
            let mut functions = Vec::new();
            for function in &package.functions {
                let symbol = CString::new(function.symbol.as_str()).unwrap();
                let mut handle = null_mut();
                driver.check(
                    "cuModuleGetFunction",
                    (driver.function)(&mut handle, modules[&function.module], symbol.as_ptr()),
                )?;
                for (index, expected) in function.parameter_sizes.iter().enumerate() {
                    let (mut offset, mut size) = (0, 0);
                    driver.check(
                        "cuFuncGetParamInfo",
                        (driver.parameter_info)(handle, index, &mut offset, &mut size),
                    )?;
                    if size != *expected {
                        return Err(invalid(
                            "loaded kernel parameter layout differs from package",
                        ));
                    }
                }
                let (mut offset, mut size) = (0, 0);
                let status = (driver.parameter_info)(
                    handle,
                    function.parameter_sizes.len(),
                    &mut offset,
                    &mut size,
                );
                if status != 1 {
                    return Err(invalid(
                        "loaded kernel parameter count differs from package",
                    ));
                }
                for [attribute, value] in &function.attributes {
                    driver.check(
                        "cuFuncSetAttribute",
                        (driver.function_attribute)(handle, *attribute, *value),
                    )?;
                }
                functions.push(handle);
                phase.advance(1);
            }
            phase.finish(format!("{} kernels", functions.len()));
            let nodes = package
                .graphs
                .iter()
                .map(|graph| graph.nodes.len() as u64 + 1)
                .sum();
            let phase = progress.phase("Prepare execution", Some(nodes));
            for graph in &package.graphs {
                let mut handle = null_mut();
                driver.check("cuGraphCreate", (driver.graph_create)(&mut handle, 0))?;
                self.templates.push(handle);
                let mut nodes = Vec::new();
                for node in &graph.nodes {
                    let mut result = null_mut();
                    match node {
                        Node::Kernel {
                            function,
                            grid,
                            block,
                            shared,
                            arguments,
                            attributes,
                        } => {
                            let mut storage = arguments
                                .iter()
                                .map(|a| self.bind(a))
                                .collect::<Result<Vec<_>>>()?;
                            let mut pointers: Vec<_> =
                                storage.iter_mut().map(|a| a.as_mut_ptr().cast()).collect();
                            let params = KernelParams {
                                function: functions[*function],
                                grid: *grid,
                                block: *block,
                                shared: *shared,
                                arguments: pointers.as_mut_ptr(),
                                extra: null_mut(),
                                kernel: null_mut(),
                                context: null_mut(),
                            };
                            driver.check(
                                "cuGraphAddKernelNode_v2",
                                (driver.graph_kernel)(&mut result, handle, null(), 0, &params),
                            )?;
                            for attribute in attributes {
                                let bytes = self.bind(&attribute.value)?;
                                let mut value = [0u64; 8];
                                std::ptr::copy_nonoverlapping(
                                    bytes.as_ptr(),
                                    value.as_mut_ptr().cast(),
                                    64,
                                );
                                driver.check(
                                    "cuGraphKernelNodeSetAttribute",
                                    (driver.graph_attribute)(
                                        result,
                                        attribute.id,
                                        value.as_ptr().cast(),
                                    ),
                                )?;
                            }
                        }
                        Node::Memset {
                            destination,
                            pitch,
                            value,
                            element_size,
                            width,
                            height,
                        } => {
                            let params = MemsetParams {
                                destination: self.address(destination),
                                pitch: *pitch,
                                value: *value,
                                element_size: *element_size,
                                width: *width,
                                height: *height,
                            };
                            driver.check(
                                "cuGraphAddMemsetNode",
                                (driver.graph_memset)(
                                    &mut result,
                                    handle,
                                    null(),
                                    0,
                                    &params,
                                    self.context,
                                ),
                            )?;
                        }
                        Node::Empty => {
                            driver.check(
                                "cuGraphAddEmptyNode",
                                (driver.graph_empty)(&mut result, handle, null(), 0),
                            )?;
                        }
                        Node::Copy {
                            source,
                            destination,
                            nbytes,
                        } => {
                            let endpoint = |reference| CopyEndpoint {
                                memory_type: 2,
                                device: self.address(reference),
                                pitch: *nbytes,
                                height: 1,
                                ..CopyEndpoint::default()
                            };
                            let params = CopyParams {
                                source: endpoint(source),
                                destination: endpoint(destination),
                                width: *nbytes,
                                height: 1,
                                depth: 1,
                            };
                            driver.check(
                                "cuGraphAddMemcpyNode",
                                (driver.graph_copy)(
                                    &mut result,
                                    handle,
                                    null(),
                                    0,
                                    &params,
                                    self.context,
                                ),
                            )?;
                        }
                    }
                    nodes.push(result);
                    phase.advance(1);
                }
                if !graph.edges.is_empty() {
                    let sources: Vec<_> = graph.edges.iter().map(|e| nodes[e.source]).collect();
                    let destinations: Vec<_> =
                        graph.edges.iter().map(|e| nodes[e.destination]).collect();
                    let data: Vec<_> = graph.edges.iter().map(|e| e.data).collect();
                    driver.check(
                        "cuGraphAddDependencies_v2",
                        (driver.graph_edges)(
                            handle,
                            sources.as_ptr(),
                            destinations.as_ptr(),
                            data.as_ptr(),
                            data.len(),
                        ),
                    )?;
                }
                let mut executable = null_mut();
                driver.check(
                    "cuGraphInstantiateWithFlags",
                    (driver.graph_instantiate)(&mut executable, handle, graph.flags),
                )?;
                self.executables.push(executable);
                self.names.push(graph.name.clone());
                phase.advance(1);
            }
            driver.check("cuCtxSynchronize", (driver.context_sync)())?;
            phase.finish(
                package
                    .execution
                    .as_ref()
                    .and_then(|execution| execution.get("max_active"))
                    .and_then(serde_json::Value::as_u64)
                    .map(|capacity| format!("concurrency {capacity}"))
                    .unwrap_or_default(),
            );
        }
        Ok(())
    }

    pub fn graph_names(&self) -> &[String] {
        &self.names
    }

    pub fn graph_bundle(&self) -> GraphBundle {
        GraphBundle {
            context: self.context as usize,
            device: self.device,
            graph_execs: self.executables.iter().map(|v| *v as usize).collect(),
            buffers: self
                .allocations
                .values()
                .map(|a| BufferSpan {
                    address: a.address as usize,
                    nbytes: a.nbytes,
                })
                .collect(),
        }
    }

    pub fn allocation(&self, name: &str) -> Result<BufferSpan> {
        let a = self
            .allocations
            .get(name)
            .ok_or_else(|| invalid(format!("unknown allocation {name:?}")))?;
        Ok(BufferSpan {
            address: a.address as usize,
            nbytes: a.nbytes,
        })
    }

    /// Copy an allocation after the caller has drained every replay that uses it.
    pub fn download(&self, name: &str) -> Result<Vec<u8>> {
        let a = self.allocation(name)?;
        let mut bytes = vec![0u8; a.nbytes];
        // SAFETY: The allocation remains owned by self and the output is sized to it.
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            self.driver
                .check("cuCtxSynchronize", (self.driver.context_sync)())?;
            self.driver.check(
                "cuMemcpyDtoH_v2",
                (self.driver.download)(bytes.as_mut_ptr().cast(), a.address as u64, bytes.len()),
            )?;
        }
        Ok(bytes)
    }

    /// Replace allocation contents after draining replay on the owned context.
    pub fn upload(&self, name: &str, bytes: &[u8]) -> Result<()> {
        let a = self.allocation(name)?;
        if bytes.len() != a.nbytes {
            return Err(invalid("upload size differs from allocation"));
        }
        // SAFETY: The owned allocation and input span have equal validated sizes.
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            self.driver
                .check("cuCtxSynchronize", (self.driver.context_sync)())?;
            self.driver.check(
                "cuMemcpyHtoD_v2",
                (self.driver.upload)(a.address as u64, bytes.as_ptr().cast(), bytes.len()),
            )?;
        }
        Ok(())
    }
}

impl Drop for NativeResources {
    fn drop(&mut self) {
        // SAFETY: This owner is the sole resource destroyer. On a failed drain,
        // retain the driver and all GPU resources until process termination.
        unsafe {
            let mut previous = null_mut();
            (self.driver.context_get)(&mut previous);
            if (self.driver.context_set)(self.context) != 0 || (self.driver.context_sync)() != 0 {
                std::mem::forget(self.driver.clone());
                return;
            }
            for executable in self.executables.iter().rev() {
                (self.driver.executable_destroy)(*executable);
            }
            for graph in self.templates.iter().rev() {
                (self.driver.graph_destroy)(*graph);
            }
            for module in self.modules.iter().rev() {
                (self.driver.module_unload)(*module);
            }
            for allocation in self.allocations.values() {
                match (allocation.memory, allocation.fixed) {
                    (Memory::Device, false) => {
                        (self.driver.free)(allocation.base);
                    }
                    (Memory::Device, true) => {
                        (self.driver.unmap)(allocation.address, allocation.nbytes);
                    }
                    (Memory::MappedHost, _) => {
                        (self.driver.host_unregister)(allocation.address as Handle);
                        libc::munmap(allocation.address as Handle, allocation.nbytes);
                    }
                }
            }
            for (address, size) in self.reservations.iter().rev() {
                (self.driver.release_address)(*address, *size);
            }
            (self.driver.context_destroy)(self.context);
            if previous != self.context {
                (self.driver.context_set)(previous);
            }
        }
    }
}
