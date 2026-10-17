//! Native callbacks used by the preparation process's PyTorch memory pool.

use std::{
    collections::BTreeMap,
    ffi::{CString, c_char, c_void},
    ptr::null_mut,
    sync::Mutex,
};

use crate::runtime_package::driver::{Access, AllocationProperties, Driver, Location};
use crate::{Error, ExecutionError, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Manifest, message)
}

// PyTorch's CPU allocator is mimalloc, which places its 1 GiB arenas at random addresses
// in 2-6 TiB; a peer rank allocates CPU tensors before reserving these ranges.
#[cfg(target_os = "linux")]
const BASE: u64 = 0x80000000000;
#[cfg(target_os = "windows")]
const BASE: u64 = 0x40000000000;
const EXTENT: usize = 1 << 40;
// Keep small Linux exports within the Windows WDDM address range as well.
const HOST_BASE: u64 = 0xc0000000000;

fn host_base(device_base: u64, world: u32) -> u64 {
    if device_base + u64::from(world) * EXTENT as u64 <= HOST_BASE {
        HOST_BASE
    } else {
        0x200000000000
    }
}

struct Arena {
    driver: Driver,
    device: i32,
    granularity: usize,
    gpudirect: bool,
    host_base: u64,
    free: Vec<(u64, usize)>,
    mappings: BTreeMap<u64, usize>,
}
static ARENA: Mutex<Option<Arena>> = Mutex::new(None);
static ERROR: Mutex<Option<CString>> = Mutex::new(None);

pub(crate) struct MemoryInventory {
    pub device: i32,
    pub free: usize,
    pub total: usize,
    pub mappings: Vec<(u64, usize)>,
}

pub(crate) fn memory_inventory() -> Result<MemoryInventory> {
    let arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let a = arena
        .as_ref()
        .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
    let (_, device) = crate::current_context()?;
    if device != a.device {
        return Err(invalid(
            "memory inventory requires the package rank's CUDA context",
        ));
    }
    let (mut free, mut total) = (0, 0);
    // SAFETY: Preparation retains the current context and both outputs are valid.
    unsafe {
        a.driver
            .check("cuCtxSynchronize", (a.driver.context_sync)())?;
        a.driver.check(
            "cuMemGetInfo_v2",
            (a.driver.memory_info)(&mut free, &mut total),
        )?;
    }
    Ok(MemoryInventory {
        device,
        free,
        total,
        mappings: a
            .mappings
            .iter()
            .map(|(&address, &size)| (address, size))
            .collect(),
    })
}

fn properties(device: i32, gpudirect: bool) -> AllocationProperties {
    AllocationProperties {
        kind: 1,
        handles: i32::from(cfg!(target_os = "linux")),
        location: Location { kind: 1, device },
        windows: null_mut(),
        // cuFile registration requires gpuDirectRDMACapable on VMM allocations.
        flags: [0, u8::from(gpudirect), 0, 0, 0, 0, 0, 0],
    }
}

fn failure(error: impl ToString) {
    *ERROR.lock().unwrap_or_else(|e| e.into_inner()) =
        Some(CString::new(error.to_string().replace('\0', " ")).unwrap());
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_initialize(rank: u32, world: u32, device: i32) -> i32 {
    let initialize = || -> Result<()> {
        let mut arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        if arena.is_some() {
            return Err(invalid("native package allocator is already initialized"));
        }
        if world == 0 || rank >= world || device < 0 {
            return Err(invalid("invalid package rank or device"));
        }
        let host_base = if cfg!(target_os = "windows") {
            HOST_BASE
        } else {
            host_base(BASE, world)
        };
        if u64::from(world) * EXTENT as u64 > BASE.abs_diff(host_base) {
            return Err(invalid(
                "package host reservations would overlap device reservations",
            ));
        }
        let driver = Driver::load()?;
        let (mut address, mut granularity, mut gpudirect) = (0, 0, 0);
        unsafe {
            driver.check(
                "cuDeviceGetAttribute(GPU_DIRECT_RDMA_SUPPORTED)",
                (driver.device_attribute)(&mut gpudirect, 116, device),
            )?;
            driver.check(
                "cuMemGetAllocationGranularity",
                (driver.granularity)(&mut granularity, &properties(device, gpudirect != 0), 0),
            )?;
            let mut reserved = Vec::new();
            for peer in 0..world {
                let base = BASE + u64::from(peer) * EXTENT as u64;
                let status = (driver.reserve)(&mut address, EXTENT, 0, base, 0);
                if status != 0 || address != base {
                    if status == 0 {
                        (driver.release_address)(address, EXTENT);
                    }
                    for address in reserved {
                        (driver.release_address)(address, EXTENT);
                    }
                    driver.check("cuMemAddressReserve", status)?;
                    return Err(invalid(format!(
                        "native package device address range {base:#x}+{EXTENT:#x} is unavailable; CUDA returned {address:#x}"
                    )));
                }
                reserved.push(address);
            }
        }
        *arena = Some(Arena {
            driver,
            device,
            granularity,
            gpudirect: gpudirect != 0,
            host_base: host_base + u64::from(rank) * EXTENT as u64,
            free: vec![(BASE + u64::from(rank) * EXTENT as u64, EXTENT / 2)],
            mappings: BTreeMap::new(),
        });
        Ok(())
    };
    match initialize() {
        Ok(()) => 0,
        Err(error) => {
            failure(error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_allocate(
    size: usize,
    device: i32,
    _stream: *mut c_void,
) -> *mut c_void {
    let allocate = || -> Result<u64> {
        let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        let a = guard
            .as_mut()
            .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
        if device != a.device {
            return Err(invalid("allocation device differs from package rank"));
        }
        let size = size
            .checked_add(a.granularity - 1)
            .ok_or_else(|| invalid("allocation size overflow"))?
            / a.granularity
            * a.granularity;
        let index = a
            .free
            .iter()
            .position(|(_, n)| *n >= size)
            .ok_or_else(|| invalid("native package address range exhausted"))?;
        let (address, available) = a.free[index];
        let mut physical = 0;
        unsafe {
            a.driver.check(
                "cuMemCreate",
                (a.driver.physical_create)(
                    &mut physical,
                    size,
                    &properties(device, a.gpudirect),
                    0,
                ),
            )?;
            let mapped = (a.driver.map)(address, size, 0, physical, 0);
            (a.driver.physical_release)(physical);
            a.driver.check("cuMemMap", mapped)?;
            let access = Access {
                location: Location { kind: 1, device },
                flags: 3,
            };
            let status = (a.driver.access)(address, size, &access, 1);
            if status != 0 {
                (a.driver.unmap)(address, size);
            }
            a.driver.check("cuMemSetAccess", status)?;
        }
        if size == available {
            a.free.remove(index);
        } else {
            a.free[index] = (address + size as u64, available - size);
        }
        a.mappings.insert(address, size);
        Ok(address)
    };
    match allocate() {
        Ok(address) => address as *mut c_void,
        // PyTorch may release cached segments and retry a physical allocation.
        Err(Error::Execution(ExecutionError::Driver(error)))
            if error.operation == "cuMemCreate" && error.code == 2 =>
        {
            null_mut()
        }
        Err(error) => {
            failure(error);
            null_mut()
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_free(
    pointer: *mut c_void,
    _size: usize,
    _device: i32,
    _stream: *mut c_void,
) {
    let release = || -> Result<()> {
        let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        let a = guard
            .as_mut()
            .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
        let address = pointer as u64;
        let size = *a
            .mappings
            .get(&address)
            .ok_or_else(|| invalid("unknown native package allocation"))?;
        unsafe {
            a.driver
                .check("cuMemUnmap", (a.driver.unmap)(address, size))?;
        }
        a.mappings.remove(&address);
        a.free.push((address, size));
        a.free.sort_unstable();
        let mut merged: Vec<(u64, usize)> = Vec::new();
        for (address, size) in a.free.drain(..) {
            if let Some(previous) = merged.last_mut().filter(|p| p.0 + p.1 as u64 == address) {
                previous.1 += size;
            } else {
                merged.push((address, size));
            }
        }
        a.free = merged;
        Ok(())
    };
    if let Err(error) = release() {
        failure(error);
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_snapshot() -> *mut c_char {
    let arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let Some(a) = arena.as_ref() else {
        return null_mut();
    };
    let value = serde_json::json!({
        "granularity": a.granularity, "mappings": a.mappings, "host_base": a.host_base,
    });
    CString::new(value.to_string()).unwrap().into_raw()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn flare_package_string_free(pointer: *mut c_char) {
    if !pointer.is_null() {
        unsafe {
            drop(CString::from_raw(pointer));
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_error() -> *const c_char {
    ERROR
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(std::ptr::null(), |s| s.as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_linux_exports_keep_every_host_range_within_windows_addresses() {
        let device_base = 8 << 40;
        for world in 1..=4 {
            let host = host_base(device_base, world);
            assert_eq!(host, 12 << 40);
            assert!(device_base + u64::from(world) * EXTENT as u64 <= host);
            assert!(host + u64::from(world) * EXTENT as u64 <= 16 << 40);
        }
    }

    #[test]
    fn larger_linux_exports_keep_host_ranges_clear_of_all_device_ranks() {
        let device_base = 8 << 40;
        for world in 5..=24 {
            let host = host_base(device_base, world);
            assert_eq!(host, 32 << 40);
            assert!(device_base + u64::from(world) * EXTENT as u64 <= host);
        }
    }
}
