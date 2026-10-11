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

const BASE: u64 = 0x200000000000;
const EXTENT: usize = 1 << 40;
struct Arena {
    driver: Driver,
    granularity: usize,
    free: Vec<(u64, usize)>,
    mappings: BTreeMap<u64, usize>,
}
static ARENA: Mutex<Option<Arena>> = Mutex::new(None);
static ERROR: Mutex<Option<CString>> = Mutex::new(None);

fn properties() -> AllocationProperties {
    AllocationProperties {
        kind: 1,
        handles: 0,
        location: Location { kind: 1, device: 0 },
        windows: null_mut(),
        flags: [0; 8],
    }
}

fn failure(error: impl ToString) {
    *ERROR.lock().unwrap_or_else(|e| e.into_inner()) =
        Some(CString::new(error.to_string().replace('\0', " ")).unwrap());
}

#[unsafe(no_mangle)]
extern "C" fn flare_package_initialize() -> i32 {
    let initialize = || -> Result<()> {
        let mut arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        if arena.is_some() {
            return Err(invalid("native package allocator is already initialized"));
        }
        let driver = Driver::load()?;
        let (mut address, mut granularity) = (0, 0);
        unsafe {
            driver.check(
                "cuMemGetAllocationGranularity",
                (driver.granularity)(&mut granularity, &properties(), 0),
            )?;
            driver.check(
                "cuMemAddressReserve",
                (driver.reserve)(&mut address, EXTENT, 0, BASE, 0),
            )?;
            if address != BASE {
                (driver.release_address)(address, EXTENT);
                return Err(invalid("native package device address range is occupied"));
            }
        }
        *arena = Some(Arena {
            driver,
            granularity,
            free: vec![(BASE, EXTENT)],
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
        if device != 0 {
            return Err(invalid("native package allocator requires TP1 device zero"));
        }
        let mut guard = ARENA.lock().unwrap_or_else(|e| e.into_inner());
        let a = guard
            .as_mut()
            .ok_or_else(|| invalid("native package allocator is uninitialized"))?;
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
                (a.driver.physical_create)(&mut physical, size, &properties(), 0),
            )?;
            let mapped = (a.driver.map)(address, size, 0, physical, 0);
            (a.driver.physical_release)(physical);
            a.driver.check("cuMemMap", mapped)?;
            let access = Access {
                location: Location { kind: 1, device: 0 },
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
    let value = serde_json::json!({"granularity": a.granularity, "mappings": a.mappings});
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
