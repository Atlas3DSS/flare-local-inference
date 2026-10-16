//! Hardware revision selection without creating a CUDA context.

use std::ffi::{c_int, c_uint};

use libloading::Library;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn platform(pci_id: u32) -> Option<&'static str> {
    if pci_id & 0xffff != 0x10de {
        return None;
    }
    match pci_id >> 16 {
        0x2e12 => Some("gb10"),
        0x2b85 | 0x2b87 | 0x2b8c | 0x2bb1 | 0x2bb2 | 0x2bb3 | 0x2bb4 | 0x2bb5 | 0x2bb9 | 0x2bbc => {
            Some("gb202")
        }
        _ => None,
    }
}

pub fn detect(count: usize) -> Result<&'static str> {
    // SAFETY: Only the system driver is loaded, and the library remains live through all calls.
    #[cfg(unix)]
    let library = unsafe { Library::new("libcuda.so.1") };
    #[cfg(windows)]
    let library = unsafe {
        libloading::os::windows::Library::load_with_flags(
            "nvcuda.dll",
            windows_sys::Win32::System::LibraryLoader::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
        .map(Library::from)
    };
    let library = library.map_err(|error| {
        format!("cannot detect hardware without the NVIDIA driver: {error}; specify --revision")
    })?;
    // SAFETY: These signatures match the CUDA driver ABI; output pointers are valid.
    unsafe {
        let init = library.get::<unsafe extern "C" fn(c_uint) -> c_int>(b"cuInit\0")?;
        let get =
            library.get::<unsafe extern "C" fn(*mut c_int, c_int) -> c_int>(b"cuDeviceGet\0")?;
        let attribute = library.get::<unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int>(
            b"cuDeviceGetAttribute\0",
        )?;
        check("cuInit", init(0))?;
        let mut selected = None;
        for ordinal in 0..count {
            let mut device = 0;
            check("cuDeviceGet", get(&mut device, ordinal as c_int))?;
            let mut pci_id = 0;
            // CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID contains the device and vendor IDs.
            check(
                "cuDeviceGetAttribute(GPU_PCI_DEVICE_ID)",
                attribute(&mut pci_id, 139, device),
            )?;
            let detected = platform(pci_id as u32).ok_or_else(|| {
                format!(
                    "GPU {ordinal} has unsupported PCI ID {:#010x}; specify --revision",
                    pci_id as u32
                )
            })?;
            if selected.is_some_and(|value| value != detected) {
                return Err(
                    "selected GPUs span gb10 and gb202; select one platform or specify --revision"
                        .into(),
                );
            }
            selected = Some(detected);
        }
        selected.ok_or_else(|| "select at least one GPU or specify --revision".into())
    }
}

fn check(operation: &str, code: c_int) -> Result<()> {
    if code != 0 {
        return Err(format!(
            "hardware detection: {operation} failed with CUDA error {code}; specify --revision"
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_ids_select_the_chip_without_matching_marketing_names() {
        assert_eq!(platform(0x2e1210de), Some("gb10"));
        assert_eq!(platform(0x2bb410de), Some("gb202"));
        assert_eq!(platform(0x2bb110de), Some("gb202"));
        assert_eq!(platform(0x2bb510de), Some("gb202"));
        assert_eq!(platform(0x2b8510de), Some("gb202"));
        assert_eq!(platform(0x26b110de), None);
        assert_eq!(platform(0x2bb41234), None);
    }
}
