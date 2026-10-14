use std::{
    fs::{File, OpenOptions},
    io,
    os::windows::{
        ffi::OsStrExt,
        fs::OpenOptionsExt,
        io::{AsRawHandle, FromRawHandle},
    },
    path::Path,
};
use windows_sys::Win32::{
    Foundation::*,
    Storage::FileSystem::*,
    System::{Console::*, IO::*, Memory::*, SystemInformation::*, Threading::*},
};

pub(crate) fn wide(path: &Path) -> io::Result<Vec<u16>> {
    let mut text: Vec<_> = path.as_os_str().encode_wide().collect();
    if text.contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "path contains NUL",
        ));
    }
    text.push(0);
    Ok(text)
}

pub(crate) fn cuda_library() -> Result<libloading::Library, libloading::Error> {
    unsafe {
        libloading::os::windows::Library::load_with_flags(
            "nvcuda.dll",
            windows_sys::Win32::System::LibraryLoader::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
        .map(Into::into)
    }
}

pub(crate) unsafe fn read_at_raw(
    file: &File,
    address: *mut u8,
    bytes: usize,
    offset: u64,
) -> io::Result<usize> {
    unsafe {
        let event = CreateEventW(std::ptr::null(), 1, 0, std::ptr::null());
        if event.is_null() {
            return Err(io::Error::last_os_error());
        }
        let mut op: OVERLAPPED = std::mem::zeroed();
        op.Anonymous.Anonymous.Offset = offset as u32;
        op.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
        // Suppress IOCP delivery for this synchronous positional operation.
        op.hEvent = (event as usize | 1) as HANDLE;
        let mut count = 0;
        let ok = ReadFile(
            file.as_raw_handle(),
            address,
            bytes.min(u32::MAX as usize) as u32,
            &mut count,
            &mut op,
        );
        let mut result = if ok != 0 {
            Ok(count as usize)
        } else {
            let error = GetLastError();
            if error == ERROR_IO_PENDING {
                if GetOverlappedResult(file.as_raw_handle(), &op, &mut count, 1) != 0 {
                    Ok(count as usize)
                } else {
                    Err(io::Error::last_os_error())
                }
            } else if error == ERROR_HANDLE_EOF {
                Ok(0)
            } else {
                Err(io::Error::from_raw_os_error(error as i32))
            }
        };
        if result.as_ref().err().and_then(io::Error::raw_os_error) == Some(ERROR_HANDLE_EOF as i32)
        {
            result = Ok(0);
        }
        CloseHandle(event);
        result
    }
}

fn check_alignment(file: File) -> io::Result<File> {
    unsafe {
        let mut info: FILE_STORAGE_INFO = std::mem::zeroed();
        if GetFileInformationByHandleEx(
            file.as_raw_handle(),
            FileStorageInfo,
            (&mut info as *mut FILE_STORAGE_INFO).cast(),
            std::mem::size_of_val(&info) as u32,
        ) == 0
        {
            return Err(io::Error::last_os_error());
        }
        for sector in [
            info.LogicalBytesPerSector,
            info.PhysicalBytesPerSectorForPerformance,
        ] {
            if sector == 0 || 4096 % sector != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("direct I/O sector size {sector} does not divide 4096"),
                ));
            }
        }
    }
    Ok(file)
}

pub(crate) fn direct_file(path: &Path, write: bool) -> io::Result<File> {
    let file = OpenOptions::new()
        .read(!write)
        .write(write)
        .create(write)
        .truncate(write)
        .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)
        .open(path)?;
    check_alignment(file)
}

pub(crate) fn direct_reopen(file: &File) -> io::Result<File> {
    let handle = unsafe {
        ReOpenFile(
            file.as_raw_handle(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    check_alignment(unsafe { File::from_raw_handle(handle) })
}

pub(crate) fn release_page_cache(_: &File) {}

pub(crate) fn page_size() -> usize {
    let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
    unsafe {
        GetSystemInfo(&mut info);
    }
    info.dwPageSize as usize
}

pub(crate) fn lower_priority(_: i32) {
    unsafe {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

pub(crate) fn terminal_size() -> (usize, usize) {
    let mut info: CONSOLE_SCREEN_BUFFER_INFO = unsafe { std::mem::zeroed() };
    if unsafe { GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &mut info) } == 0 {
        return (80, 24);
    }
    (
        (info.srWindow.Right - info.srWindow.Left + 1) as usize,
        (info.srWindow.Bottom - info.srWindow.Top + 1) as usize,
    )
}

pub(crate) fn map_host(bytes: usize) -> io::Result<usize> {
    let ptr = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            bytes,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    if ptr.is_null() {
        return Err(io::Error::last_os_error());
    }
    // Populate the pages before the slab enters the transfer pool.
    for offset in (0..bytes).step_by(page_size()) {
        unsafe {
            ptr.cast::<u8>().add(offset).write_volatile(0);
        }
    }
    Ok(ptr as usize)
}

pub(crate) unsafe fn unmap_host(address: usize, _: usize) {
    unsafe {
        VirtualFree(address as *mut _, 0, MEM_RELEASE);
    }
}

pub(crate) fn free_space(path: &Path) -> io::Result<u64> {
    let text = wide(path)?;
    let mut available = 0;
    if unsafe {
        GetDiskFreeSpaceExW(
            text.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(available)
}

/// Owns reservation-granularity ranges containing page-granularity package allocations.
#[derive(Default)]
pub(crate) struct HostReservations(Vec<(usize, usize)>);
impl HostReservations {
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub(crate) fn reserve(
        &mut self,
        ranges: impl Iterator<Item = (usize, usize)>,
    ) -> io::Result<()> {
        let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
        unsafe {
            GetSystemInfo(&mut info);
        }
        let granularity = info.dwAllocationGranularity as usize;
        let mut merged: Vec<(usize, usize)> = Vec::new();
        let mut ranges = ranges
            .map(|(base, bytes)| {
                let end = base
                    .checked_add(bytes)
                    .and_then(|n| n.checked_add(granularity - 1))
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "host reservation exceeds address space",
                        )
                    })?;
                Ok((
                    base / granularity * granularity,
                    end / granularity * granularity,
                ))
            })
            .collect::<io::Result<Vec<_>>>()?;
        ranges.sort_unstable();
        for (start, end) in ranges {
            if let Some(last) = merged.last_mut().filter(|last| start <= last.1) {
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        }
        for (start, end) in merged {
            let ptr = unsafe {
                VirtualAlloc(start as *const _, end - start, MEM_RESERVE, PAGE_READWRITE)
            };
            if ptr.is_null() {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    format!(
                        "host address collision at {start:#x}: {}",
                        io::Error::last_os_error()
                    ),
                ));
            }
            self.0.push((ptr as usize, end - start));
        }
        Ok(())
    }
    pub(crate) fn commit(&self, address: usize, bytes: usize) -> io::Result<()> {
        if !self.0.iter().any(|&(base, size)| {
            address >= base
                && address
                    .checked_add(bytes)
                    .is_some_and(|end| end <= base + size)
        }) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "host allocation lies outside its reservation",
            ));
        }
        if unsafe { VirtualAlloc(address as *const _, bytes, MEM_COMMIT, PAGE_READWRITE) }
            != address as *mut _
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}
impl Drop for HostReservations {
    fn drop(&mut self) {
        for &(address, _) in &self.0 {
            unsafe {
                VirtualFree(address as *mut _, 0, MEM_RELEASE);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::os::{FileReadAt, completion::Queue};

    #[test]
    fn adjacent_host_pages_share_a_reservation_and_collisions_preserve_contents() {
        let base = 0x1000_0000_0000;
        let mut owned = HostReservations::default();
        owned
            .reserve([(base + 4096, 4096), (base + 8192, 4096)].into_iter())
            .unwrap();
        assert_eq!(owned.0.len(), 1);
        owned.commit(base + 4096, 8192).unwrap();
        unsafe {
            ((base + 4096) as *mut u64).write(0x12345678);
        }
        let mut collision = HostReservations::default();
        assert!(
            collision
                .reserve([(base + 4096, 4096)].into_iter())
                .is_err()
        );
        assert_eq!(unsafe { ((base + 4096) as *const u64).read() }, 0x12345678);
        assert!(owned.commit(base + 65536, 4096).is_err());
        drop(owned);
        collision
            .reserve([(base + 4096, 4096)].into_iter())
            .unwrap();
        collision.commit(base + 4096, 4096).unwrap();
        assert_eq!(unsafe { ((base + 4096) as *const u64).read() }, 0);
    }

    #[test]
    fn failed_reservation_releases_partial_ranges() {
        let base = 0x1100_0000_0000;
        let mut occupied = HostReservations::default();
        occupied
            .reserve([(base + 131072, 4096)].into_iter())
            .unwrap();
        {
            let mut partial = HostReservations::default();
            assert!(
                partial
                    .reserve([(base, 4096), (base + 131072, 4096)].into_iter())
                    .is_err()
            );
        }
        let mut reclaimed = HostReservations::default();
        reclaimed.reserve([(base, 4096)].into_iter()).unwrap();
    }

    #[test]
    fn direct_io_completes_and_cancellation_drains_before_reuse() {
        let path = std::env::temp_dir().join(format!("flare-直接-{}.bin", std::process::id()));
        let address = map_host(8192).unwrap();
        unsafe {
            std::ptr::write_bytes(address as *mut u8, 0x71, 8192);
        }
        let file = direct_file(&path, true).unwrap();
        let mut queue = Queue::new(2).unwrap();
        for slot in 0..2 {
            unsafe {
                queue
                    .submit(
                        slot,
                        &file,
                        address + slot * 4096,
                        4096,
                        slot as u64 * 4096,
                        true,
                    )
                    .unwrap();
            }
        }
        while queue.pending() > 0 {
            assert_eq!(queue.wait().unwrap().1.unwrap(), 4096);
        }
        drop(queue);
        file.sync_all().unwrap();
        drop(file);
        let source = File::open(&path).unwrap();
        let file = direct_reopen(&source).unwrap();
        let mut queue = Queue::new(2).unwrap();
        for slot in 0..2 {
            unsafe {
                queue
                    .submit(
                        slot,
                        &file,
                        address + slot * 4096,
                        4096,
                        slot as u64 * 4096,
                        false,
                    )
                    .unwrap();
            }
        }
        queue.cancel();
        while queue.pending() > 0 {
            match queue.wait().unwrap().1 {
                Ok(n) => assert_eq!(n, 4096),
                Err(error) => {
                    assert_eq!(error.raw_os_error(), Some(ERROR_OPERATION_ABORTED as i32))
                }
            }
        }
        unsafe {
            queue.submit(0, &file, address, 4096, 4096, false).unwrap();
        }
        assert_eq!(queue.wait().unwrap().1.unwrap(), 4096);
        assert!(
            unsafe { std::slice::from_raw_parts(address as *const u8, 4096) }
                .iter()
                .all(|byte| *byte == 0x71)
        );
        let mut metadata = [0; 17];
        source.read_exact_at(&mut metadata, 4093).unwrap();
        assert_eq!(metadata, [0x71; 17]);
        drop(queue);
        drop(file);
        drop(source);
        unsafe {
            unmap_host(address, 8192);
        }
        std::fs::remove_file(path).unwrap();
    }
}
