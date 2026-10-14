use std::{
    fs::{File, OpenOptions},
    io,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::Path,
};

pub(crate) fn cuda_library() -> Result<libloading::Library, libloading::Error> {
    unsafe { libloading::Library::new("libcuda.so.1") }
}

pub(crate) unsafe fn read_at_raw(
    file: &File,
    address: *mut u8,
    bytes: usize,
    offset: u64,
) -> io::Result<usize> {
    let offset = libc::off_t::try_from(offset).map_err(|_| io::ErrorKind::InvalidInput)?;
    loop {
        let count = unsafe { libc::pread(file.as_raw_fd(), address.cast(), bytes, offset) };
        if count >= 0 {
            return Ok(count as usize);
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

pub(crate) fn direct_file(path: &Path, write: bool) -> io::Result<File> {
    OpenOptions::new()
        .read(!write)
        .write(write)
        .create(write)
        .truncate(write)
        .mode(0o600)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

pub(crate) fn direct_reopen(file: &File) -> io::Result<File> {
    direct_file(
        Path::new(&format!("/proc/self/fd/{}", file.as_raw_fd())),
        false,
    )
}

pub(crate) fn release_page_cache(file: &File) {
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

pub(crate) fn page_size() -> usize {
    usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(4096)
}

pub(crate) fn lower_priority(nice: i32) {
    unsafe {
        let param = std::mem::zeroed();
        libc::sched_setscheduler(0, libc::SCHED_BATCH, &param);
        libc::setpriority(libc::PRIO_PROCESS, 0, nice);
    }
}

pub(crate) fn terminal_size() -> (usize, usize) {
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    unsafe {
        libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size);
    }
    (
        if size.ws_col == 0 {
            80
        } else {
            size.ws_col as usize
        },
        if size.ws_row == 0 {
            24
        } else {
            size.ws_row as usize
        },
    )
}

pub(crate) fn map_host(bytes: usize) -> io::Result<usize> {
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    unsafe {
        libc::madvise(ptr, bytes, libc::MADV_HUGEPAGE);
    }
    Ok(ptr as usize)
}

pub(crate) unsafe fn unmap_host(address: usize, bytes: usize) {
    unsafe {
        libc::munmap(address as *mut _, bytes);
    }
}
