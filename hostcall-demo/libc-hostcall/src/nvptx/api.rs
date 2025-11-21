extern crate alloc;

use alloc::borrow::ToOwned;
use alloc::ffi::CString;
use core::arch::nvptx::*;
use core::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void};
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::hostcall::{
    acquire_return_slot, free_return_slot, pool_return_slot, sleep_ns_kernel, submit_hostcall,
};

// A macro you can define elsewhere, e.g.:
macro_rules! unsupported {
    ($f:ident) => {
        unsafe {
            let mut fmt = [0_u32];
            vprintf(
                concat!(
                    "unsupported external function ",
                    stringify!($f),
                    " called\n\0"
                )
                .as_ptr()
                .cast(),
                &raw mut fmt as *mut _,
            );
            trap();
        }
    };
}

#[unsafe(no_mangle)]
pub extern "C" fn syscall(n: c_long, arg1: c_long) -> c_long {
    unsupported!(syscall)
}

#[unsafe(no_mangle)]
pub extern "C" fn mmap64(
    addr: *mut c_void,
    length: usize,
    prot: c_int,
    flags: c_int,
    fd: c_int,
    offset: c_long,
) -> *mut c_void {
    unsupported!(mmap64)
}

#[unsafe(no_mangle)]
pub extern "C" fn clock_gettime(clock_id: c_int, tp: *mut c_void) -> c_int {
    unsupported!(clock_gettime)
}

#[unsafe(no_mangle)]
pub extern "C" fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> c_int {
    unsupported!(posix_memalign)
}

#[unsafe(no_mangle)]
pub extern "C" fn munmap(addr: *mut c_void, length: usize) -> c_int {
    unsupported!(munmap)
}

#[unsafe(no_mangle)]
pub extern "C" fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize {
    unsupported!(read)
}

#[unsafe(no_mangle)]
pub extern "C" fn fcntl(fd: c_int, cmd: c_int, arg: c_long) -> c_long {
    unsupported!(fcntl)
}

#[unsafe(no_mangle)]
pub extern "C" fn writev(fd: c_int, iov: *const c_void, iovcnt: c_int) -> isize {
    unsupported!(writev)
}
#[repr(C)]
#[derive(Debug)]
struct FileWrite{
    ptr:u64,
    len:u32,
    fd:i32,
}
#[unsafe(no_mangle)]
pub extern "C" fn write(fd: c_int, buf: *const c_void, count: usize) -> isize {
    let fw = FileWrite{fd,ptr:buf as usize as u64, len:count as u32};
    let return_slot = acquire_return_slot();
    let mut data = fw;
    submit_hostcall(
        __HOSTCALL__write.load(Ordering::Relaxed) as u32,
        return_slot,
        &raw mut data as *const (),
        core::mem::size_of::<FileWrite>() as u16,
    )
    .unwrap();
    while pool_return_slot(return_slot) == 0 {
        sleep_ns_kernel(1000_0000);
        let mut fmt = [0];
        unsafe {
            vprintf(
                c"Waiting for `write` hostcall to finish\n".as_ptr() as *const _,
                &raw mut fmt as *const _,
            )
        };
    }
    let val = pool_return_slot(return_slot);
    free_return_slot(return_slot);
   {
    let mut fmt = [0];
        unsafe {
            vprintf(
                c"`write` done!\n".as_ptr() as *const _,
                &raw mut fmt as *const _,
            )
        };
   }
    val as isize
}

#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_GetIP(ctx: *mut c_void) -> *mut c_void {
    unsupported!(_Unwind_GetIP)
}

#[unsafe(no_mangle)]
pub extern "C" fn abort() -> ! {
    unsupported!(abort)
}
#[repr(C)]
#[derive(Debug)]
struct FileOpen{
    ptr:u64,
    len:u64,
    flags:u32,
    mode:u32,
}
#[unsafe(no_mangle)]
pub extern "C" fn open64(path: *const c_char, flags: c_int, mode: c_ulong) -> c_int {
    let cstr = unsafe { core::ffi::CStr::from_ptr(path).to_owned() };
    
let fw = FileOpen{ptr:cstr.as_ptr() as usize as u64, len:cstr.count_bytes() as u64, flags:flags as u32,mode: mode as u32};
    let return_slot = acquire_return_slot();
    let mut data = fw;
    submit_hostcall(
        __HOSTCALL__open.load(Ordering::Relaxed) as u32,
        return_slot,
        &raw mut data as *const (),
        core::mem::size_of::<FileOpen>() as u16,
    )
    .unwrap();
    while pool_return_slot(return_slot) == 0 {
        sleep_ns_kernel(1000_0000);
        let mut fmt = [0];
        unsafe {
            vprintf(
                c"Waiting for `open64` hostcall to finish\n".as_ptr() as *const _,
                &raw mut fmt as *const _,
            )
        };
    }
    let val = pool_return_slot(return_slot);
    free_return_slot(return_slot);
   {
    let mut fmt = [0];
        unsafe {
            vprintf(
                c"`open64` done!\n".as_ptr() as *const _,
                &raw mut fmt as *const _,
            )
        };
   }
    val as i32

}

#[unsafe(no_mangle)]
pub extern "C" fn __errno_location() -> *mut c_int {
    unsupported!(__errno_location)
}

#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_Backtrace(trace_fn: *mut c_void, trace_arg: *mut c_void) -> c_int {
    unsupported!(_Unwind_Backtrace)
}

#[unsafe(no_mangle)]
pub extern "C" fn dl_iterate_phdr(callback: *mut c_void, data: *mut c_void) -> c_int {
    unsupported!(dl_iterate_phdr)
}

#[unsafe(no_mangle)]
pub extern "C" fn close(fd: c_int) -> c_int {
    unsupported!(close)
}

#[unsafe(no_mangle)]
pub extern "C" fn getenv(name: *const c_char) -> *mut c_char {
    unsupported!(getenv)
}

#[unsafe(no_mangle)]
pub extern "C" fn __xpg_strerror_r(errnum: c_int, buf: *mut c_char, buflen: usize) -> c_int {
    unsupported!(__xpg_strerror_r)
}

#[unsafe(no_mangle)]
pub extern "C" fn getcwd(buf: *mut c_char, size: usize) -> *mut c_char {
    unsupported!(getcwd)
}

#[unsafe(no_mangle)]
pub extern "C" fn pthread_getspecific(key: c_int) -> *mut c_void {
    core::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn pthread_setspecific(key: c_int, value: *const c_void) -> c_int {
    unsupported!(pthread_setspecific)
}

#[unsafe(no_mangle)]
pub extern "C" fn pthread_key_create(key: *mut c_int, destructor: *mut c_void) -> c_int {
    unsafe{*key = 0xDEAD_BEEF_u32 as i32};
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn pthread_key_delete(key: c_int) -> c_int {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn calloc(nmemb: usize, size: usize) -> *mut c_void {
    unsupported!(calloc)
}

#[unsafe(no_mangle)]
pub extern "C" fn malloc(size: usize) -> *mut c_void {
    let return_slot = acquire_return_slot();
    let mut data = [size];
    submit_hostcall(
        __HOSTCALL__device_malloc.load(Ordering::Relaxed) as u32,
        return_slot,
        &raw mut data as *const (),
        8,
    )
    .unwrap();
    while pool_return_slot(return_slot) == 0 {
        sleep_ns_kernel(1000_0000);
        let mut fmt = [0];
        unsafe {
            vprintf(
                c"Waiting for `malloc` hostcall to finish\n".as_ptr() as *const _,
                &raw mut fmt as *const _,
            )
        };
    }
    let val = pool_return_slot(return_slot);
    free_return_slot(return_slot);
    {
        let mut fmt = [0];
        unsafe {
            vprintf(
                c"`malloc` done!\n".as_ptr() as *const _,
                &raw mut fmt as *const _,
            )
        };
    }
    val as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "C" fn free(ptr: *mut c_void) {
    let return_slot = acquire_return_slot();
    let mut data = [ptr as u64];
    submit_hostcall(
        __HOSTCALL__device_free.load(Ordering::Relaxed) as u32,
        return_slot,
        &raw mut data as *const (),
        8,
    )
    .unwrap();
    while pool_return_slot(return_slot) == 0 {
        sleep_ns_kernel(1000_0000);
        let mut fmt = [0];
        unsafe {
            vprintf(
                c"Waiting for `free` hostcall to finish\n".as_ptr() as *const _,
                &raw mut fmt as *const _,
            )
        };
    }
    free_return_slot(return_slot);
}

#[unsafe(no_mangle)]
pub extern "C" fn realloc(ptr: *mut c_void, size: usize) -> *mut c_void {
    unsupported!(realloc)
}

#[unsafe(no_mangle)]
pub extern "C" fn realpath(path: *const c_char, resolved_path: *mut c_char) -> *mut c_char {
    unsupported!(realpath)
}

#[unsafe(no_mangle)]
pub extern "C" fn lseek64(fd: c_int, offset: c_long, whence: c_int) -> c_long {
    unsupported!(lseek64)
}

#[unsafe(no_mangle)]
pub extern "C" fn fstat64(fd: c_int, buf: *mut c_void) -> c_int {
    unsupported!(fstat64)
}

#[unsafe(no_mangle)]
pub extern "C" fn stat64(path: *const c_char, buf: *mut c_void) -> c_int {
    unsupported!(stat64)
}

#[unsafe(no_mangle)]
pub extern "C" fn readlink(path: *const c_char, buf: *mut c_char, bufsiz: usize) -> isize {
    unsupported!(readlink)
}

#[unsafe(no_mangle)]
pub extern "C" fn statx(
    dirfd: c_int,
    pathname: *const c_char,
    flags: c_int,
    mask: c_uint,
    statxbuf: *mut (),
) -> c_int {
    unsupported!(statx)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gettid() -> u32 {
    unsupported!(gettid)
}
#[unsafe(no_mangle)]
#[used]
pub static __HOSTCALL__device_malloc: AtomicU64 = AtomicU64::new(u64::MAX);
#[unsafe(no_mangle)]
#[used]
pub static __HOSTCALL__device_free: AtomicU64 = AtomicU64::new(u64::MAX);
#[unsafe(no_mangle)]
#[used]
pub static __HOSTCALL__write: AtomicU64 = AtomicU64::new(u64::MAX);
#[unsafe(no_mangle)]
#[used]
pub static __HOSTCALL__open: AtomicU64 = AtomicU64::new(u64::MAX);
#[unsafe(no_mangle)]
#[used]
pub static __HOSTCALL__exit: AtomicU64 = AtomicU64::new(u64::MAX);

/// Request the host to exit; blocks until acknowledged.
pub fn host_exit(code: i32) {
    let return_slot = acquire_return_slot();
    let mut data = [code as u64];
    submit_hostcall(
        __HOSTCALL__exit.load(Ordering::Relaxed) as u32,
        return_slot,
        &raw mut data as *const (),
        8,
    )
    .unwrap();
    while pool_return_slot(return_slot) == 0 {
        sleep_ns_kernel(1000_0000);
    }
    free_return_slot(return_slot);
}

#[unsafe(no_mangle)]
pub extern "C" fn _exit(status: c_int) -> ! {
    host_exit(status);
    unsafe { core::arch::nvptx::trap() }
}

#[unsafe(no_mangle)]
pub extern "C" fn exit(status: c_int) -> ! {
    _exit(status)
}
