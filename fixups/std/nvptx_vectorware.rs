#![no_std]
#![feature(alloc_error_handler, rustc_attrs)]

// Minimal std facade for the custom nvptx target. We wire the allocator symbols
// into libc-hostcall so kernel code can call into host-managed allocation and
// logging primitives without pulling in the full std stack.

extern crate alloc;
use libc_hostcall::{free, malloc};
use core::panic::PanicInfo;
use core::{cmp, ptr};

#[inline]
unsafe fn hostcall_alloc(size: usize, align: usize) -> *mut u8 {
    // Rely on CUDA's allocator; ensure at least alignment bytes to satisfy common layouts.
    let ptr = unsafe { malloc(size.max(align)) } as *mut u8;
    if ptr.is_null() { core::ptr::null_mut() } else { ptr }
}

#[inline]
unsafe fn hostcall_dealloc(ptr: *mut u8, _size: usize, _align: usize) {
    unsafe { free(ptr.cast()) }
}

#[inline]
unsafe fn hostcall_realloc(
    ptr: *mut u8,
    old_size: usize,
    new_size: usize,
    align: usize,
) -> *mut u8 {
    let new_ptr = hostcall_alloc(new_size, align);
    if !new_ptr.is_null() && !ptr.is_null() {
        ptr::copy_nonoverlapping(ptr, new_ptr, cmp::min(old_size, new_size));
        hostcall_dealloc(ptr, old_size, align);
    }
    new_ptr
}

#[inline]
unsafe fn hostcall_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
    let ptr = hostcall_alloc(size, align);
    if !ptr.is_null() {
        ptr::write_bytes(ptr, 0, size);
    }
    ptr
}

#[inline]
fn hostcall_abort() -> ! {
    loop {
        core::hint::spin_loop();
    }
}

mod __rustc {
    use super::*;

    // Rust-mangled allocator symbols expected by core/alloc. Using
    // `rustc_std_internal_symbol` keeps the crate-disambiguated names in sync
    // with the rest of the std/alloc build so we don't have to hardcode hashes.
    #[rustc_std_internal_symbol]
    pub unsafe extern "Rust" fn __rust_alloc(size: usize, align: usize) -> *mut u8 {
        hostcall_alloc(size, align)
    }

    #[rustc_std_internal_symbol]
    pub unsafe extern "Rust" fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize) {
        hostcall_dealloc(ptr, size, align)
    }

    #[rustc_std_internal_symbol]
    pub unsafe extern "Rust" fn __rust_realloc(
        ptr: *mut u8,
        old_size: usize,
        new_size: usize,
        align: usize,
    ) -> *mut u8 {
        hostcall_realloc(ptr, old_size, new_size, align)
    }

    #[rustc_std_internal_symbol]
    pub unsafe extern "Rust" fn __rust_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
        hostcall_alloc_zeroed(size, align)
    }

    #[rustc_std_internal_symbol]
    pub unsafe extern "Rust" fn __rust_abort() -> ! {
        hostcall_abort()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    for i in 0..n {
        let lhs = *a.add(i);
        let rhs = *b.add(i);
        if lhs != rhs {
            return lhs as i32 - rhs as i32;
        }
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn strlen(s: *const u8) -> usize {
    let mut len = 0;
    while *s.add(len) != 0 {
        len += 1;
    }
    len
}

// Rust-mangled allocator symbols expected by core/alloc.
#[alloc_error_handler]
fn alloc_error(_layout: core::alloc::Layout) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

pub mod prelude {
    pub mod rust_2021 {
        pub use alloc::borrow::ToOwned;
        pub use alloc::string::{String, ToString};
        pub use alloc::vec::Vec;
        pub use core::prelude::rust_2021::*;
        pub use crate::io::Write;
    }
}

pub mod fmt {
    pub use core::fmt::*;
}

pub mod io {
    use core::fmt;

    #[derive(Debug, Copy, Clone)]
    pub struct Error;

    pub type Result<T> = core::result::Result<T, Error>;

    impl From<fmt::Error> for Error {
        fn from(_: fmt::Error) -> Self {
            Error
        }
    }

    pub trait Write {
        fn write(&mut self, buf: &[u8]) -> Result<usize>;

        fn flush(&mut self) -> Result<()> {
            Ok(())
        }

        fn write_all(&mut self, mut buf: &[u8]) -> Result<()> {
            while !buf.is_empty() {
                let n = self.write(buf)?;
                buf = &buf[n..];
            }
            Ok(())
        }
    }
}

pub mod fs {
    use alloc::ffi::CString;
    use alloc::vec::Vec;
    use core::ffi::c_char;

    use super::io::{Error, Result, Write};
    use libc_hostcall::{open64, write};

    const O_CREAT: i32 = 0o100;
    const O_TRUNC: i32 = 0o1000;
    const O_WRONLY: i32 = 0o1;

    pub struct File {
        fd: i32,
    }

    impl File {
        pub fn create(path: &str) -> Result<Self> {
            let cstr = CString::new(path).map_err(|_| Error)?;
            let fd = unsafe { open64(cstr.as_ptr() as *const c_char, O_CREAT | O_TRUNC | O_WRONLY, 0o644) };
            if fd < 0 {
                return Err(Error);
            }
            Ok(File { fd })
        }
    }

    impl Write for File {
        fn write(&mut self, buf: &[u8]) -> Result<usize> {
            let written = unsafe { write(self.fd, buf.as_ptr().cast(), buf.len()) };
            if written < 0 {
                Err(Error)
            } else {
                Ok(written as usize)
            }
        }
    }
}

pub mod borrow {
    pub use alloc::borrow::*;
}

pub mod string {
    pub use alloc::string::*;
}

pub mod vec {
    pub use alloc::vec::*;
}

pub mod result {
    pub use core::result::*;
}

pub mod option {
    pub use core::option::*;
}

pub mod slice {
    pub use core::slice::*;
}

pub mod str {
    pub use core::str::*;
}

pub use alloc::format;
