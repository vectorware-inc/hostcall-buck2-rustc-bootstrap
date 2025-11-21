#![no_std]
#![feature(alloc_error_handler)]

extern crate alloc;
extern crate libc_hostcall;

use core::panic::PanicInfo;

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
