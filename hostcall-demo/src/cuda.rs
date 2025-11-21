use std::{
    ffi::CString,
    process::abort,
    sync::{Arc, LazyLock},
    time::Duration,
};

use cust::{
    error::CudaError,
    launch,
    memory::{AsyncCopyDestination, CopyDestination, DevicePointer, DeviceSlice},
    module::{Module, Symbol},
    sys,
    stream::{Stream, StreamFlags},
};
use cust::sys::CUresult::CUDA_SUCCESS;

use crate::runtime::{DevicePtr, GPUModule, Runtime};
// We use a single stream for all the copies. Why?
// Non-async memcpy is *blocking* and causes us to wait till a kernel is done running.
// The hostcall mechanism requires us to be able to write/read from GPU memory while a kernel is running.
// An async memcpy MUST be associated with a stream - hence the reason `COPY_STREAM` exists.
static COPY_STREAM: LazyLock<Stream> =
    LazyLock::new(|| Stream::new(StreamFlags::NON_BLOCKING, Some(1)).unwrap());
impl DevicePtr for DevicePointer<u8> {
    fn from_u64(val: u64) -> Self {
        Self::from_raw(val)
    }

    fn to_u64(self) -> u64 {
        self.as_raw()
    }
}
// We implement `GPUModule` for an `Arc<Module>`.
// We need to access a module to write to symbols within it, but we *also* want the user to still
// have access to this module - to launch kernels with.
impl GPUModule for Arc<cust::module::Module> {
    type Error = CudaError;
    type DevicePtr = DevicePointer<u8>;
    fn alloc(&mut self, size: usize) -> Result<Self::DevicePtr, CudaError> {
        unsafe { cust::memory::cuda_malloc::<u8>(size) }
    }
    fn write_pointer_to_symbol(
        &mut self,
        name: &str,
        value: Self::DevicePtr,
    ) -> Result<(), CudaError> {
        let sym: Symbol<DevicePointer<Self::DevicePtr>> =
            self.get_global(&CString::new(name).unwrap())?;
        // Transmute needed to deal with some stupid restrictions of `cust`.
        // It prevents us from async writing to kernel globals(we need that, see COPY_STREAM).
        let ptr: DevicePointer<Self::DevicePtr> = unsafe { std::mem::transmute(sym) };
        let mut slice = unsafe { DeviceSlice::from_raw_parts(ptr, 1) };
        unsafe { slice.async_copy_from(&[value], &*COPY_STREAM) }?;
        // Sync - this is needed for lifetime reasons(value would get out of scope)
        COPY_STREAM.synchronize()
    }
    fn write_u64_to_symbol(
        &mut self,
        name: &str,
        value: u64,
        offset: u64,
    ) -> Result<bool, CudaError> {
        // Some sanity checks
        assert_eq!(offset % (size_of::<u64>() as u64), 0);
        // If this line panics, that is a result of an over-zealous `cust` check, that panics in perfectly safe code.
        if name == "__HOSTCALL_RETURN_SLOTS__" {
            // Manually copy with the driver API to avoid any cust size checks or type mismatches.
            let sym: Result<Symbol<[u64; 64]>, CudaError> =
                self.get_global(&CString::new(name).unwrap());
            let sym = match sym {
                Ok(sym) => sym,
                Err(CudaError::NotFound) => return Ok(false),
                Err(_) => sym.unwrap(),
            };
            let ptr: DevicePointer<[u64; 64]> = unsafe { std::mem::transmute(sym) };
            let raw = ptr.as_raw() + offset;
            let status = unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    raw,
                    &value as *const _ as *const _,
                    8,
                    COPY_STREAM.as_inner(),
                )
            };
            if status != CUDA_SUCCESS {
                return Err(CudaError::IllegalAddress);
            }
            COPY_STREAM.synchronize()?;
            return Ok(true);
        }

        let sym: Result<Symbol<DevicePointer<u64>>, CudaError> =
            self.get_global(&CString::new(name).unwrap());
        let sym = match sym {
            Ok(sym) => sym,
            Err(CudaError::NotFound) => return Ok(false),
            Err(_) => sym.unwrap(),
        };
        let ptr: DevicePointer<u64> = unsafe { std::mem::transmute(sym) };
        let ptr = unsafe { ptr.add((offset as isize / 8).try_into().unwrap()) };
        let mut slice = unsafe { DeviceSlice::from_raw_parts(ptr, 1) };
        unsafe { slice.async_copy_from(&[value], &*COPY_STREAM) }.unwrap();
        // Sync - this is needed for lifetime reasons(value would get out of scope)
        COPY_STREAM.synchronize().unwrap();
        Ok(true)
    }
    fn write_bytes_to_addr(
        &mut self,
        bytes: &[u8],
        ptr: Self::DevicePtr,
    ) -> Result<(), Self::Error> {
        // Slice sillines needed to deal with some stupid restrictions of `cust`.
        // It prevents us from async writing to *pointers*(we need that, see COPY_STREAM).
        unsafe {
            DeviceSlice::from_raw_parts(ptr, bytes.len()).async_copy_from(bytes, &*COPY_STREAM)
        }?;
        // Sync - this is needed for lifetime reasons(bytes could get out of scope)
        COPY_STREAM.synchronize()
    }
    fn read_u64_slice(
        &mut self,
        ptr: Self::DevicePtr,
        len: usize,
        out: &mut Vec<u64>,
    ) -> Result<(), Self::Error> {
        out.resize(len, 0);
        unsafe {
            DeviceSlice::<u64>::from_raw_parts(ptr.cast(), len).async_copy_to(out, &*COPY_STREAM)
        }?;
        // Sync - this is needed for lifetime reasons(out could get out of scope)
        COPY_STREAM.synchronize()
    }
    fn read_u8_buff(
        &mut self,
        ptr: Self::DevicePtr,
        len: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), Self::Error> {
        out.resize(len, 0);
        unsafe {
            DeviceSlice::<u8>::from_raw_parts(ptr.cast(), len).async_copy_to(out, &*COPY_STREAM)
        }?;
        // Sync - this is needed for lifetime reasons(out could get out of scope)
        COPY_STREAM.synchronize()
    }

    fn free(&mut self, ptr: Self::DevicePtr) -> Result<(), Self::Error> {
        unsafe { cust::memory::cuda_free_async(&*COPY_STREAM, ptr) }?;
        COPY_STREAM.synchronize()
    }
}
#[repr(C)]
#[derive(Debug)]
pub(crate) struct UnparsedCommand {
    pub(crate) cmd: u32,
    pub(crate) len: u16,
    pub(crate) res: u16,
    pub(crate) data: [u8],
}
fn parse_command<'a>(mut buff: &mut &'a [u64]) -> Result<&'a UnparsedCommand, ()> {
    // SAFETY: the `buff` is at least big enough to fit a command with 0 data.
    if buff.len() <= 1 {
        return Err(());
    }
    let cmd = unsafe {
        &*(std::slice::from_raw_parts(&buff[0], 0) as *const [u64] as *const UnparsedCommand)
    };
    *buff = &buff[1..];
    let len = cmd.len;
    // SAFETY: check that the `buff` is big enough to fit all the data of this command.
    assert!(buff.len() * 4 > len as usize);
    let cmd = unsafe {
        &*(std::slice::from_raw_parts(cmd as *const _ as *const u8, len as usize) as *const [u8]
            as *const UnparsedCommand)
    };
    *buff = &buff[(cmd.len.div_ceil(4) as usize)..];
    Ok(cmd)
}
pub(crate) fn parse_commands<'a>(mut buff: &'a [u64]) -> Vec<&'a UnparsedCommand> {
    let mut commands = vec![];
    while let Ok(cmd) = parse_command(&mut buff) {
        if cmd.cmd == 0 {
            if cmd.res == 0 {
                break;
            }
            continue;
        }
        commands.push(cmd);
    }
    commands
}
fn ignore_hostcall(_: &mut Arc<cust::module::Module>, _: &[u8]) -> u64 {
    1
}
#[test]
fn cust_context() {
    let ctxt = cust::quick_init().unwrap();
    let module = Arc::new(Module::from_file("test.ptx").unwrap());
    let mut runtime = Runtime::new(module.clone()).unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, Some(0)).unwrap();
    unsafe { launch!(module.kernel_main<<<1,1,0,stream>>>() ) }.unwrap();
    runtime.register_hostcall("open", ignore_hostcall).unwrap();
    runtime.register_hostcall("write", ignore_hostcall).unwrap();

    for _ in 0..100 {
        runtime.pool_commands().unwrap();
    }
}
