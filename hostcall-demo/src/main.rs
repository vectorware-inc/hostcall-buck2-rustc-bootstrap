use cust::{
    error::CudaError,
    launch,
    memory::{AsyncCopyDestination, CopyDestination, DevicePointer, DeviceSlice},
    module::Module,
    stream::{Stream, StreamFlags},
};
use std::{
    env,
    ffi::CString,
    io::Write,
    os::fd::{AsFd, AsRawFd, FromRawFd},
    process::abort,
    sync::{Arc, LazyLock},
    time::Duration,
};
mod hostcalls;
mod runtime;

use std::fmt::Debug;

use crate::{
    cuda::{UnparsedCommand, parse_commands},
    runtime::{DevicePtr, GPUModule, Runtime},
};
mod cuda;
#[repr(C)]
#[derive(Debug)]
struct FileOpen {
    ptr: u64,
    len: u64,
    flags: u32,
    mode: u32,
}
fn device_malloc(gpu: &mut Arc<cust::module::Module>, data: &[u8]) -> u64 {
    eprintln!("Doing a `malloc`!");
    let len = u64::from_le_bytes(data[..8].try_into().unwrap());
    gpu.alloc(len as usize).unwrap().as_raw()
}
fn device_free<G: GPUModule>(gpu: &mut G, data: &[u8]) -> u64 {
    let ptr = u64::from_le_bytes(data[..8].try_into().unwrap());
    match gpu.free(G::DevicePtr::from_u64(ptr)) {
        Ok(_) => 1,
        Err(_) => 2,
    }
}
fn open(gpu: &mut Arc<cust::module::Module>, data: &[u8]) -> u64 {
    assert!(data.len() >= size_of::<FileOpen>());
    assert!(data.as_ptr().cast::<FileOpen>().is_aligned());
    let file_open = unsafe { &*data.as_ptr().cast::<FileOpen>() };
    let mut file_path = Vec::new();
    gpu.read_u8_buff(
        DevicePointer::from_raw(file_open.ptr),
        file_open.len as usize,
        &mut file_path,
    )
    .unwrap();
    let file_path = std::str::from_utf8(&file_path).unwrap();
    eprintln!("We being asked to open a file {file_open:?}, named {file_path:?}!");
    let file = std::fs::File::create(file_path);
    eprintln!("Opened a file:{file:?}");
    let file = file.unwrap();
    let fd = file.as_raw_fd();
    std::mem::forget(file);
    fd as u64
}
#[repr(C)]
#[derive(Debug)]
struct FileWrite {
    ptr: u64,
    len: u32,
    fd: i32,
}
fn write(gpu: &mut Arc<cust::module::Module>, data: &[u8]) -> u64 {
    assert!(data.len() >= size_of::<FileWrite>());
    assert!(data.as_ptr().cast::<FileWrite>().is_aligned());
    let file_write = unsafe { &*data.as_ptr().cast::<FileWrite>() };
    let mut buff = Vec::new();
    eprintln!("We being asked to write to a file! {file_write:?} ");
    let write_data = gpu.read_u8_buff(
        DevicePointer::from_raw(file_write.ptr),
        file_write.len as usize,
        &mut buff,
    );
    eprintln!("Buffer read. It is {buff:?}");
    eprintln!("write_data:{buff:?}");
    let len = unsafe {
        std::fs::File::from_raw_fd(file_write.fd)
            .write(&buff)
            .unwrap()
    };
    (len as u64).max(1)
}
fn main() {
    let ctxt = cust::quick_init().unwrap();
    let ptx_path = env::var("HOSTCALL_PTX")
        .or_else(|_| {
            option_env!("HOSTCALL_PTX_PATH")
                .map(|s| s.to_string())
                .ok_or(env::VarError::NotPresent)
        })
        .unwrap_or_else(|_| "test.ptx".to_string());
    let module = Arc::new(Module::from_file(&ptx_path).unwrap());
    let mut runtime = Runtime::new(module.clone()).unwrap();
    runtime.register_hostcall("open", open).unwrap();
    runtime.register_hostcall("write", write).unwrap();
    runtime
        .register_hostcall("device_malloc", device_malloc)
        .unwrap();
    runtime
        .register_hostcall("device_free", device_free)
        .unwrap();

    let stream = Stream::new(StreamFlags::NON_BLOCKING, Some(0)).unwrap();

    unsafe { launch!(module.kernel_main<<<1,1,0,stream>>>() ) }.unwrap();

    let start_time = std::time::Instant::now();
    for _ in 0..1000 {
        std::thread::sleep(Duration::from_nanos(100 as u64));
        std::thread::yield_now();
        runtime.pool_commands().unwrap();
    }
    stream.synchronize().unwrap();
    eprintln!("{} ns", start_time.elapsed().as_micros())
}
