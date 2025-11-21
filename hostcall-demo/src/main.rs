use cust::{
    error::CudaError,
    launch,
    memory::{AsyncCopyDestination, CopyDestination, DevicePointer, DeviceSlice},
    module::Module,
    stream::{Stream, StreamFlags},
};
use cust::sys;
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

fn read_symbol_u64(module: &Module, name: &str, count: usize) -> Vec<u64> {
    // __HOSTCALL_RETURN_SLOTS__ is an array; use the correct element type to satisfy cust's size checks.
    let (ptr, elems): (DevicePointer<u64>, usize) = if name == "__HOSTCALL_RETURN_SLOTS__" {
        let sym = module
            .get_global::<[u64; 64]>(&CString::new(name).unwrap())
            .unwrap();
        let ptr: DevicePointer<[u64; 64]> = unsafe { std::mem::transmute(sym) };
        // SAFE: pointer arithmetic on DevicePointer mirrors C-style pointer math.
        (unsafe { ptr.cast::<u64>() }, 64)
    } else {
        let sym = module
            .get_global::<u64>(&CString::new(name).unwrap())
            .unwrap();
        let ptr: DevicePointer<u64> = unsafe { std::mem::transmute(sym) };
        (ptr, 1)
    };
    let count = count.min(elems);
    let mut out = vec![0_u64; count];
    let dbg_stream = Stream::new(StreamFlags::NON_BLOCKING, Some(2)).unwrap();
    unsafe {
        DeviceSlice::from_raw_parts(ptr, count).async_copy_to(&mut out, &dbg_stream)
    }
    .unwrap();
    dbg_stream.synchronize().unwrap();
    out
}

fn read_buffer_words(module: &Module, ptr: u64, words: usize) -> Vec<u64> {
    let mut out = vec![0_u64; words];
    let dbg_stream = Stream::new(StreamFlags::NON_BLOCKING, Some(3)).unwrap();
    unsafe {
        DeviceSlice::from_raw_parts(DevicePointer::<u64>::from_raw(ptr).cast(), words)
            .async_copy_to(&mut out, &dbg_stream)
    }
    .unwrap();
    dbg_stream.synchronize().unwrap();
    out
}

fn write_return_slot(module: &Module, slot_bytes: u64, value: u64) {
    let sym = module
        .get_global::<[u64; 64]>(&CString::new("__HOSTCALL_RETURN_SLOTS__").unwrap())
        .unwrap();
    let ptr: DevicePointer<[u64; 64]> = unsafe { std::mem::transmute(sym) };
    let raw = ptr.as_raw() + slot_bytes;
    let status = unsafe { sys::cuMemcpyHtoD_v2(raw, &value as *const _ as *const _, 8) };
    assert_eq!(status, sys::CUresult::CUDA_SUCCESS);
}

fn main() {
    let ctxt = cust::quick_init().unwrap();
    let module = Arc::new(
        Module::from_ptx(std::str::from_utf8(culinux_ptx_rs::PTX).unwrap(), &[]).unwrap(),
    );
    let mut runtime = Runtime::new(module.clone()).unwrap();
    runtime.register_hostcall("open", open).unwrap();
    runtime.register_hostcall("write", write).unwrap();
    runtime
        .register_hostcall("device_malloc", device_malloc)
        .unwrap();
    runtime
        .register_hostcall("device_free", device_free)
        .unwrap();
    let debug = env::var("HOSTCALL_DEBUG").is_ok();
    if debug {
        eprintln!(
            "[hostcall-debug] __HOSTCALL__open {:?}",
            read_symbol_u64(&module, "__HOSTCALL__open", 1)
        );
        eprintln!(
            "[hostcall-debug] __HOSTCALL_BUFF_PTR__ {:?}",
            read_symbol_u64(&module, "__HOSTCALL_BUFF_PTR__", 1)
        );
        eprintln!(
            "[hostcall-debug] __HOSTCALL_BUFF_SIZE__ {:?}",
            read_symbol_u64(&module, "__HOSTCALL_BUFF_SIZE__", 1)
        );
        eprintln!(
            "[hostcall-debug] __HOSTCALL_BUFF_TOP__ {:?}",
            read_symbol_u64(&module, "__HOSTCALL_BUFF_TOP__", 1)
        );
        eprintln!(
            "[hostcall-debug] return slots {:?}",
            read_symbol_u64(&module, "__HOSTCALL_RETURN_SLOTS__", 4)
        );
    }

    let stream = Stream::new(StreamFlags::NON_BLOCKING, Some(0)).unwrap();

    unsafe { launch!(module.kernel_main<<<1,1,0,stream>>>() ) }.unwrap();

    let start_time = std::time::Instant::now();
    if debug {
        for poll in 0..5 {
            let buff_ptr = read_symbol_u64(&module, "__HOSTCALL_BUFF_PTR__", 1)[0];
            let buff_size = read_symbol_u64(&module, "__HOSTCALL_BUFF_SIZE__", 1)[0] as usize;
            let top = read_symbol_u64(&module, "__HOSTCALL_BUFF_TOP__", 1)[0];
            let words = if buff_size == 0 { 8 } else { buff_size.min(16) };
            let buff = read_buffer_words(&module, buff_ptr, words);
            eprintln!(
                "[hostcall-debug] poll {poll}: ptr {buff_ptr} size {buff_size} top {top} head {:?}",
                &buff[..words.min(buff.len())]
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        eprintln!("[hostcall-debug] running a single runtime poll");
        runtime.pool_commands().unwrap();
        eprintln!(
            "[hostcall-debug] return slots after poll {:?}",
            read_symbol_u64(&module, "__HOSTCALL_RETURN_SLOTS__", 4)
        );
    }
    let mut polls = 0;
    for _ in 0..1_000_000 {
        std::thread::sleep(Duration::from_nanos(100 as u64));
        std::thread::yield_now();
        runtime.pool_commands().unwrap();
        polls += 1;
    }
    stream.synchronize().unwrap();
    eprintln!("{} ns (polled {polls} times)", start_time.elapsed().as_micros())
}
