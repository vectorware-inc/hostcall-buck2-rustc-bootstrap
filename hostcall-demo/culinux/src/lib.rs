#![no_std]
#![feature(abi_gpu_kernel, asm_experimental_arch, stdarch_nvptx, rustc_private)]

extern crate std;

use std::borrow::ToOwned;
use std::format;
use std::io::Write;

#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn kernel_main() {
    let msg = format!("This file was created *from a GPU*, using the Rust standard library :D\n We are {:?}", "VectorWare".to_owned());
    std::fs::File::create("rust_from_gpu.txt")
        .unwrap()
        .write_all(msg.as_bytes())
        .unwrap();
    // Second write to confirm kernel continues normally without exit.
    std::fs::File::create("rust_from_gpu6.txt")
        .unwrap()
        .write_all(msg.as_bytes())
        .unwrap();
}
#[used]
static KEEP_KERNEL_MAIN: &[extern "gpu-kernel" fn() -> ()] = &[kernel_main];
