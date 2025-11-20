#![feature(abi_gpu_kernel, asm_experimental_arch, stdarch_nvptx)]

#[allow(unused_extern_crates)]
extern crate libc_hostcall as _libc_hostcall;
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn kernel_main() {
    use std::io::Write;
    let msg = format!("This file was created *from a GPU*, using the Rust standard library :D\n We are {:?}", "VectorWare".to_owned());
    std::fs::File::create("rust_from_gpu.txt").unwrap().write_all(msg.as_bytes()).unwrap();
}
#[used]
static KEEP_KERNEL_MAIN: &[extern "gpu-kernel" fn() -> ()] = &[kernel_main];
