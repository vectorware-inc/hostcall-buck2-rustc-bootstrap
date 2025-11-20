#![no_std]
#![cfg_attr(target_arch = "nvptx64", feature(asm_experimental_arch, stdarch_nvptx, c_str_literals))]

#[cfg(target_arch = "nvptx64")]
mod nvptx;

#[cfg(target_arch = "nvptx64")]
pub use nvptx::*;
