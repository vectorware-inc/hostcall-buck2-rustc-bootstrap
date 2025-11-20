use crate::runtime::{GPUModule, Runtime};
/// Injects all the fs hostcalls into the GPU module.
pub fn fs_hostcalls(r: Runtime<impl GPUModule>) {}
