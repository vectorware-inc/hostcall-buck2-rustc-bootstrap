use std::fmt::Debug;

use crate::cuda::parse_commands;
pub trait HostCall<G: GPUModule> {
    fn process(&mut self, gpu: &mut G, data: &[u8]) -> u64;
}
impl<G: GPUModule, F: FnMut(&mut G, &[u8]) -> u64> HostCall<G> for F {
    fn process(&mut self, gpu: &mut G, data: &[u8]) -> u64 {
        self(gpu, data)
    }
}
pub struct Runtime<G: GPUModule> {
    gpu: G,
    /// A pointer to the current buffer, and the next one.
    buffers: [G::DevicePtr; 2],
    /// A set of symbols, and their associated HostCall impls.
    host_calls: Vec<(String, Box<dyn HostCall<G>>)>,
}
impl<G: GPUModule> Drop for Runtime<G> {
    fn drop(&mut self) {
        for res in 0..64 {
            if let Err(err) = self
                .gpu
                .write_u64_to_symbol("__HOSTCALL_RETURN_SLOTS__", 0xDEAD_BEEF, res * 8)
            {
                eprintln!(
                    "[hostcall] failed to reset return slot {res}: {err:?}, skipping the rest"
                );
                break;
            }
        }
    }
}
impl<G: GPUModule> Runtime<G> {
    pub fn new(mut gpu: G) -> Result<Self, G::Error> {
        let buffers = [gpu.alloc(4096).unwrap(), gpu.alloc(4096).unwrap()];
        gpu.write_pointer_to_symbol("__HOSTCALL_BUFF_PTR__", buffers[0])?;
        // The CUDA side treats __HOSTCALL_BUFF_SIZE__ as a byte count, so use the full buffer length.
        let wrote_size = gpu.write_u64_to_symbol("__HOSTCALL_BUFF_SIZE__", 4096, 0)?;
        if !wrote_size {
            eprintln!("[hostcall] failed to write __HOSTCALL_BUFF_SIZE__");
        }
        gpu.write_u64_to_symbol("__HOSTCALL_BUFF_TOP__", 0, 0)
            .unwrap();
        Ok(Self {
            gpu,
            buffers: buffers,
            host_calls: vec![],
        })
    }
    pub fn register_hostcall(
        &mut self,
        name: impl Into<String>,
        hostcall: impl HostCall<G> + 'static,
    ) -> Result<(), G::Error> {
        let idx = self.host_calls.len() + 1;
        let name = name.into();
        self.gpu
            .write_u64_to_symbol(&format!("__HOSTCALL__{name}"), idx as u64, 0)?;
        self.host_calls.push((name, Box::new(hostcall)));
        Ok(())
    }
    fn pool_cmdbuffer(&mut self, buff: &mut Vec<u64>) -> Result<(), G::Error> {
        // Read the buffer the GPU just filled.
        self.gpu
            .read_u64_slice(self.buffers[0], 4096 / size_of::<u64>(), buff)?;
        // If nothing was written yet, avoid swapping/clearing buffers so we don't race the GPU.
        if buff.iter().all(|&w| w == 0) {
            return Ok(());
        }
        // Prepare the alternate buffer for the next round.
        self.gpu.write_bytes_to_addr(&[0; 4096], self.buffers[1])?;
        self.gpu
            .write_pointer_to_symbol("__HOSTCALL_BUFF_PTR__", self.buffers[1])?;
        let wrote_top = self.gpu.write_u64_to_symbol("__HOSTCALL_BUFF_TOP__", 0, 0)?;
        if !wrote_top {
            eprintln!("[hostcall] failed to write __HOSTCALL_BUFF_TOP__");
        }
        // Swap so the GPU writes into buffers[0] next and we read the other slot.
        self.buffers = [self.buffers[1], self.buffers[0]];
        Ok(())
    }
    pub fn pool_commands(&mut self) -> Result<(), G::Error> {
        let mut buff = Vec::new();
        self.pool_cmdbuffer(&mut buff).unwrap();
        if buff.iter().any(|&w| w != 0) {
            eprintln!("[hostcall] command buffer head: {:?}", &buff[..4.min(buff.len())]);
        }
        let cmds = parse_commands(&mut &buff[..]);
        if !cmds.is_empty() {
            let summary: Vec<(u32, u16, u16)> = cmds
                .iter()
                .map(|c| (c.cmd, c.len, c.res))
                .collect();
            eprintln!("[hostcall] parsed cmds: {summary:?}");
        }
        //eprintln!("Preparing to exec hostcalls: {cmds:?} {buff:?}");
        for cmd in cmds {
            //eprintln!("cmd:{cmd:?}");
            let Some((sym, handler)) = self.host_calls.get_mut(cmd.cmd as usize - 1) else {
                eprintln!("Unsupported hostcall {cmd:?}");
                let wrote = self
                    .gpu
                    .write_u64_to_symbol("__HOSTCALL_RETURN_SLOTS__", 1, cmd.res as u64 * 8)?;
                if !wrote {
                    eprintln!("[hostcall] failed to write return slot for unsupported call");
                }
                continue;
            };
            //eprintln!("Executing hostcall {sym:?}");
            let res = handler.process(&mut self.gpu, &cmd.data);
            let wrote = self
                .gpu
                .write_u64_to_symbol("__HOSTCALL_RETURN_SLOTS__", res, cmd.res as u64 * 8)?;
            if !wrote {
                eprintln!(
                    "[hostcall] failed to write return slot {} for {sym}",
                    cmd.res
                );
            }
            eprintln!("[hostcall] completed {sym} -> {res} (slot {})", cmd.res);
        }
        Ok(())
    }
}
pub(crate) trait DevicePtr: Copy + Debug {
    fn from_u64(val: u64) -> Self;
    fn to_u64(self) -> u64;
}
pub(crate) trait GPUModule {
    type Error: Debug;
    type DevicePtr: DevicePtr;
    fn alloc(&mut self, size: usize) -> Result<Self::DevicePtr, Self::Error>;
    fn free(&mut self, ptr: Self::DevicePtr) -> Result<(), Self::Error>;
    fn write_pointer_to_symbol(
        &mut self,
        name: &str,
        value: Self::DevicePtr,
    ) -> Result<(), Self::Error>;
    fn write_u64_to_symbol(
        &mut self,
        name: &str,
        value: u64,
        offset: u64,
    ) -> Result<bool, Self::Error>;
    fn read_u64_slice(
        &mut self,
        ptr: Self::DevicePtr,
        len: usize,
        out: &mut Vec<u64>,
    ) -> Result<(), Self::Error>;
    fn read_u8_buff(
        &mut self,
        ptr: Self::DevicePtr,
        len: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), Self::Error>;
    fn write_bytes_to_addr(
        &mut self,
        bytes: &[u8],
        ptr: Self::DevicePtr,
    ) -> Result<(), Self::Error>;
}
