use core::arch::asm;
use core::arch::nvptx::vprintf;
use core::mem::size_of;
use core::ptr::{copy_nonoverlapping, null_mut};
use core::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
#[unsafe(no_mangle)]
#[used]
pub static __HOSTCALL_BUFF_PTR__: AtomicPtr<u8> = AtomicPtr::new(null_mut());
#[unsafe(no_mangle)]
#[used]
pub static __HOSTCALL_BUFF_SIZE__: AtomicUsize = AtomicUsize::new(0);
#[unsafe(no_mangle)]
#[used]
pub static __HOSTCALL_BUFF_TOP__: AtomicUsize = AtomicUsize::new(0);
#[unsafe(no_mangle)]
#[used]
pub static __HOSTCALL_RETURN_SLOTS__: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
pub fn acquire_return_slot() -> u16 {
    let idx = 1;
    __HOSTCALL_RETURN_SLOTS__[idx].store(0, Ordering::Relaxed);
    return idx as u16;
}
pub fn free_return_slot(slot: u16) {}
#[inline(never)]
pub fn pool_return_slot(slot: u16) -> u64 {
    unsafe{core::ptr::read_volatile(__HOSTCALL_RETURN_SLOTS__.as_ptr().cast::<u64>().add(slot as usize))}
}
#[repr(C)]
struct CommandHeader {
    cmd: u32,
    len: u16,
    res: u16,

}

pub fn sleep_ns_kernel(ns: u32) {
    unsafe {
        asm!(
            "nanosleep.u32 {t};",
            t = in(reg32) ns,
            options(nostack)
        )
    };
}
pub fn submit_hostcall(cmd: u32, res: u16, cmd_data: *const (), len: u16) -> Result<(), ()> {
    let bytes = size_of::<CommandHeader>() + len as usize;
    let words = ((bytes + size_of::<u64>() - 1) / size_of::<u64>()) * size_of::<u64>();
    let offset = __HOSTCALL_BUFF_TOP__.fetch_add(words, Ordering::Relaxed);
    if offset + words > __HOSTCALL_BUFF_SIZE__.load(Ordering::Relaxed) {
        let mut fmt = [0];
        unsafe {
            vprintf(
                c"HOSTCALL_BUFF_FULL!\n".as_ptr() as *const _,
                &raw mut fmt as *const _,
            )
        };

        return Err(());
    }
    let hostcall_buff_ptr = __HOSTCALL_BUFF_PTR__.load(Ordering::Relaxed);

    let cmd_ptr: &mut CommandHeader =
        unsafe { &mut *hostcall_buff_ptr.offset(offset as isize).cast() };

    cmd_ptr.res = res;
    cmd_ptr.len = len;
    let data_dst = unsafe { (cmd_ptr as *mut _ as *mut u8).add(size_of::<CommandHeader>()) };
    unsafe {
        copy_nonoverlapping(cmd_data as *const u8, data_dst, len as usize);
    };
    cmd_ptr.cmd = cmd;
    if hostcall_buff_ptr != __HOSTCALL_BUFF_PTR__.load(Ordering::Relaxed) {
        return Err(());
    }
    Ok(())
}
