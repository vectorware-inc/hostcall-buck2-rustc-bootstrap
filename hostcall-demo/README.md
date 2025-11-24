# Hostcalls

This crate is a small demo of GPU code invoking host CPU code (a "hostcall"). The kernel is written in Rust (see [`culinux/src/lib.rs`](culinux/src/lib.rs)) and opens/writes files via hostcalls; the host runtime handles `open` and `write` on the CPU side.

## How to run

1) Install Buck2: follow https://buck2.build/docs/getting_started/install/ (binary or cargo install).
2) Run the demo without extra logging: `buck2 run //hostcall-demo:hostcalls_bin`.
3) Run with hostcall debug output: `HOSTCALL_DEBUG=1 buck2 run //hostcall-demo:hostcalls_bin`.

# Building with Buck

This repository now exposes native Buck2 targets that cover the demo end to end:

- `buck2 build //hostcall-demo:hostcalls_bin` – builds the host runtime against the
  locally vendored third-party crates produced via [`reindeer`](https://github.com/facebookincubator/reindeer).
- `buck2 build //hostcall-demo:culinux_ptx` – stitches together the nvptx static
  archives (`std`, `alloc`, `core`, `panic_abort`, and `libc-hostcall`), links them
  into bitcode with `llvm-link` from `//stage0:ci_llvm`, runs a small `opt` pipeline,
  and emits a PTX module.
- `buck2 run //hostcall-demo:hostcalls_bin` – the binary depends on the PTX target and
  embeds its hash/path at compile time, so invoking it via Buck automatically runs the
  full hostcall demo end to end.

The PTX rule relies on the vendored `ci-llvm` toolchain plus the nvptx sysroot this
repository produces; no external nightly toolchain or `build.sh` invocation is needed.

## How to run

1) Install Buck2: follow https://buck2.build/docs/getting_started/install/ (binary or cargo install).
2) Run the demo without extra logging: `buck2 run //hostcall-demo:hostcalls_bin`.
3) Run with hostcall debug output: `HOSTCALL_DEBUG=1 buck2 run //hostcall-demo:hostcalls_bin`.

### Design notes

#### Buck2 layout and flow
This repo uses a single Buck2 workspace for both host and nvptx artifacts. Host pieces (binaries, libraries, third-party crates from `reindeer`) live under `hostcall-demo/` and are built with the bootstrap Rust rules. The nvptx toolchain, target specs, and minimal std/alloc fixups are defined at the workspace root (for reuse across crates) and wired into the nvptx platform (`//platforms/nvptx:library`). The PTX pipeline is expressed as Buck targets (`culinux_ptx`, `culinux_ptx_rs_lib`) so the host binary depends directly on the generated PTX and its hash—no external scripts or manual file plumbing are needed.

#### PTX build inside Buck
Buck2 flattens the nvptx sysroot (core/alloc/std/panic_abort/libc-hostcall) and the kernel into LLVM bitcode via `llvm-link` from `//stage0:ci_llvm`, runs internalize + global DCE + O2, lowers to PTX, and embeds the PTX bytes plus a SHA-256 hash into the host binary. Keeping this in Buck provides deterministic toolchain inputs and automatically invalidates caches when the kernel or sysroot change.

#### Platform and target selection
All nvptx builds are forced through `//platforms/nvptx:library` and the `nvptx64-vectorware-*` target spec in `//target_specs`, which pins the nvptx JSON and selects the linux-oriented libc emulation we ship. This keeps the ABI/flags in sync with the nvptx sysroot and ci-llvm version. The generated PTX is wrapped by `culinux_ptx_rs_lib`, exposing the bytes and hash to the host crate at compile time.

#### Hostcall path
Kernels declare `__HOSTCALL__*` globals. The host registers handlers, writes numeric IDs into those globals, and drives a double-buffered GPU command queue (two 4 KiB buffers). Commands are parsed on the host, executed, and completions are written into per-kernel return slots that the GPU polls. Dynamic IDs decouple kernel codegen from host symbol names, and double-buffering keeps enqueue and processing overlapped.

#### Std/alloc fixups
`fixups/std/nvptx_vectorware.rs` provides the minimal std/alloc surface for nvptx, routes allocation to libc-hostcall `malloc/free`, and exports allocator symbols with `#[rustc_std_internal_symbol]` Rust-ABI shims so names stay stable across crate-disambiguator changes while satisfying core/alloc expectations.

#### Host runtime
The host runtime (built on `cust`) loads PTX, patches globals (`__HOSTCALL__*`, buffer pointers, sizes, return slots) via async device writes, and polls the command buffer while the kernel runs. This allows hostcalls to execute concurrently with GPU work without blocking the stream.

# Inner workings
## Command buffer
Under the hood, the GPU uses a double-buffered command queue so the host can read one buffer while the GPU writes the other.
```c
// A pointer to the current buffer. Swapped by the host after each poll.
// `uint64_t` keeps pointer alignment.
__device__ uint64_t* __HOSTCALL_BUFF_PTR__;
// Current top of the buffer in bytes. Threads atomically bump this to reserve space.
__device__ unsigned long long __HOSTCALL_BUFF_TOP__;
// Size of the buffer in bytes.
__device__ unsigned long long __HOSTCALL_BUFF_SIZE__;
```
The host allocates two 4 KiB buffers, initializes `__HOSTCALL_BUFF_PTR__` and `__HOSTCALL_BUFF_SIZE__`, and flips the pointer on each poll after clearing the next buffer. This keeps enqueueing and host processing overlapped.

This is an internal detail; GPU code uses `submit_hostcall` to enqueue work.
## `submit_hostcall`
`submit_hostcall` takes a command ID, a return-slot ID, and an arbitrary payload. It reserves space in the command buffer and writes a header plus payload.
```c
__device__ enum HOSTCALL_SUBMIT_RESULT submit_hostcall(
        uint32_t CMD, // The ID of this command
        uint16_t RES, // The ID of the return slot(will explain later)
        // Command data payload
        const void* CMD_DATA,
        uint16_t CMD_DATA_LEN
    )
```
It will then pack this data into the command buffer - but we need to allocate space there first.
To do so, it computes the size of the command in 64 bit doublewords.
```c
// We compute the size of the entire command here - in 64 bit intigers.
// This is needed to allocate enough space in the command buffer
uint64_t bytes = (sizeof(struct CommandHeader) + CMD_DATA_LEN);
unsigned long long words = ((bytes + sizeof(uint64_t)- 1) / sizeof(uint64_t)) * sizeof(uint64_t);
```
This is needed, because all commands are aligned to 64 bytes(this simplifes decoding).
We then just atomically increment `__HOSTCALL_BUFF_TOP__`, allocating memory for the hostcall.
```c
uint64_t offset = atomicAdd(&__HOSTCALL_BUFF_TOP__, words);
// Handle the overflow case
if(offset + words > __HOSTCALL_BUFF_SIZE__) {
    return HOSTCALL_BUFF_FULL;
}
```
The kernel then writes the header and payload. It snapshots `__HOSTCALL_BUFF_PTR__` before writing and checks for a torn write (host swapping the buffer mid-write).
```c
uint64_t* hostcall_buff_ptr = __HOSTCALL_BUFF_PTR__;
// Copy some command data
struct CommandHeader* cmd = (struct CommandHeader*)&hostcall_buff_ptr[offset];
cmd->RES = RES;
cmd->CMD_DATA_LEN = CMD_DATA_LEN;
// Copy command payload
memcpy(cmd->DATA, CMD_DATA, CMD_DATA_LEN);
cmd->CMD = CMD;
// Write tear detection
if (hostcall_buff_ptr != __HOSTCALL_BUFF_PTR__)return HOSTCALL_WRITE_TORN;
return HOSTCALL_OK;
```

And that is all of `submit_hostcall` - there are a few more things to explain, tough.

## Hostcall resolution

Hostcalls are assigned IDs dynamically at runtime. To declare a hostcall, the kernel defines globals named `__HOSTCALL__$NAME`, for example:
```c
__device__ uint64_t __HOSTCALL__open;
__device__ uint64_t __HOSTCALL__write;
```
When the host registers handlers, it writes sequential IDs into these globals. GPU code then submits the command using those IDs:
```c
struct WriteCommand cmd;
cmd.data = data;
cmd.length = length;
cmd.fd = fd;
submit_hostcall(__HOSTCALL__write, res,&write_cmd,sizeof(write_cmd));
```

There is one more big thing to discuss - return slots. Why they exist, their shortcomings, and benefits.

## Return slots
After submitting a hostcall, the GPU needs a place the host can write the result. Return slots provide that shared location.
```c
// 64 for the purposes of the demo
__device__ uint64_t __HOSTCALL_RETURN_SLOTS__[64];
```
`acquire_return_slot` returns a slot ID (currently a fixed slot for the demo); the host writes the result into that slot, and the GPU polls until it becomes nonzero. The host runtime clears slots on drop to avoid stale values.
```c
// Allocate space for the host to write the timestamp to
uint16_t res = acquire_return_slot();
```
Slots are bounded (64 in this demo) and avoid per-call GPU allocations; because they are statics, the host can always reach them to signal completion or failure.

# CPU-side of things

The host runtime is written in Rust using `cust`:

- Loads the embedded PTX (`culinux_ptx_rs::PTX`) and registers hostcall handlers for `open`, `write`, `device_malloc`, and `device_free`.
- Allocates two device buffers, initializes `__HOSTCALL_BUFF_PTR__`, `__HOSTCALL_BUFF_SIZE__`, and `__HOSTCALL_BUFF_TOP__`, and swaps buffers after each poll.
- Parses commands, invokes the registered handler, writes the result into `__HOSTCALL_RETURN_SLOTS__`, and continues polling while the kernel runs.

The runtime(implemented in `runtime.rs`) is abstracted over a `GPUModule` trait - I wanted to be able to add an OpenCL/Vulkan/Whatever host in the future, if possible.

This maybe lead to some over-abstractions - who knows.

TODO: describe the hostcall API in more detail, talk about zero-copy & macros we could use to make this easier.

# Issues in `cust` & Panics

This was built against a patched version of `cust`, to prevent some issues.
That crate has some over-zealous checks, that cause it to panic when we write to global arrays.
Essentially, if try to get the first element of this array:
```c
uint64_t RES[64];
```
By getting a pointer to a `u64`, and then offsetting it, `cust` would panic, cause `u64` is not the same size as `RES`.
Such a write is still sound:there is no requirement that we write to a whole array at once. Still, the check gets erroneously triggered.

I will find a way around this soon, or submit a PR with my `cust` patch. Or we may ditch cust entirely - who knows.
