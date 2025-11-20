# Hostcalls

This crate is a small demo of GPU code invoking host CPU code(I call this a "hostcall").

It includes an example CUDA shader, which opens a file, and then writes a small message into it.

```c
extern "C" __global__ void kernel_main() {
   // Create `file_from_gpu.txt` on our hard drive... 
   int32_t fd = open("file_from_gpu.txt",strlen("file_from_gpu.txt"), 0, 2);
   // ... and write our neat message to it!
   write(fd, "GPU says hello world :)!", strlen("GPU says hello world :)!"));
}
```

The repo also contains a small CUDA-based(can be extended to other APIs) runtime, which can be used to invoke this kernel, and implement the required host-side functionality(opening & writing to files). 

An implementation of a command can be as simple as this:
### CPU-side command impl
```rust
fn unix_time(gpu:&mut impl GPUModule, cmd_data:&[u8])->u64{
    // Rust-side code to get the timestamp 
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("time should go forward");
    // Our data is small enough to fit in 64 bytes - we can return it directly. 
    since_the_epoch.as_milis()
}
// Register the hostcall with the runtime.
// This will cause the runtime to set appropieate kernel-side data, 
// allowing the kernel to invoke a host function
runtime.register_hostcall("unix_time", unix_time).unwrap();
```
### GPU-side glue code
```c
// The command ID - will be automatically resolved by the runtime
__device__ uint64_t __HOSTCALL__unix_time;
// A wrapper around the hostcall
__device__ uint64_t unix_time(){
    // Allocate space for the host to write the timestamp to
    uint16_t res = acquire_return_slot();
    // Submit the command to a queue. 
    submit_hostcall(__HOSTCALL__unix_time, res,NULL,0);
    // Periodically check that the return slot has a nonzero value
    // This indicates that the hostcall is done. 
    while(pool_return_slot(res) == 0){
        // Yield execution, allowing a different warp to be scheduled to this SM
        __nanosleep(1);
    }
    // Extract the timestamp 
    uint64_t resval = pool_return_slot(res);
    // Free the space host used to write the timestamp
    free_return_slot(res);
    return resval;
}
```
This code is enough to allow a GPU kernel to get the current UNIX timestamp. 

# Building with Buck

This repository now exposes native Buck2 targets that cover the demo end to end:

- `buck2 build //hostcall-demo:hostcalls_bin` – builds the host runtime against the
  locally vendored third-party crates produced via `reindeer`.
- `buck2 build //hostcall-demo:culinux_ptx` – reproduces the old `build.sh` flow inside
  a deterministic genrule. The rule invokes `cargo +nightly build -Zbuild-std`, performs
  the `llvm-link/opt/llc` pipeline, and emits `libculinux.ptx` into `buck-out`.
- `buck2 run //hostcall-demo:hostcalls_bin` – the binary now depends on the PTX target
  and embeds its path at compile time, so invoking it via Buck automatically runs the
  full hostcall demo.

The kernel build expects a `nightly` toolchain with `-Zbuild-std` enabled as well as
the LLVM utilities `llvm-link`, `opt`, `llvm-dis`, and `llc` in `PATH`.

# Inner workings 
## Command buffer
Under the hood, the GPU uses a buffer for communication.
```c
// A pointer to the current buffer. Will get peroidcally swapped out by the CPU.
// We use `uint64_t` here to always maintain correct aligement(for things like pointers). 
__device__ uint64_t* __HOSTCALL_BUFF_PTR__;
// The current top of the buffer - threads atomically increment this to allocate space for commands. 
__device__ unsigned long long __HOSTCALL_BUFF_TOP__;
// Size of the buffer - configurable, if a lot of threads are writing big commands. 
__device__ unsigned long long __HOSTCALL_BUFF_SIZE__;
```
`__HOSTCALL_BUFF_PTR__` is periodically swapped, to point to a different buffer.
This allows the CPU to be reading one buffer, while the GPU is writing to a different one. 

This is, however, just implementation detali, that GPU code ought not really on. The GPU code should just call `submit_hostcall` instead. 
## `submit_hostcall`
The interface of `submit_hostcall` is pretty simple. It takes in a command id, an id of a result slot(where the return variable will be written), and the command payload.
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
All we need to do now is write our command data packet into the allocated space. 
I save a copy of `hostcall_buff_ptr` locally here - this allows us to detect torn writes(host swaps the buffer from under us when we are writing a command). 
I have not seen a torn write quite yet, but I suspect they could be a problem.
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

Hostcalls are assigned IDs dynamically, at runtime(hostcalls can even be added when a kernel is running!).

I will discuss the low-level details here. I plan for them to be hidden from the user(via macros), but I still think they are worth understanding. 

To declare a hostcall, the kernel must create a 64 bit constant of name `__HOSTCALL__$NAME`, where `$NAME` is the name of a given hostcall. For example, to declare that you require hostcalls `open` & `write`, you would add this:
```c
__device__ uint64_t __HOSTCALL__open;
__device__ uint64_t __HOSTCALL__write;
```
When the hostcall `open` & `write` is registered, the runtime will write the IDs of those hostcalls to the corresponding kernel variables.

Originally, I planned on doing some PTX-parsing + code generation here(to create the glue code on the fly), but I decided against that.
It is hard to say if that would have any benefit, and would make the implementation considerably more complex.

We can generate the glue code ahead of time with macros.

To perform those hostcalls, you then just need to submit the command with the right ID to the buffer, like this:
```c
struct WriteCommand cmd;
cmd.data = data;
cmd.length = length;
cmd.fd = fd;
submit_hostcall(__HOSTCALL__write, res,&write_cmd,sizeof(write_cmd));
```

There is one more big thing to discuss - return slots. Why they exist, their shortcomings, and benefits. 

## Return slots
What kind of problem do return slots solve? Well, after we submit a hostcall, we need the host to be able to tell the GPU it is done.
Ideally, the host would write a value *somehwhere*, telling us it is done. 

We can't use a pointer to local variable here - it is in kernel-local space, and thus inaccessible to the host.
We can't use kernel's malloc here either - it allocates space inaccessible to the host.

So... where do we write the result of the hostcall? To a "return slot". 
```c
// 64 for the purposes of the demo
__device__ uint64_t __HOSTCALL_RETURN_SLOTS__[64];
```
This is arguably a dirty hack, but it works. We can always write to kernel statics like this. 
We should probably do something smarter here in the future, but this works well enough for now. 
`acquire_return_slot` just uses some atomics(not implemented yet) to get a free return slot. 
```c
// Allocate space for the host to write the timestamp to
uint16_t res = acquire_return_slot();
```
The exact allocation algorithm comes from a lockless allocator I was tinkering with some time back.
It is not *limited to* 64 slots(can handle arbitrary amount of them), but it is easier to get it working for those.

### Shortcomings of return slots

My main problem with returns slots is mostly ideological - they just feel a bit dirty to me.
They set a limit on the amount of commands in flight, which I don't like. 
The amount of slots could be grown at any time, but it is still another thing to manage. 

### Benefits of return slots

I have *heard* that CUDA malloc is quite slow. Return slots allow us to bypass it completely, and resort to atomics. Remais to be seen if they are better.

The host can also very easily find all the return slots - this is very useful in the case of runtime issues. 
If, for example, the host is crashing, it can write an error code to all the return slots, preventing deadlocks.

# CPU-side of things

I will now very briefly/broadly describe the CPU-side of things. 
It is written in Rust, so it should be a bit simpler to understand.

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
