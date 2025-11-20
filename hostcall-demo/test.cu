#include <cstdint> 
#include <stdio.h>
__device__ uint64_t __HOSTCALL__open = ~0;
__device__ uint64_t __HOSTCALL__write = ~0;
__device__ uint64_t __HOSTCALL__device_malloc = ~0;

__device__ uint64_t* __HOSTCALL_BUFF_PTR__;
__device__ unsigned long long __HOSTCALL_BUFF_SIZE__;
__device__ unsigned long long __HOSTCALL_BUFF_TOP__;
__device__ uint64_t __HOSTCALL_RETURN_SLOTS__[64];
__device__ uint64_t __HOSTCALL_SLOT_MASK__;
// Skeleton IMPL of return slots. Does not use atomics yet :(
__device__ uint16_t acquire_return_slot(){
    uint16_t slot = 0;
    __HOSTCALL_RETURN_SLOTS__[slot] = 0;
    return slot;
}
__device__ void free_return_slot(uint16_t slot){

}
__device__ uint64_t pool_return_slot(uint16_t slot){
    return __HOSTCALL_RETURN_SLOTS__[slot];
}
enum HOSTCALL_SUBMIT_RESULT{
    HOSTCALL_OK = 0,
    HOSTCALL_BUFF_FULL = 1,
    HOSTCALL_WRITE_TORN = 2,
    HOSTCALL_INVALID = 3,
};
struct CommandHeader {
    uint32_t CMD;
    uint16_t LEN;
    uint16_t RES;
    uint64_t DATA[];
};
__device__ enum HOSTCALL_SUBMIT_RESULT submit_hostcall(uint32_t CMD, uint16_t RES,const void* CMD_PTR, uint16_t LEN){
    uint64_t bytes = (sizeof(struct CommandHeader) + LEN);
    unsigned long long words = ((bytes + sizeof(uint64_t)- 1) / sizeof(uint64_t)) * sizeof(uint64_t);
    uint64_t offset = atomicAdd(&__HOSTCALL_BUFF_TOP__, words);
    if(offset + words > __HOSTCALL_BUFF_SIZE__) {
        printf("HOSTCALL_BUFF_FULL %ld %p!\n",__HOSTCALL_BUFF_SIZE__,__HOSTCALL_BUFF_PTR__);
        return HOSTCALL_BUFF_FULL;
    }
    uint64_t* hostcall_buff_ptr = __HOSTCALL_BUFF_PTR__;

    struct CommandHeader* cmd = (struct CommandHeader*)&hostcall_buff_ptr[offset];

    cmd->RES = RES;
    cmd->LEN = LEN;
    memcpy(cmd->DATA, CMD_PTR, LEN);
    cmd->CMD = CMD;
    if (hostcall_buff_ptr != __HOSTCALL_BUFF_PTR__)return HOSTCALL_WRITE_TORN;
    return HOSTCALL_OK;
}
struct OpenCommand{
    const char* file_path;
    size_t file_path_len;
    int flags;
    int mode;
};
struct WriteCommand{
    const char* data;
    int length;
    int fd;
};
__device__ int32_t write(int fd, const char* data, int length){
    uint16_t res = acquire_return_slot();
    struct WriteCommand cmd;
    cmd.data = data;
    cmd.length = length;
    cmd.fd = fd;
    submit_hostcall(__HOSTCALL__write, res,&cmd,sizeof(cmd));
    while(pool_return_slot(res) == 0){
        printf("Waiting for host to write to %d in __HOSTCALL__write\n",res);
        __nanosleep(10);
    }
    uint32_t resval = pool_return_slot(res);
    free_return_slot(res);
    return resval;
}
__device__ int32_t open(const char* file_path, size_t file_path_len, int mode, int flags){
    uint16_t res = acquire_return_slot();
    struct OpenCommand cmd;
    cmd.file_path_len = file_path_len;
    cmd.file_path = file_path;
    cmd.flags = flags;
    cmd.mode = mode;
    submit_hostcall(__HOSTCALL__open, res,&cmd,sizeof(cmd));
    while(pool_return_slot(res) == 0){
        printf("Waiting for host to write to %d in __HOSTCALL__open\n",res);
        __nanosleep(10);
    }
    uint32_t resval = pool_return_slot(res);
    free_return_slot(res);
    return resval;
}
__device__ void* device_malloc(size_t size){
    uint16_t res = acquire_return_slot();
    uint64_t cmd = size;
    submit_hostcall(__HOSTCALL__device_malloc, res,&cmd,sizeof(cmd));
    while(pool_return_slot(res) == 0){
        printf("Waiting for host to write to %d in __HOSTCALL__device_malloc\n",res);
        __nanosleep(10);
    }
    uint64_t resval = pool_return_slot(res);
    free_return_slot(res);
    return (void*)resval;
}
extern "C" __global__ void kernel_main() {  
   int32_t fd = open("file_from_gpu.txt",strlen("file_from_gpu.txt"), 0, 2);
   uint64_t start = clock64();
   char* data = (char*)device_malloc(1000);
   uint64_t end = clock64();
   printf("allocated %p in %lu\n",data, end - start);
   start = clock64();
   char* data2 = (char*)malloc(1000);
   end = clock64();
    printf("allocated %p from CUDA fixed heap in %lu\n",data, end - start);
   memcpy(data,"GPU says hello world :)!", strlen("GPU says hello world :)!") + 1);
   write(fd,data, strlen("GPU says hello world :)!"));
}
