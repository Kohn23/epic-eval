//
// Created by Shujian Qian on 2023-11-20.
//
#include "util_gpu_error_check.cuh"
#include "util_math.h"


namespace epic {

void* allocateDeviceMemory(size_t size)
{
    void *ptr = nullptr;
    gpu_err_check(cudaMalloc(&ptr, size));
    gpu_err_check(cudaMemset(ptr, 0, size));
    return ptr;
}

void* freeDeviceMemory(void* ptr)
{
    gpu_err_check(cudaFree(ptr));
    return nullptr;
}

void *allocatePinnedMemory(size_t size)
{
    void *retval = nullptr;
    gpu_err_check(cudaMallocHost(&retval, size, cudaHostAllocDefault));
    return retval;
}

void freePinedMemory(void *ptr)
{
    gpu_err_check(cudaFreeHost(ptr));
}

} // namespace epic