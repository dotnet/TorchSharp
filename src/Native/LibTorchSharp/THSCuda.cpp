// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSCuda.h"

#include <iostream>
#include <fstream>

#ifdef CUDA_TOOLKIT_FOUND
cudaDeviceProp THSCuda_get_device_prop(int device)
{
    cudaDeviceProp cdp;
    //cudaGetDeviceProperties(&cdp, device);
    cudaGetDeviceProperties_v2(&cdp, device);
    return cdp;
}
#endif

int THSCuda_get_major_compute_capability(int device)
{
#ifdef CUDA_TOOLKIT_FOUND
    return THSCuda_get_device_prop(device).major;
#else
    return -1;
#endif
}

int THSCuda_get_minor_compute_capability(int device)
{
#ifdef CUDA_TOOLKIT_FOUND
    return THSCuda_get_device_prop(device).minor;
#else
    return -1;
#endif
}


int THSCuda_get_device_count(int* count)
{
#ifdef CUDA_TOOLKIT_FOUND
    return cudaGetDeviceCount(count);
#else
    return -1;
#endif
}

int THSCuda_get_free_total(int device, int* id, size_t* free, size_t* total)
{
#ifdef CUDA_TOOLKIT_FOUND
    cudaError_t res = cudaSetDevice(device);
    if (res != CUDA_SUCCESS)
        return -1;
    res = cudaGetDevice(id);
    if (res != CUDA_SUCCESS)
        return -1;
    return cudaMemGetInfo(free, total);
#else
    return -1;
#endif
}

size_t THSCuda_get_total_memory(int device)
{
#ifdef CUDA_TOOLKIT_FOUND
    return THSCuda_get_device_prop(device).totalConstMem;
#else
    return 0; //Is size_t (unsigned long) so cant be negative.
#endif
    //RETURN_CUDA_DEVICE(THSCuda_get_device_prop(device).totalConstMem)
}


size_t THSCuda_get_global_total_memory(int device)
{
#ifdef CUDA_TOOLKIT_FOUND
    return THSCuda_get_device_prop(device).totalGlobalMem;
#else
    return 0;
#endif  
}

const char* THSCuda_get_cuda_version()
{
#ifdef CUDA_TOOLKIT_FOUND
    int runtimeVersion;
    cudaError_t err = cudaRuntimeGetVersion(&runtimeVersion);

    if (err != cudaSuccess) {
        std::cerr << "Error getting CUDA runtime version: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    int major = runtimeVersion / 1000;
    int minor = (runtimeVersion % 1000) / 10;
    int patch = runtimeVersion % 10;

    std::string cudaVersionString = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    //std::cout << "CUDA Runtime Version: " << cudaVersionString << std::endl;
    return cudaVersionString.c_str();
#else
    return nullptr;
#endif
}


//TODO: implement more function
