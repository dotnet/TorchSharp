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

//TODO: implement more function
