// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSCuda.h"

#include <iostream>
#include <fstream>

#ifdef TORCHSHARP_CUDA_TOOLKIT_FOUND
cudaDeviceProp THSCuda_get_device_prop()
{
    int device = 0;
    cudaDeviceProp cdp;
    //cudaGetDeviceProperties(&cdp, device);
    cudaGetDeviceProperties_v2(&cdp, device);
    return cdp;
}
#endif

int THSCuda_get_major_compute_capability()
{
#ifdef TORCHSHARP_CUDA_TOOLKIT_FOUND
    return THSCuda_get_device_prop().major;
#else
    return -1;
#endif
}

int THSCuda_get_minor_compute_capability()
{
#ifdef TORCHSHARP_CUDA_TOOLKIT_FOUND
    return THSCuda_get_device_prop().minor;
#else
    return -1;
#endif
}
