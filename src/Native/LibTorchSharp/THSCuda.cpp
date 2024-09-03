// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSCuda.h"

#include <iostream>
#include <fstream>


cudaDeviceProp THSCuda_get_device_prop()
{
    int device = 0;
    cudaDeviceProp cdp;
    //cudaGetDeviceProperties_v2(&cdp, device);
    cudaGetDeviceProperties(&cdp, device);
    return cdp;
}

int THSCuda_get_major_compute_capability()
{
    return THSCuda_get_device_prop().major;
}

int THSCuda_get_minor_compute_capability()
{
    return THSCuda_get_device_prop().minor;
}
