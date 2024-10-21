// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"
#ifdef TORCHSHARP_CUDA_TOOLKIT_FOUND
#include "cuda.h"
#include "cuda_runtime_api.h"

cudaDeviceProp THSCuda_get_device_prop(int device=0);

inline int show_available_memory()
{
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount(&num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        cudaSetDevice(gpu_id);
        int id;
        cudaGetDevice(&id);
        cudaMemGetInfo(&free, &total);
        std::cout << "GPU " << id << " memory: free=" << free << ", total=" << total << std::endl;
    }
    return 0;
}
#endif

EXPORT_API(int) THSCuda_get_major_compute_capability(int device);
EXPORT_API(int) THSCuda_get_minor_compute_capability(int device);
EXPORT_API(int) THSCuda_get_device_count(int* count);
EXPORT_API(int) THSCuda_get_free_total(int device, int* id, size_t* free, size_t* total);
EXPORT_API(size_t) THSCuda_get_total_memory(int device);
EXPORT_API(size_t) THSCuda_get_global_total_memory(int device);