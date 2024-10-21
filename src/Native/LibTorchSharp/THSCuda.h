// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"
#ifdef TORCHSHARP_CUDA_TOOLKIT_FOUND
#include "cuda.h"
#include "cuda_runtime_api.h"

cudaDeviceProp THSCuda_get_device_prop();

#endif

EXPORT_API(int) THSCuda_get_major_compute_capability();
EXPORT_API(int) THSCuda_get_minor_compute_capability();