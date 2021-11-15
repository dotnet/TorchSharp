// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

// API

EXPORT_API(Tensor) THSVision_AdjustHue(const Tensor img, const double hue_factor);

EXPORT_API(Tensor) THSVision_GenerateAffineGrid(Tensor theta, const int64_t w, const int64_t h, const int64_t ow, const int64_t oh);
EXPORT_API(Tensor) THSVision_ApplyGridTransform(Tensor i, Tensor g, const int8_t m, const float* fill, const int64_t fill_length);

EXPORT_API(Tensor) THSVision_PerspectiveGrid(const float* coeffs, const int64_t coeffs_length, const int64_t ow, const int64_t oh, const int8_t scalar_type, const int device_type, const int device_index);

EXPORT_API(Tensor) THSVision_ScaleChannel(Tensor ic);

EXPORT_API(void) THSVision_ComputeOutputSize(const float* matrix, const int64_t matrix_length, const int64_t w, const int64_t h);
