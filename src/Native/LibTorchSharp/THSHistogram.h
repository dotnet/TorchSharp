// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

EXPORT_API(void) THSHistogram_histogram(const Tensor input, const int64_t bins, const double* ranges, int64_t length, Tensor weight, bool density, Tensor hist, Tensor hist_bins);
EXPORT_API(void) THSHistogram_histogram_tensor(const Tensor input, const Tensor bins, Tensor weight, bool density, Tensor hist, Tensor hist_bins);
EXPORT_API(void) THSHistogram_histogramdd(const Tensor input, const int64_t* bins, int64_t length, const double* ranges, int64_t length_ranges, Tensor weight, bool density, Tensor hist, Tensor* (*allocator)(size_t length_loc));
EXPORT_API(void) THSHistogram_histogramdd_intbins(const Tensor input, int64_t bins, const double* ranges, int64_t length_ranges, Tensor weight, bool density, Tensor hist, Tensor* (*allocator)(size_t length_loc));
EXPORT_API(void) THSHistogram_histogramdd_tensors(const Tensor input, Tensor* tensors, int64_t length, const double* ranges, int64_t length_ranges, Tensor weight, bool density, Tensor hist, Tensor* (*allocator)(size_t length_loc));