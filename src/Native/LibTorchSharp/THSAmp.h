// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

//https://github.com/pytorch/pytorch/blob/main/torch/_meta_registrations.py#L5957
//EXPORT_API(void) THSAmp_amp_foreach_non_finite_check_and_unscale_(const at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale);

EXPORT_API(void) THSAmp_amp_foreach_non_finite_check_and_unscale_(Tensor* self, const int64_t tLength, at::Tensor& found_inf, const at::Tensor& inv_scale);
