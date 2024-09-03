// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"
#include "Utils.h"

//https://github.com/pytorch/pytorch/blob/main/torch/_meta_registrations.py#L5957
//EXPORT_API(void) THSAmp_amp_foreach_non_finite_check_and_unscale_(const at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale);

EXPORT_API(void) THSAmp_amp_foreach_non_finite_check_and_unscale_(Tensor* self, const int64_t tLength, at::Tensor& found_inf, const at::Tensor& inv_scale);

//EXPORT_API(void) THSAmp_amp_update_scale_(const at::Tensor& self, const at::Tensor& inv_scale);

EXPORT_API(Tensor) THSAmp_amp_update_scale_(at::Tensor& self, at::Tensor& growth_tracker, const at::Tensor& found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval);
EXPORT_API(Tensor) THSAmp_amp_update_scale_out(at::Tensor& out, const at::Tensor& self, at::Tensor& growth_tracker, const at::Tensor& found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval);
EXPORT_API(Tensor) THSAmp_amp_update_scale_outf(const at::Tensor& self, at::Tensor& growth_tracker, const at::Tensor& found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval, at::Tensor& out);
EXPORT_API(Tensor) THSAMP_amp_update_scale(const at::Tensor& self, const at::Tensor& growth_tracker, const at::Tensor& found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval, Tensor* sec);
   
EXPORT_API(bool) THSAmp_is_torch_function_mode_enabled();

//Maybe the best work is call THSTorch_is_autocast_enabled(enum of devices c# as int8_t);
EXPORT_API(bool) THSAmp_is_autocast_cache_enabled();
EXPORT_API(bool) THSAmp_is_autocast_cpu_enabled();
EXPORT_API(bool) THSAmp_is_autocast_gpu_enabled();
EXPORT_API(bool) THSAmp_is_autocast_xpu_enabled();
EXPORT_API(bool) THSAmp_is_autocast_hpu_enabled();

#if (TORCH_VERSION_MAJOR ==2 && TORCH_VERSION_MINOR > 0)
EXPORT_API(bool) THSAmp_is_autocast_ipu_enabled();
EXPORT_API(bool) THSAmp_is_autocast_xla_enabled();
#endif

EXPORT_API(int8_t) THSAmp_get_autocast_cpu_dtype();
EXPORT_API(int8_t) THSAmp_get_autocast_gpu_dtype();
EXPORT_API(int8_t) THSAmp_get_autocast_xpu_dtype();

EXPORT_API(int) THSAmp_autocast_increment_nesting();
EXPORT_API(int) THSAmp_autocast_decrement_nesting();

EXPORT_API(void) THSAmp_set_autocast_enabled(bool enabled);
EXPORT_API(void) THSAmp_set_autocast_cache_enabled(bool enabled);
EXPORT_API(void) THSAmp_set_autocast_cpu_dtype(int8_t dtype);
EXPORT_API(void) THSAmp_set_autocast_gpu_dtype(int8_t dtype);
EXPORT_API(void) THSAmp_set_autocast_xpu_dtype(int8_t dtype);

EXPORT_API(void) THSAmp_clear_autocast_cache();

//EXPORT_API(bool) THSTorch_jit_is_scripting();