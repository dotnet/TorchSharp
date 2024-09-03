// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSAmp.h"

#include <iostream>
#include <fstream>

/*void THSAmp_amp_foreach_non_finite_check_and_unscale_(const at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale)
{
    torch::_amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale);
}*/

void THSAmp_amp_foreach_non_finite_check_and_unscale_(Tensor* self, const int64_t tLength, at::Tensor& found_inf, const at::Tensor& inv_scale)
{
    torch::_amp_foreach_non_finite_check_and_unscale_(toTensors<at::Tensor>((torch::Tensor**)self, tLength),found_inf,inv_scale);
    
}

/*void THSAmp_amp_update_scale_(Tensor* self, const int64_t tLength, __resharper_unknown_type& found_inf, const __resharper_unknown_type& inv_scale)
{
    torch::_amp_update_scale()
}*/


bool THSAmp_is_torch_function_mode_enabled()
{
    return at::impl::torch_function_mode_enabled(); //https://github.com/pytorch/pytorch/blob/2c91e13afc6edcfe0a0e6189a88aae4ecbbf3516/torch/csrc/autograd/init.cpp#L911
}

bool THSAmp_is_autocast_cache_enabled()
{
    return at::autocast::is_autocast_cache_enabled();
}

bool THSAmp_is_autocast_cpu_enabled()
{
    return at::autocast::is_cpu_enabled();  //https://github.com/pytorch/pytorch/blob/2c91e13afc6edcfe0a0e6189a88aae4ecbbf3516/torch/csrc/autograd/init.cpp#L523
}

bool THSAmp_is_autocast_gpu_enabled()
{
    return at::autocast::is_enabled(); //https://github.com/pytorch/pytorch/blob/2c91e13afc6edcfe0a0e6189a88aae4ecbbf3516/torch/amp/autocast_mode.py#L363
}
bool THSAmp_is_autocast_xpu_enabled()
{
    return at::autocast::is_xpu_enabled();
}
bool THSAmp_is_autocast_hpu_enabled()
{
    return at::autocast::is_hpu_enabled();
}

#if (TORCH_VERSION_MAJOR ==2 && TORCH_VERSION_MINOR > 0)
bool THSAmp_is_autocast_ipu_enabled()
{
    return at::autocast::is_ipu_enabled();
}

bool THSAmp_is_autocast_xla_enabled()
{
    return at::autocast::is_xla_enabled();
}

#endif

int8_t THSAmp_get_autocast_cpu_dtype()
{
    return (int8_t)at::autocast::get_autocast_cpu_dtype();
}

int8_t THSAmp_get_autocast_gpu_dtype()
{
    //TODO: Implement AUTOCAST AMP AND GRADSCALER

    //INFO: Enter/Exit function of autocast_mode not need to do in C/C++ only in C# with Disposable can handle all of that function (if exists)
    //https://github.com/pytorch/pytorch/blob/main/torch/amp/autocast_mode.py

    //https://github.com/pytorch/pytorch/blob/2c91e13afc6edcfe0a0e6189a88aae4ecbbf3516/torch/csrc/autograd/init.cpp#L629
    //https://github.com/pytorch/pytorch/blob/2c91e13afc6edcfe0a0e6189a88aae4ecbbf3516/aten/src/ATen/autocast_mode.h#L20
    return (int8_t)at::autocast::get_autocast_gpu_dtype();
}

int8_t THSAmp_get_autocast_xpu_dtype()
{
    return (int8_t)at::autocast::get_autocast_xpu_dtype();
}


int THSAmp_autocast_increment_nesting()
{
    return at::autocast::increment_nesting();
}

int THSAmp_autocast_decrement_nesting()
{
    return at::autocast::decrement_nesting();
}

void THSAmp_set_autocast_enabled(bool enabled)
{
    at::autocast::set_enabled(enabled);
}

void THSAmp_set_autocast_cache_enabled(bool enabled)
{
    at::autocast::set_autocast_cache_enabled(enabled);
}

void THSAmp_set_autocast_cpu_dtype(int8_t dtype)
{
    at::autocast::set_autocast_cpu_dtype((c10::ScalarType)dtype);
}

void THSAmp_set_autocast_gpu_dtype(int8_t dtype)
{
    at::autocast::set_autocast_gpu_dtype((c10::ScalarType)dtype);
}

void THSAmp_set_autocast_xpu_dtype(int8_t dtype)
{
    at::autocast::set_autocast_xpu_dtype((c10::ScalarType)dtype);
}

void THSAmp_clear_autocast_cache()
{
    at::autocast::clear_cache();
}