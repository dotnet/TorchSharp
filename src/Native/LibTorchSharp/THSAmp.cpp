// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSAmp.h"

#include <iostream>
#include <fstream>
#include "torch/torch.h"
#include "torch/cuda.h"

/*void THSAmp_amp_foreach_non_finite_check_and_unscale_(const at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale)
{
    torch::_amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale);
}*/

void THSAmp_amp_foreach_non_finite_check_and_unscale_(Tensor* self, const int64_t tLength, at::Tensor& found_inf, const at::Tensor& inv_scale)
{
    torch::_amp_foreach_non_finite_check_and_unscale_(toTensors<at::Tensor>((torch::Tensor**)self, tLength),found_inf,inv_scale);
}

Tensor THSAmp_amp_update_scale_(at::Tensor& self, at::Tensor& growth_tracker, const at::Tensor& found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
    CATCH_TENSOR(torch::_amp_update_scale_(self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);)
}
Tensor THSAmp_amp_update_scale_out(at::Tensor& out, const at::Tensor& self, at::Tensor& growth_tracker, const at::Tensor& found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval){
    CATCH_TENSOR(torch::_amp_update_scale_out(out, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);)
}
Tensor THSAmp_amp_update_scale_outf(const at::Tensor& self, at::Tensor& growth_tracker, const at::Tensor& found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval, at::Tensor& out){
    CATCH_TENSOR(torch::_amp_update_scale_outf(self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval, out);)
}

Tensor THSAMP_amp_update_scale(const at::Tensor& self, const at::Tensor& growth_tracker, const at::Tensor& found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval, Tensor* sec)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = torch::_amp_update_scale(self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);)
    *sec = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

bool THSAmp_is_torch_function_mode_enabled()
{
    return at::impl::torch_function_mode_enabled(); //https://github.com/pytorch/pytorch/blob/2c91e13afc6edcfe0a0e6189a88aae4ecbbf3516/torch/csrc/autograd/init.cpp#L911
}

bool THSAmp_is_autocast_cache_enabled()
{
    return at::autocast::is_autocast_cache_enabled();
}

bool THSAmp_is_autocast_available(int8_t device)
{
    return at::autocast::is_autocast_available((c10::DeviceType)device);
}


bool THSAmp_is_autocast_enabled(int8_t device)
{
    return at::autocast::is_autocast_enabled((at::DeviceType)device);
}

int8_t THSAmp_get_autocast_dtype(int8_t device)
{
    return (int8_t)at::autocast::get_autocast_dtype((at::DeviceType)device);
}

void THSAmp_set_autocast_dtype(int8_t device, int8_t dtype)
{
    at::autocast::set_autocast_dtype((at::DeviceType)device, (at::ScalarType)dtype);
}

void THSAmp_set_autocast_enabled(int8_t device, bool enabled)
{
    at::autocast::set_autocast_enabled((at::DeviceType)device, enabled);
}
int THSAmp_autocast_increment_nesting()
{
    return at::autocast::increment_nesting();
}

int THSAmp_autocast_decrement_nesting()
{
    return at::autocast::decrement_nesting();
}

void THSAmp_clear_autocast_cache()
{
    at::autocast::clear_cache();
}
void THSAmp_set_autocast_cache_enabled(bool enabled)
{
    at::autocast::set_autocast_cache_enabled(enabled);
}