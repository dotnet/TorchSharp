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
