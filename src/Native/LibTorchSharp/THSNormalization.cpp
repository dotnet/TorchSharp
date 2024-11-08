// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSNN.h"

#include <torch/nn/init.h>

Tensor THSNN_batch_norm(const Tensor input, Tensor running_mean, const Tensor running_var, const Tensor weight, const Tensor bias, const bool training, const double momentum, const double eps)
{
    c10::optional<at::Tensor> w = weight != nullptr ? *weight : c10::optional<at::Tensor>();    
    c10::optional<at::Tensor> b = bias != nullptr ? *bias : c10::optional<at::Tensor>();
    c10::optional<at::Tensor> rm = running_mean != nullptr ? *running_mean : c10::optional<at::Tensor>();
    c10::optional<at::Tensor> rv = running_var != nullptr ? *running_var : c10::optional<at::Tensor>();

    CATCH_TENSOR(torch::batch_norm(*input, w, b, rm, rv, training, momentum, eps, false));
}

Tensor THSNN_normalize(const Tensor input, const double p, const int64_t dim, const double eps)
{
    auto opts = torch::nn::functional::NormalizeFuncOptions()
        .p(p)
        .dim(dim)
        .eps(eps);
    CATCH_TENSOR(torch::nn::functional::normalize(*input, opts));
}

Tensor THSNN_group_norm(const Tensor input, const int64_t num_groups, const Tensor weight, const Tensor bias, const double eps)
{
    auto opts = torch::nn::functional::GroupNormFuncOptions(num_groups)
        .eps(eps);
    if (weight != nullptr) opts.weight(*weight);
    if (bias != nullptr) opts.bias(*bias);
    CATCH_TENSOR(torch::nn::functional::group_norm(*input, opts));
}

Tensor THSNN_instance_norm(const Tensor input, const Tensor running_mean, const Tensor running_var, const Tensor weight, const Tensor bias, const bool use_input_stats, const double momentum, const double eps)
{
    auto opts = torch::nn::functional::InstanceNormFuncOptions()
        .use_input_stats(use_input_stats)
        .momentum(momentum)
        .eps(eps);
    if (running_mean != nullptr) opts.running_mean(*running_mean);
    if (running_var != nullptr) opts.running_var(*running_var);
    if (weight != nullptr) opts.weight(*weight);
    if (bias != nullptr) opts.bias(*bias);
    CATCH_TENSOR(torch::nn::functional::instance_norm(*input, opts));
}

Tensor THSNN_layer_norm(const Tensor input, const int64_t* normalized_shape, const int64_t normalized_shape_len, const Tensor weight, const Tensor bias, const double eps)
{
    auto opts = torch::nn::functional::LayerNormFuncOptions(
        std::vector<int64_t>(normalized_shape, normalized_shape + normalized_shape_len))
        .eps(eps);
    if (weight != nullptr) opts.weight(*weight);
    if (bias != nullptr) opts.bias(*bias);
    CATCH_TENSOR(torch::nn::functional::layer_norm(*input, opts));
}

Tensor THSNN_local_response_norm(const Tensor input, const int64_t size, const double alpha, const double beta, const double k)
{
    auto opts = torch::nn::functional::LocalResponseNormFuncOptions(size)
        .alpha(alpha)
        .beta(beta)
        .k(k);
    CATCH_TENSOR(torch::nn::functional::local_response_norm(*input, opts));
}
