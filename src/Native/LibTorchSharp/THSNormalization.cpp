// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSNN.h"

#include <torch/nn/init.h>

NNModule THSNN_BatchNorm1d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::BatchNorm1dOptions(features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);

        res = create_module<torch::nn::BatchNorm1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_BatchNorm1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::BatchNorm1d>()->forward(*tensor));
}

NNModule THSNN_BatchNorm2d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::BatchNorm2dOptions(features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);

        res = create_module<torch::nn::BatchNorm2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_BatchNorm2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::BatchNorm2d>()->forward(*tensor));
}

NNModule THSNN_BatchNorm3d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::BatchNorm3dOptions(features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);

        res = create_module<torch::nn::BatchNorm3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_BatchNorm3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::BatchNorm3d>()->forward(*tensor));
}

NNModule THSNN_GroupNorm_ctor(const int64_t num_groups, const int64_t num_channels, const double eps, const bool affine, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::GroupNormOptions(num_groups, num_channels).eps(eps).affine(affine);
        res = create_module<torch::nn::GroupNormImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_GroupNorm_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::GroupNorm>()->forward(*tensor));
}

Tensor THSNN_GroupNorm_bias(const NNModule module)
{
    return get_bias<torch::nn::GroupNorm>(module);
}

void THSNN_GroupNorm_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::GroupNorm>(module, bias);
}

Tensor THSNN_GroupNorm_weight(const NNModule module)
{
    return get_weight<torch::nn::GroupNorm>(module);
}

void THSNN_GroupNorm_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::GroupNorm>(module, weight);
}


NNModule THSNN_InstanceNorm1d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::InstanceNorm1dOptions(features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);

        res = create_module<torch::nn::InstanceNorm1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_InstanceNorm1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::InstanceNorm1d>()->forward(*tensor));
}

NNModule THSNN_InstanceNorm2d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::InstanceNorm2dOptions(features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);

        res = create_module<torch::nn::InstanceNorm2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_InstanceNorm2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::InstanceNorm2d>()->forward(*tensor));
}

NNModule THSNN_InstanceNorm3d_ctor(const int64_t features, const double eps, const double momentum, const bool affine, const bool track_running_stats, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::InstanceNorm3dOptions(features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);

        res = create_module<torch::nn::InstanceNorm3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_InstanceNorm3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::InstanceNorm3d>()->forward(*tensor));
}

NNModule THSNN_LayerNorm_ctor(const int64_t* norm_shape, const int64_t norm_shape_len, const double eps, const bool elementwise_affine, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        std::vector<int64_t> normalized_shape;
        for (int64_t i = 0; i < norm_shape_len; ++i)
        {
            normalized_shape.push_back(norm_shape[i]);
        }
        auto opts = torch::nn::LayerNormOptions(normalized_shape).eps(eps).elementwise_affine(elementwise_affine);
        res = create_module<torch::nn::LayerNormImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LayerNorm_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::LayerNorm>()->forward(*tensor));
}

Tensor THSNN_LayerNorm_bias(const NNModule module)
{
    return get_bias<torch::nn::LayerNorm>(module);
}

void THSNN_LayerNorm_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::LayerNorm>(module, bias);
}

Tensor THSNN_LayerNorm_weight(const NNModule module)
{
    return get_weight<torch::nn::LayerNorm>(module);
}

void THSNN_LayerNorm_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::LayerNorm>(module, weight);
}


NNModule THSNN_LocalResponseNorm_ctor(const int64_t size, const double alpha, const double beta, const double k, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LocalResponseNormOptions(size)
        .alpha(alpha)
        .beta(beta)
        .k(k);
        res = create_module<torch::nn::LocalResponseNormImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LocalResponseNorm_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::LocalResponseNorm>()->forward(*tensor));
}

void THSNN_BatchNorm1d_reset_stats(const NNModule module)
{
    CATCH((*module)->as<torch::nn::BatchNorm1d>()->reset_running_stats(););
}

Tensor THSNN_BatchNorm1d_get_mean(const NNModule module)
{
    CATCH(
        auto m = (*module)->as<torch::nn::BatchNorm1d>()->running_mean;
        return m.defined() ? ResultTensor(m) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_BatchNorm1d_get_var(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::BatchNorm1d>()->running_var;
        return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_BatchNorm1d_get_batches(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::BatchNorm1d>()->num_batches_tracked;
        return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

void THSNN_BatchNorm1d_set_mean(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::BatchNorm1d>()->running_mean = *bias;
    );
}

void THSNN_BatchNorm1d_set_var(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::BatchNorm1d>()->running_var = *bias;
    );
}

Tensor THSNN_BatchNorm1d_bias(const NNModule module)
{
    return get_bias<torch::nn::BatchNorm1d>(module);
}

void THSNN_BatchNorm1d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::BatchNorm1d>(module, bias);
}

Tensor THSNN_BatchNorm1d_weight(const NNModule module)
{
    return get_weight<torch::nn::BatchNorm1d>(module);
}

void THSNN_BatchNorm1d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::BatchNorm1d>(module, weight);
}

void THSNN_BatchNorm2d_reset_stats(const NNModule module)
{
    CATCH((*module)->as<torch::nn::BatchNorm2d>()->reset_running_stats(););
}

Tensor THSNN_BatchNorm2d_get_mean(const NNModule module)
{
    CATCH(
        auto m = (*module)->as<torch::nn::BatchNorm2d>()->running_mean;
        return m.defined() ? ResultTensor(m) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_BatchNorm2d_get_var(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::BatchNorm2d>()->running_var;
        return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_BatchNorm2d_get_batches(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::BatchNorm2d>()->num_batches_tracked;
        return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

void THSNN_BatchNorm2d_set_mean(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::BatchNorm2d>()->running_mean = *bias;
    );
}

void THSNN_BatchNorm2d_set_var(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::BatchNorm2d>()->running_var = *bias;
    );
}

Tensor THSNN_BatchNorm2d_bias(const NNModule module)
{
    return get_bias<torch::nn::BatchNorm2d>(module);
}

void THSNN_BatchNorm2d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::BatchNorm2d>(module, bias);
}

Tensor THSNN_BatchNorm2d_weight(const NNModule module)
{
    return get_weight<torch::nn::BatchNorm2d>(module);
}

void THSNN_BatchNorm2d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::BatchNorm2d>(module, weight);
}

void THSNN_BatchNorm3d_reset_stats(const NNModule module)
{
    CATCH((*module)->as<torch::nn::BatchNorm3d>()->reset_running_stats(););
}

Tensor THSNN_BatchNorm3d_get_mean(const NNModule module)
{
    CATCH(
        auto m = (*module)->as<torch::nn::BatchNorm3d>()->running_mean;
        return m.defined() ? ResultTensor(m) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_BatchNorm3d_get_var(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::BatchNorm3d>()->running_var;
        return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_BatchNorm3d_get_batches(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::BatchNorm3d>()->num_batches_tracked;
        return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

void THSNN_BatchNorm3d_set_mean(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::BatchNorm3d>()->running_mean = *bias;
    );
}

void THSNN_BatchNorm3d_set_var(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::BatchNorm3d>()->running_var = *bias;
    );
}

Tensor THSNN_BatchNorm3d_bias(const NNModule module)
{
    return get_bias<torch::nn::BatchNorm3d>(module);
}

void THSNN_BatchNorm3d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::BatchNorm3d>(module, bias);
}

Tensor THSNN_BatchNorm3d_weight(const NNModule module)
{
    return get_weight<torch::nn::BatchNorm3d>(module);
}

void THSNN_BatchNorm3d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::BatchNorm3d>(module, weight);
}

void THSNN_InstanceNorm1d_reset_stats(const NNModule module)
{
    CATCH((*module)->as<torch::nn::InstanceNorm1d>()->reset_running_stats(););
}

Tensor THSNN_InstanceNorm1d_get_mean(const NNModule module)
{
    CATCH(
        auto m = (*module)->as<torch::nn::InstanceNorm1d>()->running_mean;
    return m.defined() ? ResultTensor(m) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_InstanceNorm1d_get_var(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::InstanceNorm1d>()->running_var;
    return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_InstanceNorm1d_get_batches(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::InstanceNorm1d>()->num_batches_tracked;
    return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

void THSNN_InstanceNorm1d_set_mean(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::InstanceNorm1d>()->running_mean = *bias;
    );
}

void THSNN_InstanceNorm1d_set_var(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::InstanceNorm1d>()->running_var = *bias;
    );
}

Tensor THSNN_InstanceNorm1d_bias(const NNModule module)
{
    return get_bias<torch::nn::InstanceNorm1d>(module);
}

void THSNN_InstanceNorm1d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::InstanceNorm1d>(module, bias);
}

Tensor THSNN_InstanceNorm1d_weight(const NNModule module)
{
    return get_weight<torch::nn::InstanceNorm1d>(module);
}

void THSNN_InstanceNorm1d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::InstanceNorm1d>(module, weight);
}

void THSNN_InstanceNorm2d_reset_stats(const NNModule module)
{
    CATCH((*module)->as<torch::nn::InstanceNorm2d>()->reset_running_stats(););
}

Tensor THSNN_InstanceNorm2d_get_mean(const NNModule module)
{
    CATCH(
        auto m = (*module)->as<torch::nn::InstanceNorm2d>()->running_mean;
    return m.defined() ? ResultTensor(m) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_InstanceNorm2d_get_var(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::InstanceNorm2d>()->running_var;
    return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_InstanceNorm2d_get_batches(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::InstanceNorm2d>()->num_batches_tracked;
    return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

void THSNN_InstanceNorm2d_set_mean(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::InstanceNorm2d>()->running_mean = *bias;
    );
}

void THSNN_InstanceNorm2d_set_var(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::InstanceNorm2d>()->running_var = *bias;
    );
}

Tensor THSNN_InstanceNorm2d_bias(const NNModule module)
{
    return get_bias<torch::nn::InstanceNorm2d>(module);
}

void THSNN_InstanceNorm2d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::InstanceNorm2d>(module, bias);
}

Tensor THSNN_InstanceNorm2d_weight(const NNModule module)
{
    return get_weight<torch::nn::InstanceNorm2d>(module);
}

void THSNN_InstanceNorm2d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::InstanceNorm2d>(module, weight);
}

void THSNN_InstanceNorm3d_reset_stats(const NNModule module)
{
    CATCH((*module)->as<torch::nn::InstanceNorm3d>()->reset_running_stats(););
}

Tensor THSNN_InstanceNorm3d_get_mean(const NNModule module)
{
    CATCH(
        auto m = (*module)->as<torch::nn::InstanceNorm3d>()->running_mean;
    return m.defined() ? ResultTensor(m) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_InstanceNorm3d_get_var(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::InstanceNorm3d>()->running_var;
    return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

Tensor THSNN_InstanceNorm3d_get_batches(const NNModule module)
{
    CATCH(
        auto v = (*module)->as<torch::nn::InstanceNorm3d>()->num_batches_tracked;
    return v.defined() ? ResultTensor(v) : nullptr;
    );
    return nullptr;
}

void THSNN_InstanceNorm3d_set_mean(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::InstanceNorm3d>()->running_mean = *bias;
    );
}

void THSNN_InstanceNorm3d_set_var(const NNModule module, const Tensor bias)
{
    CATCH(
        (*module)->as<torch::nn::InstanceNorm3d>()->running_var = *bias;
    );
}

Tensor THSNN_InstanceNorm3d_bias(const NNModule module)
{
    return get_bias<torch::nn::InstanceNorm3d>(module);
}

void THSNN_InstanceNorm3d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::InstanceNorm3d>(module, bias);
}

Tensor THSNN_InstanceNorm3d_weight(const NNModule module)
{
    return get_weight<torch::nn::InstanceNorm3d>(module);
}

void THSNN_InstanceNorm3d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::InstanceNorm3d>(module, weight);
}

Tensor THSNN_batch_norm(const Tensor input, Tensor running_mean, const Tensor running_var, const Tensor weight, const Tensor bias, const bool training, const double momentum, const double eps)
{
    auto opts = torch::nn::functional::BatchNormFuncOptions()
        .training(training)
        .momentum(momentum)
        .eps(eps);
    if (weight != nullptr) opts.weight(*weight);
    if (bias != nullptr) opts.bias(*bias);
    CATCH_TENSOR(torch::nn::functional::batch_norm(*input, *running_mean, *running_var, opts));
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
