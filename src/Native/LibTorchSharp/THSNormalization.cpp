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

#define BATCHNORM_STATS(type) \
void THSNN_##type##_reset_stats(const NNModule module)\
{\
    CATCH((*module)->as<torch::nn::##type##>()->reset_running_stats(););\
}\
Tensor THSNN_##type##_get_mean(const NNModule module)\
{\
    CATCH(\
        auto m = (*module)->as<torch::nn::##type##>()->running_mean;\
        return m.defined() ? ResultTensor(m) : nullptr;\
    );\
    return nullptr;\
}\
Tensor THSNN_##type##_get_var(const NNModule module)\
{\
    CATCH(\
        auto v = (*module)->as<torch::nn::##type##>()->running_var;\
        return v.defined() ? ResultTensor(v) : nullptr;\
    );\
    return nullptr;\
}\
Tensor THSNN_##type##_bias(const NNModule module)\
{\
    return get_bias<torch::nn::##type##>(module);\
}\
void THSNN_##type##_set_bias(const NNModule module, const Tensor bias)\
{\
    set_bias<torch::nn::##type##>(module, bias);\
}\
Tensor THSNN_##type##_weight(const NNModule module)\
{\
    return get_weight<torch::nn::##type##>(module);\
}\
void THSNN_##type##_set_weight(const NNModule module, const Tensor weight)\
{\
    set_weight<torch::nn::##type##>(module, weight);\
}

BATCHNORM_STATS(BatchNorm1d)
BATCHNORM_STATS(BatchNorm2d)
BATCHNORM_STATS(BatchNorm3d)