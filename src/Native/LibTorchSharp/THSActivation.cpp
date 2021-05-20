// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSNN.h"

#include <torch/nn/init.h>

NNModule THSNN_ELU_ctor(const double alpha, const bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ELUOptions().alpha(alpha).inplace(inplace);
    res = create_module<torch::nn::ELUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ELU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ELU>()->forward(*tensor));
}

NNModule THSNN_CELU_ctor(const double alpha, const bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::CELUOptions().alpha(alpha).inplace(inplace);
    res = create_module<torch::nn::CELUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_CELU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::CELU>()->forward(*tensor));
}

NNModule THSNN_ReLU_ctor(bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReLUOptions(inplace);
    res = create_module<torch::nn::ReLUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ReLU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReLU>()->forward(*tensor));
}

NNModule THSNN_RReLU_ctor(const double lower, const double upper, const bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::RReLUOptions().lower(lower).upper(upper).inplace(inplace);
    res = create_module<torch::nn::RReLUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_RReLU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::RReLU>()->forward(*tensor));
}

NNModule THSNN_ReLU6_ctor(bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ReLU6Options(inplace);
    res = create_module<torch::nn::ReLU6Impl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ReLU6_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ReLU6>()->forward(*tensor));
}

NNModule THSNN_LeakyReLU_ctor(const double negative_sloope, const bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LeakyReLUOptions().negative_slope(negative_sloope).inplace(inplace);
    res = create_module<torch::nn::LeakyReLUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LeakyReLU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::LeakyReLU>()->forward(*tensor));
}

NNModule THSNN_SELU_ctor(bool inplace, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::SELUOptions(inplace);
    res = create_module<torch::nn::SELUImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_SELU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::SELU>()->forward(*tensor));
}

NNModule THSNN_Tanh_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::TanhImpl>(outAsAnyModule);
    );
}

Tensor THSNN_Tanh_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Tanh>()->forward(*tensor));
}

NNModule THSNN_Sigmoid_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::SigmoidImpl>(outAsAnyModule);
    );
}

Tensor THSNN_Sigmoid_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Sigmoid>()->forward(*tensor));
}

NNModule THSNN_Softmax2d_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::Softmax2dImpl>(outAsAnyModule);
    );
}

Tensor THSNN_Softmax2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Softmax2d>()->forward(*tensor));
}

NNModule THSNN_GELU_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::GELUImpl>(outAsAnyModule);
    );
}

Tensor THSNN_GELU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::GELU>()->forward(*tensor));
}

NNModule THSNN_SiLU_ctor(NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        res = create_module<torch::nn::SiLUImpl>(outAsAnyModule);
    );
}

Tensor THSNN_SiLU_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::SiLU>()->forward(*tensor));
}

NNModule THSNN_Softmax_ctor(const int64_t dim, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::SoftmaxOptions(dim);
    res = create_module<torch::nn::SoftmaxImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Softmax_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Softmax>()->forward(*tensor));
}

NNModule THSNN_Softmin_ctor(const int64_t dim, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::SoftminOptions(dim);
    res = create_module<torch::nn::SoftminImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Softmin_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Softmin>()->forward(*tensor));
}

NNModule THSNN_LogSoftmax_ctor(int64_t dim, NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::LogSoftmaxOptions(dim);
    res = create_module<torch::nn::LogSoftmaxImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_LogSoftmax_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::LogSoftmax>()->forward(*tensor));
}