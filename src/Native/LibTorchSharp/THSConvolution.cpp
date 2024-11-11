// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSNN.h"

#include <torch/nn/init.h>

template<typename T>
void ApplyPaddingMode(T& opts, const int64_t padding)
{
    if (padding == 0)
        opts = opts.padding_mode(torch::kZeros);
    if (padding == 1)
        opts = opts.padding_mode(torch::kReflect);
    if (padding == 2)
        opts = opts.padding_mode(torch::kReplicate);
    if (padding == 3)
        opts = opts.padding_mode(torch::kCircular);
}

NNModule THSNN_Conv1d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    torch::nn::Conv1dOptions::padding_t padd(padding);
    if (padding == -1)
    {
        padd = torch::kSame;
    }

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv1dOptions(inputChannel, outputChannel, kernelSize)
            .stride(stride)
            .padding(padd)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);

        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Conv1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv1d>()->forward(*tensor));
}

Tensor THSNN_Conv1d_bias(const NNModule module)
{
    return get_bias<torch::nn::Conv1d>(module);
}

void THSNN_Conv1d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Conv1d>(module, bias);
}

Tensor THSNN_Conv1d_weight(const NNModule module)
{
    return get_weight<torch::nn::Conv1d>(module);
}

void THSNN_Conv1d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Conv1d>(module, weight);
}

NNModule THSNN_Conv2d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    torch::nn::Conv2dOptions::padding_t padd =
        (padding == -1) ? torch::kSame :
        (padding == 0) ? torch::kValid :
        torch::nn::Conv2dOptions::padding_t(padding);

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv2dOptions(inputChannel, outputChannel, kernelSize)
            .stride(stride)
            .padding(padd)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv2dImpl>(opts, outAsAnyModule);
    );
}

NNModule THSNN_Conv2d_ctor_1(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelX, const int64_t kernelY,
    const int64_t strideX, const int64_t strideY,
    const int64_t paddingX, const int64_t paddingY,
    const int64_t dilationX, const int64_t dilationY,
    const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    torch::nn::Conv2dOptions::padding_t padd =
        (paddingX == -1) && (paddingY == 0) ? torch::kSame :
        (paddingX == 0) && (paddingY == 0) ? torch::kValid :
        (torch::nn::Conv2dOptions::padding_t)torch::ExpandingArray<2>({ paddingX, paddingY });

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv2dOptions(inputChannel, outputChannel, { kernelX, kernelY })
        .stride({ strideX, strideY })
        .padding(padd)
        .dilation({ dilationX, dilationY })
        .groups(groups)
        .bias(bias);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Conv2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv2d>()->forward(*tensor));
}

Tensor THSNN_Conv2d_bias(const NNModule module)
{
    return get_bias<torch::nn::Conv2d>(module);
}

void THSNN_Conv2d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Conv2d>(module, bias);
}

Tensor THSNN_Conv2d_weight(const NNModule module)
{
    return get_weight<torch::nn::Conv2d>(module);
}

void THSNN_Conv2d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Conv2d>(module, weight);
}

NNModule THSNN_Conv3d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    torch::nn::Conv3dOptions::padding_t padd =
        (padding == -1) ? torch::kSame :
        (padding == 0) ? torch::kValid :
        torch::nn::Conv3dOptions::padding_t(padding);

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv3dOptions(inputChannel, outputChannel, kernelSize)
            .stride(stride)
            .padding(padd)
            .dilation(dilation)
            .groups(groups)
            .bias(bias);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv3dImpl>(opts, outAsAnyModule);
    );
}

NNModule THSNN_Conv3d_ctor_1(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelX, const int64_t kernelY, const int64_t kernelZ,
    const int64_t strideX, const int64_t strideY, const int64_t strideZ,
    const int64_t paddingX, const int64_t paddingY, const int64_t paddingZ,
    const int64_t dilationX, const int64_t dilationY, const int64_t dilationZ,
    const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    torch::nn::Conv3dOptions::padding_t padd =
        (paddingX == -1) && (paddingY == 0) && (paddingZ == 0) ? torch::kSame :
        (paddingX == 0) && (paddingY == 0) && (paddingZ == 0) ? torch::kValid :
        (torch::nn::Conv3dOptions::padding_t)torch::ExpandingArray<3>({ paddingX, paddingY, paddingZ });

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::Conv3dOptions(inputChannel, outputChannel, { kernelX, kernelY, kernelZ })
        .stride({ strideX, strideY, strideZ })
        .padding(padd)
        .dilation({ dilationX, dilationY, dilationZ })
        .groups(groups)
        .bias(bias);
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::Conv3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_Conv3d_bias(const NNModule module)
{
    return get_bias<torch::nn::Conv3d>(module);
}

void THSNN_Conv3d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::Conv3d>(module, bias);
}

Tensor THSNN_Conv3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::Conv3d>()->forward(*tensor));
}

Tensor THSNN_Conv3d_weight(const NNModule module)
{
    return get_weight<torch::nn::Conv3d>(module);
}

void THSNN_Conv3d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::Conv3d>(module, weight);
}


NNModule THSNN_ConvTranspose1d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConvTranspose1dOptions(inputChannel, outputChannel, kernelSize)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias)
        .output_padding(output_padding);
    ApplyPaddingMode(opts, paddingMode);

    res = create_module<torch::nn::ConvTranspose1dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ConvTranspose1d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConvTranspose1d>()->forward(*tensor));
}

Tensor THSNN_ConvTranspose1d_bias(const NNModule module)
{
    return get_bias<torch::nn::ConvTranspose1d>(module);
}

void THSNN_ConvTranspose1d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::ConvTranspose1d>(module, bias);
}

Tensor THSNN_ConvTranspose1d_weight(const NNModule module)
{
    return get_weight<torch::nn::ConvTranspose1d>(module);
}

void THSNN_ConvTranspose1d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::ConvTranspose1d>(module, weight);
}

NNModule THSNN_ConvTranspose2d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConvTranspose2dOptions(inputChannel, outputChannel, kernelSize)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias)
        .output_padding(output_padding);
    ApplyPaddingMode(opts, paddingMode);

    res = create_module<torch::nn::ConvTranspose2dImpl>(opts, outAsAnyModule);
    );
}

NNModule THSNN_ConvTranspose2d_ctor_1(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelX, const int64_t kernelY,
    const int64_t strideX, const int64_t strideY,
    const int64_t paddingX, const int64_t paddingY,
    const int64_t output_paddingX, const int64_t output_paddingY,
    const int64_t dilationX, const int64_t dilationY,
    const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    auto padd = torch::ExpandingArray<2>({ paddingX, paddingY });

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConvTranspose2dOptions(inputChannel, outputChannel, { kernelX, kernelY })
        .stride({ strideX, strideY })
        .padding(padd)
        .dilation({ dilationX, dilationY })
        .groups(groups)
        .bias(bias)
        .output_padding({ output_paddingX, output_paddingY });
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::ConvTranspose2dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ConvTranspose2d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConvTranspose2d>()->forward(*tensor));
}

Tensor THSNN_ConvTranspose2d_bias(const NNModule module)
{
    return get_bias<torch::nn::ConvTranspose2d>(module);
}

void THSNN_ConvTranspose2d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::ConvTranspose2d>(module, bias);
}

Tensor THSNN_ConvTranspose2d_weight(const NNModule module)
{
    return get_weight<torch::nn::ConvTranspose2d>(module);
}

void THSNN_ConvTranspose2d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::ConvTranspose2d>(module, weight);
}

NNModule THSNN_ConvTranspose3d_ctor(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelSize, const int64_t stride, const int64_t padding, const int64_t output_padding,
    const int64_t dilation, const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConvTranspose3dOptions(inputChannel, outputChannel, kernelSize)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias)
        .output_padding(output_padding);
    ApplyPaddingMode(opts, paddingMode);

    res = create_module<torch::nn::ConvTranspose3dImpl>(opts, outAsAnyModule);
    );
}

NNModule THSNN_ConvTranspose3d_ctor_1(const int64_t inputChannel, const int64_t outputChannel,
    const int64_t kernelX, const int64_t kernelY, const int64_t kernelZ,
    const int64_t strideX, const int64_t strideY, const int64_t strideZ,
    const int64_t paddingX, const int64_t paddingY, const int64_t paddingZ,
    const int64_t output_paddingX, const int64_t output_paddingY, const int64_t output_paddingZ,
    const int64_t dilationX, const int64_t dilationY, const int64_t dilationZ,
    const int64_t paddingMode, const int64_t groups, const bool bias,
    NNAnyModule* outAsAnyModule)
{
    auto padd = torch::ExpandingArray<3>({ paddingX, paddingY, paddingZ });

    CATCH_RETURN_NNModule(
        auto opts = torch::nn::ConvTranspose3dOptions(inputChannel, outputChannel, { kernelX, kernelY, kernelZ })
        .stride({ strideX, strideY, strideZ })
        .padding(padd)
        .dilation({ dilationX, dilationY, dilationZ })
        .groups(groups)
        .bias(bias)
        .output_padding({output_paddingX, output_paddingY, output_paddingZ});
        ApplyPaddingMode(opts, paddingMode);

        res = create_module<torch::nn::ConvTranspose3dImpl>(opts, outAsAnyModule);
    );
}

Tensor THSNN_ConvTranspose3d_bias(const NNModule module)
{
    return get_bias<torch::nn::ConvTranspose3d>(module);
}

void THSNN_ConvTranspose3d_set_bias(const NNModule module, const Tensor bias)
{
    set_bias<torch::nn::ConvTranspose3d>(module, bias);
}

Tensor THSNN_ConvTranspose3d_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<torch::nn::ConvTranspose3d>()->forward(*tensor));
}

Tensor THSNN_ConvTranspose3d_weight(const NNModule module)
{
    return get_weight<torch::nn::ConvTranspose3d>(module);
}

void THSNN_ConvTranspose3d_set_weight(const NNModule module, const Tensor weight)
{
    set_weight<torch::nn::ConvTranspose3d>(module, weight);
}
