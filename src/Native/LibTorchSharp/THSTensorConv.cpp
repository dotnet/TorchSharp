// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSTensor.h"

#include <iostream>
#include <fstream>



Tensor THSTensor_adaptive_avg_pool1d(
    const Tensor tensor,
    const int64_t* outputSize, const int outputSizeLength)
{
    CATCH_TENSOR(torch::adaptive_avg_pool1d(
        *tensor,
        at::ArrayRef<int64_t>(outputSize, outputSizeLength)));
}

Tensor THSTensor_adaptive_avg_pool2d(
    const Tensor tensor,
    const int64_t* outputSize, const int outputSizeLength)
{
    CATCH_TENSOR(torch::adaptive_avg_pool2d(
        *tensor,
        at::ArrayRef<int64_t>(outputSize, outputSizeLength)));
}

Tensor THSTensor_adaptive_avg_pool3d(
    const Tensor tensor,
    const int64_t* outputSize, const int outputSizeLength)
{
    CATCH_TENSOR(torch::adaptive_avg_pool3d(
        *tensor,
        at::ArrayRef<int64_t>(outputSize, outputSizeLength)));
}

Tensor THSTensor_adaptive_avg_pool3d_backward_out(
    const Tensor grad_input,
    const Tensor grad_output,
    const Tensor tensor)
{
    CATCH_TENSOR(torch::adaptive_avg_pool3d_backward_out(
        *grad_input,
        *grad_output,
        *tensor));
}

Tensor THSTensor_avg_pool1d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    bool ceil_mode,
    bool count_include_pad)
{
    CATCH_TENSOR(torch::avg_pool1d(
        *tensor,
        at::ArrayRef<int64_t>(kernelSize, kernelSizeLength),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        ceil_mode,
        count_include_pad));
}

Tensor THSTensor_avg_pool2d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    bool ceil_mode,
    bool count_include_pad)
{
    CATCH_TENSOR(torch::avg_pool2d(
        *tensor,
        at::ArrayRef<int64_t>(kernelSize, kernelSizeLength),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        ceil_mode,
        count_include_pad));
}

Tensor THSTensor_avg_pool2d_backward(
    const Tensor grad_output,
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    bool ceil_mode,
    bool count_include_pad,
    const int64_t divisor_override)
{
    CATCH_TENSOR(torch::avg_pool2d_backward(
        *grad_output,
        *tensor,
        at::ArrayRef<int64_t>(kernelSize, kernelSizeLength),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        ceil_mode,
        count_include_pad,
        (divisor_override == 0 ? c10::optional<int64_t>() : c10::optional<int64_t>(divisor_override))));
}

Tensor THSTensor_avg_pool3d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    bool ceil_mode,
    bool count_include_pad)
{
    CATCH_TENSOR(torch::avg_pool3d(
        *tensor,
        at::ArrayRef<int64_t>(kernelSize, kernelSizeLength),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        ceil_mode,
        count_include_pad));
}

Tensor THSTensor_avg_pool3d_backward(
    const Tensor grad_output,
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    bool ceil_mode,
    bool count_include_pad,
    const int64_t divisor_override)
{
    CATCH_TENSOR(torch::avg_pool3d_backward(
        *grad_output,
        *tensor,
        at::ArrayRef<int64_t>(kernelSize, kernelSizeLength),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        ceil_mode,
        count_include_pad,
        (divisor_override == 0 ? c10::optional<int64_t>() : c10::optional<int64_t>(divisor_override))));
}
Tensor THSTensor_conv_transpose1d(
    const Tensor input, const Tensor weight, const Tensor bias,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* outputPadding, const int outputPaddingLength,
    const int64_t* dilation, const int dilationLength,
    int64_t groups)
{
    CATCH_TENSOR(torch::conv_transpose1d(*input, *weight, (bias ? *bias : at::Tensor()),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        at::ArrayRef<int64_t>(outputPadding, outputPaddingLength),
        groups,
        at::ArrayRef<int64_t>(dilation, dilationLength)));
}

Tensor THSTensor_conv_transpose2d(
    const Tensor input, const Tensor weight, const Tensor bias,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* outputPadding, const int outputPaddingLength,
    const int64_t* dilation, const int dilationLength,
    int64_t groups)
{
    CATCH_TENSOR(torch::conv_transpose2d(*input, *weight, (bias ? *bias : at::Tensor()),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        at::ArrayRef<int64_t>(outputPadding, outputPaddingLength),
        groups,
        at::ArrayRef<int64_t>(dilation, dilationLength)));
}

Tensor THSTensor_conv_transpose3d(
    const Tensor input, const Tensor weight, const Tensor bias,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* outputPadding, const int outputPaddingLength,
    const int64_t* dilation, const int dilationLength,
    int64_t groups)
{
    CATCH_TENSOR(torch::conv_transpose3d(*input, *weight, (bias ? *bias : at::Tensor()),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        at::ArrayRef<int64_t>(outputPadding, outputPaddingLength),
        groups,
        at::ArrayRef<int64_t>(dilation, dilationLength)));
}

Tensor THSTensor_conv1d(
    const Tensor input,
    const Tensor weight,
    const Tensor bias,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    int64_t groups)
{
    CATCH_TENSOR(torch::conv1d(*input, *weight, (bias ? *bias : at::Tensor()),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        at::ArrayRef<int64_t>(dilation, dilationLength),
        groups));
}

Tensor THSTensor_conv2d(
    const Tensor input,
    const Tensor weight,
    const Tensor bias,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    int64_t groups)
{
    CATCH_TENSOR(torch::conv2d(*input, *weight, (bias ? *bias : at::Tensor()),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        at::ArrayRef<int64_t>(dilation, dilationLength),
        groups));
}

Tensor THSTensor_conv3d(
    const Tensor input,
    const Tensor weight,
    const Tensor bias,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    int64_t groups)
{
    CATCH_TENSOR(torch::conv3d(*input, *weight, (bias ? *bias : at::Tensor()),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        at::ArrayRef<int64_t>(dilation, dilationLength),
        groups));
}


Tensor THSTensor_max_pool1d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode)
{
    CATCH_TENSOR(torch::max_pool1d(
        *tensor,
        at::ArrayRef<int64_t>(kernelSize, kernelSizeLength),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        at::ArrayRef<int64_t>(dilation, dilationLength),
        ceil_mode));
}

void THSTensor_max_pool1d_with_indices(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode)
{
    CATCH(
        auto res = torch::max_pool1d_with_indices(
            *tensor,
            at::ArrayRef<int64_t>(kernelSize, kernelSizeLength),
            at::ArrayRef<int64_t>(stride, strideLength),
            at::ArrayRef<int64_t>(padding, paddingLength),
            at::ArrayRef<int64_t>(dilation, dilationLength),
            ceil_mode);

    Tensor * result = allocator(2);
    result[0] = new torch::Tensor(std::get<0>(res));
    result[1] = new torch::Tensor(std::get<1>(res));
    )
}

Tensor THSTensor_max_pool2d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode)
{
    CATCH_TENSOR(torch::max_pool2d(
        *tensor,
        at::ArrayRef<int64_t>(kernelSize, kernelSizeLength),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        at::ArrayRef<int64_t>(dilation, dilationLength),
        ceil_mode));
}

void THSTensor_max_pool2d_with_indices(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode)
{
    CATCH(
        auto res = torch::max_pool2d_with_indices(
            *tensor,
            at::ArrayRef<int64_t>(kernelSize, kernelSizeLength),
            at::ArrayRef<int64_t>(stride, strideLength),
            at::ArrayRef<int64_t>(padding, paddingLength),
            at::ArrayRef<int64_t>(dilation, dilationLength),
            ceil_mode);
    Tensor * result = allocator(2);
    result[0] = new torch::Tensor(std::get<0>(res));
    result[1] = new torch::Tensor(std::get<1>(res));
    )
}

Tensor THSTensor_max_pool3d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode)
{
    CATCH_TENSOR(torch::max_pool3d(
        *tensor,
        at::ArrayRef<int64_t>(kernelSize, kernelSizeLength),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength),
        at::ArrayRef<int64_t>(dilation, dilationLength),
        ceil_mode));
}

void THSTensor_max_pool3d_with_indices(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode)
{
    CATCH(
        auto res = torch::max_pool3d_with_indices(
            *tensor,
            at::ArrayRef<int64_t>(kernelSize, kernelSizeLength),
            at::ArrayRef<int64_t>(stride, strideLength),
            at::ArrayRef<int64_t>(padding, paddingLength),
            at::ArrayRef<int64_t>(dilation, dilationLength),
            ceil_mode);
    Tensor * result = allocator(2);
    result[0] = new torch::Tensor(std::get<0>(res));
    result[1] = new torch::Tensor(std::get<1>(res));
    )
}

Tensor THSTensor_maxunpool2d(
    const Tensor tensor,
    const Tensor indices,
    const int64_t* outputSize, const int outputSizeLength)
{
    CATCH_TENSOR(torch::max_unpool2d(
        *tensor,
        *indices,
        at::ArrayRef<int64_t>(outputSize, outputSizeLength)));
}

Tensor THSTensor_maxunpool3d(
    const Tensor tensor,
    const Tensor indices,
    const int64_t* outputSize, const int outputSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength)
{
    CATCH_TENSOR(torch::max_unpool3d(
        *tensor,
        *indices,
        at::ArrayRef<int64_t>(outputSize, outputSizeLength),
        at::ArrayRef<int64_t>(stride, strideLength),
        at::ArrayRef<int64_t>(padding, paddingLength)));
}


Tensor THSTensor_unsqueeze(Tensor tensor, int64_t dim)
{
    CATCH_TENSOR(tensor->unsqueeze(dim))
}

Tensor THSTensor_unsqueeze_(Tensor tensor, int64_t dim)
{
    CATCH_TENSOR(tensor->unsqueeze_(dim))
}

Tensor THSTensor_upsample_nearest1d(
    const Tensor tensor,
    const int64_t* outputSize, const int outputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength)
{
    CATCH_TENSOR(torch::upsample_nearest1d(
        *tensor,
        (outputSize == 0 ? c10::optional<c10::IntArrayRef>() : c10::optional<c10::IntArrayRef>(at::IntArrayRef(outputSize, outputSizeLength))),
        (scaleFactors == 0 ? c10::optional<c10::ArrayRef<double>>() : c10::optional<c10::ArrayRef<double>>(at::ArrayRef<double>(scaleFactors, scaleFactorsLength))))
    );
}

Tensor THSTensor_upsample_nearest1d_backward(
    const Tensor grad_output,
    const int64_t* outputSize, const int outputSizeLength,
    const int64_t* inputSize, const int inputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength)
{
    CATCH_TENSOR(torch::upsample_nearest1d_backward(
        *grad_output,
        (outputSize == 0 ? c10::optional<c10::IntArrayRef>() : c10::optional<c10::IntArrayRef>(at::IntArrayRef(outputSize, outputSizeLength))),
        at::IntArrayRef(inputSize, inputSizeLength),
        (scaleFactors == 0 ? c10::optional<c10::ArrayRef<double>>() : c10::optional<c10::ArrayRef<double>>(at::ArrayRef<double>(scaleFactors, scaleFactorsLength))))
    );
}

Tensor THSTensor_upsample_nearest2d(
    const Tensor tensor,
    const int64_t* outputSize, const int outputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength)
{
    CATCH_TENSOR(torch::upsample_nearest2d(
        *tensor,
        (outputSize == 0 ? c10::optional<c10::IntArrayRef>() : c10::optional<c10::IntArrayRef>(at::IntArrayRef(outputSize, outputSizeLength))),
        (scaleFactors == 0 ? c10::optional<c10::ArrayRef<double>>() : c10::optional<c10::ArrayRef<double>>(at::ArrayRef<double>(scaleFactors, scaleFactorsLength))))
    );
}

Tensor THSTensor_upsample_nearest2d_backward(
    const Tensor grad_output,
    const int64_t* outputSize, const int outputSizeLength,
    const int64_t* inputSize, const int inputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength)
{
    CATCH_TENSOR(torch::upsample_nearest2d_backward(
        *grad_output,
        (outputSize == 0 ? c10::optional<c10::IntArrayRef>() : c10::optional<c10::IntArrayRef>(at::IntArrayRef(outputSize, outputSizeLength))),
        at::IntArrayRef(inputSize, inputSizeLength),
        (scaleFactors == 0 ? c10::optional<c10::ArrayRef<double>>() : c10::optional<c10::ArrayRef<double>>(at::ArrayRef<double>(scaleFactors, scaleFactorsLength))))
    );
}

Tensor THSTensor_upsample_nearest3d(
    const Tensor tensor,
    const int64_t* outputSize, const int outputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength)
{
    CATCH_TENSOR(torch::upsample_nearest3d(
        *tensor,
        (outputSize == 0 ? c10::optional<c10::IntArrayRef>() : c10::optional<c10::IntArrayRef>(at::IntArrayRef(outputSize, outputSizeLength))),
        (scaleFactors == 0 ? c10::optional<c10::ArrayRef<double>>() : c10::optional<c10::ArrayRef<double>>(at::ArrayRef<double>(scaleFactors, scaleFactorsLength))))
    );
}

Tensor THSTensor_upsample_nearest3d_backward(
    const Tensor grad_output,
    const int64_t* outputSize, const int outputSizeLength,
    const int64_t* inputSize, const int inputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength)
{
    CATCH_TENSOR(torch::upsample_nearest3d_backward(
        *grad_output,
        (outputSize == 0 ? c10::optional<c10::IntArrayRef>() : c10::optional<c10::IntArrayRef>(at::IntArrayRef(outputSize, outputSizeLength))),
        at::IntArrayRef(inputSize, inputSizeLength),
        (scaleFactors == 0 ? c10::optional<c10::ArrayRef<double>>() : c10::optional<c10::ArrayRef<double>>(at::ArrayRef<double>(scaleFactors, scaleFactorsLength))))
    );
}