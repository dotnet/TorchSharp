// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSTensor.h"

#include <iostream>
#include <fstream>

Tensor THSTensor_abs(const Tensor tensor)
{
    CATCH_TENSOR(tensor->abs());
}

Tensor THSTensor_abs_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->abs_());
}

Tensor THSTensor_acos(const Tensor tensor)
{
    CATCH_TENSOR(tensor->acos());
}

Tensor THSTensor_acos_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->acos_());
}

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

Tensor THSTensor_adaptive_avg_pool3d_backward(
    const Tensor grad_output,
    const Tensor tensor)
{
    CATCH_TENSOR(torch::adaptive_avg_pool3d_backward(
        *grad_output,
        *tensor));
}

Tensor THSTensor_add(const Tensor left, const Tensor right, const Scalar alpha)
{
    CATCH_TENSOR(left->add(*right, *alpha));
}

Tensor THSTensor_add_(const Tensor left, const Tensor right, const Scalar alpha)
{
    CATCH_TENSOR(left->add_(*right, *alpha));
}

Tensor THSTensor_add_scalar(const Tensor left, const Scalar right, const Scalar alpha)
{
    CATCH_TENSOR(left->add(*right, *alpha));
}

Tensor THSTensor_add_scalar_(const Tensor left, const Scalar right, const Scalar alpha)
{
    CATCH_TENSOR(left->add_(*right, *alpha));
}

Tensor THSTensor_addbmm(const Tensor mat, const Tensor batch1, const Tensor batch2, const float beta, const float alpha)
{
    CATCH_TENSOR(mat->addbmm(*batch1, *batch2, beta, alpha));
}

Tensor THSTensor_addbmm_(const Tensor mat, const Tensor batch1, const Tensor batch2, const float beta, const float alpha)
{
    CATCH_TENSOR(mat->addbmm_(*batch1, *batch2, beta, alpha));
}

Tensor THSTensor_addcdiv(const Tensor left, const Tensor tensor1, const Tensor tensor2, const Scalar value)
{
    CATCH_TENSOR(left->addcdiv(*tensor1, *tensor2, *value));
}

Tensor THSTensor_addcdiv_(const Tensor left, const Tensor tensor1, const Tensor tensor2, const Scalar value)
{
    CATCH_TENSOR(left->addcdiv_(*tensor1, *tensor2, *value));
}

Tensor THSTensor_addcmul(const Tensor left, const Tensor tensor1, const Tensor tensor2, const Scalar value)
{
    CATCH_TENSOR(left->addcmul(*tensor1, *tensor2, *value));
}

Tensor THSTensor_addcmul_(const Tensor left, const Tensor tensor1, const Tensor tensor2, const Scalar value)
{
    CATCH_TENSOR(left->addcmul_(*tensor1, *tensor2, *value));
}

Tensor THSTensor_addmm(const Tensor mat, const Tensor mat1, const Tensor mat2, const float beta, const float alpha)
{
    CATCH_TENSOR(mat->addmm(*mat1, *mat2, beta, alpha));
}

Tensor THSTensor_addmm_(const Tensor mat, const Tensor mat1, const Tensor mat2, const float beta, const float alpha)
{
    CATCH_TENSOR(mat->addmm_(*mat1, *mat2, beta, alpha));
}

Tensor THSTensor_addmv(const Tensor mat, const Tensor mat1, const Tensor vec2, const float beta, const float alpha)
{
    CATCH_TENSOR(mat->addmv(*mat1, *vec2, beta, alpha));
}

Tensor THSTensor_addmv_(const Tensor mat, const Tensor mat1, const Tensor vec2, const float beta, const float alpha)
{
    CATCH_TENSOR(mat->addmv_(*mat1, *vec2, beta, alpha));
}

Tensor THSTensor_addr(const Tensor mat, const Tensor mat1, const Tensor vec2, const float beta, const float alpha)
{
    CATCH_TENSOR(mat->addr(*mat1, *vec2, beta, alpha));
}

Tensor THSTensor_addr_(const Tensor mat, const Tensor mat1, const Tensor vec2, const float beta, const float alpha)
{
    CATCH_TENSOR(mat->addr_(*mat1, *vec2, beta, alpha));
}

int THSTensor_allclose(const Tensor left, const Tensor right, double rtol, double atol, bool equal_nan)
{
    CATCH_RETURN(int, 0, left->allclose(*right, rtol, atol, equal_nan));
}

Tensor THSTensor_all(const Tensor tensor)
{
    CATCH_TENSOR(tensor->all());
}

Tensor THSTensor_all_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
{
    CATCH_TENSOR(tensor->all(dim, keepdim));
}

Tensor THSTensor_angle(const Tensor tensor)
{
    CATCH_TENSOR(tensor->angle());
}

Tensor THSTensor_any(const Tensor tensor)
{
    CATCH_TENSOR(tensor->any());
}

Tensor THSTensor_any_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
{
    CATCH_TENSOR(tensor->any(dim, keepdim));
}

Tensor THSTensor_arange(const Scalar start, const Scalar end, const Scalar step, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::arange(*start, *end, *step, options));
}

Tensor THSTensor_arange_out(const Scalar start, const Scalar end, const Scalar step, const Tensor out)
{
    CATCH_TENSOR(torch::arange_out(*out, *start, *end, *step));
}


Tensor THSTensor_linspace(const double start, const double end, const int64_t steps, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::linspace(start, end, steps, options));
}

Tensor THSTensor_logspace(const double start, const double end, const int64_t steps, double base, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::logspace(start, end, steps, base, options));
}


Tensor THSTensor_arccosh(const Tensor tensor)
{
    CATCH_TENSOR(tensor->arccosh());
}

Tensor THSTensor_arccosh_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->arccosh_());
}

Tensor THSTensor_arcsinh(const Tensor tensor)
{
    CATCH_TENSOR(tensor->arcsinh());
}

Tensor THSTensor_arcsinh_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->arcsinh_());
}

Tensor THSTensor_arctanh(const Tensor tensor)
{
    CATCH_TENSOR(tensor->arctanh());
}

Tensor THSTensor_arctanh_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->arctanh_());
}

Tensor THSTensor_argmax(const Tensor tensor)
{
    CATCH_TENSOR(tensor->argmax());
}

Tensor THSTensor_argmax_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
{
    CATCH_TENSOR(tensor->argmax(dim, keepdim));
}

Tensor THSTensor_argmin(const Tensor tensor)
{
    CATCH_TENSOR(tensor->argmin());
}

Tensor THSTensor_argmin_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
{
    CATCH_TENSOR(tensor->argmin(dim, keepdim));
}

Tensor THSTensor_asin(const Tensor tensor)
{
    CATCH_TENSOR(tensor->asin());
}

Tensor THSTensor_asin_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->asin_());
}

Tensor THSTensor_atan(const Tensor tensor)
{
    CATCH_TENSOR(tensor->atan());
}

Tensor THSTensor_atan_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->atan_());
}

Tensor THSTensor_atan2(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->atan2(*other));
}

Tensor THSTensor_atan2_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->atan2_(*other));
}

Tensor THSTensor_atleast_1d(const Tensor tensor)
{
    CATCH_TENSOR(torch::atleast_1d(*tensor));
}

Tensor THSTensor_atleast_2d(const Tensor tensor)
{
    CATCH_TENSOR(torch::atleast_2d(*tensor));
}

Tensor THSTensor_atleast_3d(const Tensor tensor)
{
    CATCH_TENSOR(torch::atleast_3d(*tensor));
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

void THSTensor_backward(Tensor tensor)
{
    CATCH(
        tensor->backward();
    )
}

Tensor THSTensor_baddbmm(
    const Tensor batch1,
    const Tensor batch2,
    const Tensor mat,
    const float beta,
    const float alpha)
{
    CATCH_TENSOR(mat->baddbmm(*batch1, *batch2, beta, alpha));
}

Tensor THSTensor_bincount(const Tensor tensor, const Tensor weights, const int64_t minlength)
{
    CATCH_TENSOR(tensor->bincount((weights ? *weights : at::Tensor()), minlength));
}

Tensor THSTensor_bitwise_and(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->bitwise_and(*other));
}

Tensor THSTensor_bitwise_and_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->bitwise_and_(*other));
}

Tensor THSTensor_bitwise_not(const Tensor tensor)
{
    CATCH_TENSOR(tensor->bitwise_not());
}

Tensor THSTensor_bitwise_not_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->bitwise_not_());
}

Tensor THSTensor_bitwise_or(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->bitwise_or(*other));
}

Tensor THSTensor_bitwise_or_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->bitwise_or_(*other));
}

Tensor THSTensor_bitwise_xor(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->bitwise_xor(*other));
}

Tensor THSTensor_bitwise_xor_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->bitwise_xor_(*other));
}

Tensor THSTensor_block_diag(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::block_diag(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_bmm(const Tensor batch1, const Tensor batch2)
{
    CATCH_TENSOR(batch1->bmm(*batch2));
}

Tensor THSTensor_broadcast_to(const Tensor tensor, const int64_t* shape, const int shape_len)
{
    CATCH_TENSOR(tensor->broadcast_to(at::ArrayRef<int64_t>(shape, shape_len)));
}

EXPORT_API(Tensor) THSTensor_bucketize(const Tensor tensor, const Tensor boundaries, const bool out_int32, const bool right)
{
    CATCH_TENSOR(torch::bucketize(*tensor, *boundaries, out_int32, right));
}

double THSTensor_clip_grad_norm_(const Tensor* tensors, const int length, const double max_norm, const double norm_type)
{
    double res = 0.0;
    CATCH(
        res = torch::nn::utils::clip_grad_norm_(toTensors<at::Tensor>((torch::Tensor**)tensors, length), max_norm, norm_type);
    );
    return res;
}

Tensor THSTensor_cat(const Tensor* tensors, const int length, const int64_t dim)
{
    CATCH_TENSOR(torch::cat(toTensors<at::Tensor>((torch::Tensor**)tensors, length), dim));
}

Tensor THSTensor_ceil(const Tensor tensor)
{
    CATCH_TENSOR(tensor->ceil());
}

Tensor THSTensor_ceil_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->ceil_());
}

Tensor THSTensor_celu(const Tensor tensor)
{
    CATCH_TENSOR(torch::celu(*tensor));
}

Tensor THSTensor_celu_(const Tensor tensor)
{
    CATCH_TENSOR(torch::celu_(*tensor));
}

Tensor THSTensor_cholesky(const Tensor tensor, const bool upper)
{
    CATCH_TENSOR(tensor->cholesky(upper));
}

Tensor THSTensor_cholesky_inverse(const Tensor tensor, const bool upper)
{
    CATCH_TENSOR(tensor->cholesky_inverse(upper));
}

Tensor THSTensor_cholesky_solve(const Tensor tensor, const Tensor tensor2, const bool upper)
{
    CATCH_TENSOR(tensor->cholesky_solve(*tensor2, upper));
}

Tensor THSTensor_clamp(const Tensor tensor, const Scalar min, const Scalar max)
{
    CATCH_TENSOR(tensor->clamp(*min, *max));
}

Tensor THSTensor_clamp_(const Tensor tensor, const Scalar min, const Scalar max)
{
    CATCH_TENSOR(tensor->clamp_(*min, *max));
}

Tensor THSTensor_clamp_max(const Tensor tensor, const Scalar max)
{
    CATCH_TENSOR(tensor->clamp_max(*max));
}

Tensor THSTensor_clamp_max_(const Tensor tensor, const Scalar max)
{
    CATCH_TENSOR(tensor->clamp_max_(*max));
}

Tensor THSTensor_clamp_min(const Tensor tensor, const Scalar min)
{
    CATCH_TENSOR(tensor->clamp_min(*min));
}

Tensor THSTensor_clamp_min_(const Tensor tensor, const Scalar min)
{
    CATCH_TENSOR(tensor->clamp_min_(*min));
}

Tensor THSTensor_clone(const Tensor tensor)
{
    CATCH_TENSOR(tensor->clone());
}

Tensor THSTensor_copy_(const Tensor input, const Tensor other, const bool non_blocking)
{
    CATCH_TENSOR(input->copy_(*other, non_blocking));
}

Tensor THSTensor_contiguous(const Tensor tensor)
{
    CATCH_TENSOR(tensor->contiguous());
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

Tensor THSTensor_copysign(const Tensor input, const Tensor other)
{
    CATCH_TENSOR(input->copysign(*other));
}

Tensor THSTensor_cos(const Tensor tensor)
{
    CATCH_TENSOR(tensor->cos());
}

Tensor THSTensor_cos_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->cos_());
}

Tensor THSTensor_cosh(const Tensor tensor)
{
    CATCH_TENSOR(tensor->cosh());
}

Tensor THSTensor_cosh_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->cosh_());
}

Tensor THSTensor_cross(const Tensor tensor, const Tensor other, const int64_t dim)
{
    CATCH_TENSOR(tensor->cross(*other, dim));
}

Tensor THSTensor_cpu(const Tensor tensor)
{
    CATCH_TENSOR(tensor->cpu());
}

Tensor THSTensor_cuda(const Tensor tensor)
{
    CATCH_TENSOR(tensor->cuda());
}

void THSTensor_cummax(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim)
{
    auto max = tensor->cummax(dim);
    Tensor* result = allocator(2);
    result[0] = new torch::Tensor(std::get<0>(max));
    result[1] = new torch::Tensor(std::get<1>(max));
}

void THSTensor_cummin(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim)
{
    auto max = tensor->cummin(dim);
    Tensor* result = allocator(2);
    result[0] = new torch::Tensor(std::get<0>(max));
    result[1] = new torch::Tensor(std::get<1>(max));
}

Tensor THSTensor_cumprod(const Tensor tensor, const int64_t dim, bool has_type, const int8_t dtype)
{
    CATCH_TENSOR(has_type ? tensor->cumprod(dim, (c10::ScalarType)dtype) : tensor->cumprod(dim));
}

Tensor THSTensor_cumsum(const Tensor tensor, const int64_t dim, bool has_type, const int8_t dtype)
{
    CATCH_TENSOR(has_type ? tensor->cumsum(dim, (c10::ScalarType)dtype) : tensor->cumsum(dim));
}

void* THSTensor_data(const Tensor tensor)
{
    CATCH_RETURN(void*, NULL, tensor->data_ptr());
}

float THSTensor_data_idx_float16(const Tensor tensor, const int64_t i)
{
    CATCH_RETURN(float, NULL, (float)(tensor->data_ptr<c10::Half>())[i]);
}

float THSTensor_data_idx_bfloat16(const Tensor tensor, const int64_t i)
{
    CATCH_RETURN(float, NULL, (float)(tensor->data_ptr<c10::BFloat16>())[i]);
}

Tensor THSTensor_deg2rad(const Tensor tensor)
{
    CATCH_TENSOR(torch::deg2rad(*tensor));
}

const char* THSTensor_device_str(const Tensor tensor)
{
    auto device = tensor->device();

    return make_sharable_string(device.str());
}

int THSTensor_device_index(const Tensor tensor)
{
    auto device = tensor->device();
    return device.index();
}

int THSTensor_device_type(const Tensor tensor)
{
    auto device = tensor->device();
    return (int)device.type();
}

Tensor THSTensor_diag(const Tensor tensor, const int64_t diagonal)
{
    CATCH_TENSOR(tensor->diag(diagonal));
}

Tensor THSTensor_diagflat(const Tensor tensor, const int64_t offset)
{
    CATCH_TENSOR(tensor->diagflat(offset));
}

Tensor THSTensor_diagonal(const Tensor tensor, const int64_t offset, const int64_t dim1, const int64_t dim2)
{
    CATCH_TENSOR(tensor->diagonal(offset, dim1, dim2));
}

Tensor THSTensor_diff(const Tensor tensor, const int64_t n, const int64_t dim, const Tensor prepend, const Tensor append)
{
    c10::optional<at::Tensor> prep = prepend != nullptr ? *prepend : c10::optional<at::Tensor>(c10::nullopt);
    c10::optional<at::Tensor> app = append != nullptr ? *append : c10::optional<at::Tensor>(c10::nullopt);
    CATCH_TENSOR(tensor->diff(n, dim, prep, app));
}

void THSTensor_dispose(const Tensor tensor)
{
    delete tensor;
}

Tensor THSTensor_digamma(const Tensor tensor)
{
    CATCH_TENSOR(tensor->digamma());
}

Tensor THSTensor_digamma_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->digamma_());
}

Tensor THSTensor_dist(const Tensor tensor, const Tensor other, const float p)
{
    CATCH_TENSOR(tensor->dist(*other, p));
}

Tensor THSTensor_div(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->div(*right));
}

Tensor THSTensor_div_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->div_(*right));
}

Tensor THSTensor_div_scalar(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->div(*right));
}

Tensor THSTensor_div_scalar_(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->div_(*right));
}

Tensor THSTensor_dot(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->dot(*right));
}

Tensor THSTensor_einsum(const char* equation, const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::einsum(equation, toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

int64_t THSTensor_element_size(const Tensor tensor)
{
    CATCH_RETURN(int64_t, 0, tensor->element_size());
}

Tensor THSTensor_elu(const Tensor tensor, const Scalar alpha, const Scalar scale, const Scalar input_scale)
{
    CATCH_TENSOR(torch::elu(*tensor, *alpha, *scale, *input_scale));
}

Tensor THSTensor_elu_(const Tensor tensor, const Scalar alpha, const Scalar scale, const Scalar input_scale)
{
    CATCH_TENSOR(torch::elu_(*tensor, *alpha, *scale, *input_scale));
}

Tensor THSTensor_empty_out(const int64_t* sizes, const int length, const Tensor out)
{
    CATCH_TENSOR(torch::empty_out(*out, at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_empty(
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::empty(at::ArrayRef<int64_t>(sizes, length), options));
}

Tensor THSTensor_empty_like(
    const Tensor input,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::empty_like(*input, options));
}

Tensor THSTensor_empty_strided(
    const int64_t* sizes,
    const int sz_length,
    const int64_t* strides,
    const int str_length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::empty_strided(at::ArrayRef<int64_t>(sizes, sz_length), at::ArrayRef<int64_t>(strides, str_length), options));
}

Tensor THSTensor_as_strided(
    const Tensor input,
    const int64_t* sizes,
    const int sz_length,
    const int64_t* strides,
    const int str_length,
    const int64_t storage_offset)
{
    CATCH_TENSOR(input->as_strided(at::ArrayRef<int64_t>(sizes, sz_length), at::ArrayRef<int64_t>(strides, str_length), storage_offset));
}

Tensor THSTensor_eye(const int64_t n, const int64_t m, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::eye(n, m, options));
}

Tensor THSTensor_eye_out(const int64_t n, const int64_t m, const Tensor out)
{
    CATCH_TENSOR(torch::eye_out(*out, n, m));
}


Tensor THSTensor_eq(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->eq(*right));
}

Tensor THSTensor_eq_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->eq_(*right));
}

Tensor THSTensor_eq_scalar(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->eq(*right));
}

Tensor THSTensor_eq_scalar_(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->eq_(*right));
}

int THSTensor_equal(const Tensor left, const Tensor right)
{
    CATCH_RETURN(int, 0, left->equal(*right));
}

Tensor THSTensor_exp(const Tensor tensor)
{
    CATCH_TENSOR(tensor->exp());
}

Tensor THSTensor_exp2(const Tensor tensor)
{
    CATCH_TENSOR(tensor->exp2());
}

Tensor THSTensor_exp_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->exp_());
}

Tensor THSTensor_expm1(const Tensor tensor)
{
    CATCH_TENSOR(tensor->expm1());
}

Tensor THSTensor_expm1_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->expm1_());
}

Tensor THSTensor_erf(const Tensor tensor)
{
    CATCH_TENSOR(tensor->erf());
}

Tensor THSTensor_erf_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->erf_());
}

Tensor THSTensor_erfc(const Tensor tensor)
{
    CATCH_TENSOR(tensor->erfc());
}

Tensor THSTensor_erfc_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->erfc_());
}

Tensor THSTensor_erfinv(const Tensor tensor)
{
    CATCH_TENSOR(tensor->erfinv());
}

Tensor THSTensor_erfinv_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->erfinv_());
}

Tensor THSTensor_expand(const Tensor tensor, const int64_t* sizes, const int length, bool implicit)
{
    CATCH_TENSOR(tensor->expand(at::ArrayRef<int64_t>(sizes, length), implicit));
}

Tensor THSTensor_repeat(const Tensor tensor, const int64_t* sizes, const int length)
{
    CATCH_TENSOR(tensor->repeat(at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_movedim(const Tensor tensor, const int64_t* src, const int src_len, const int64_t* dst, const int dst_len)
{
    CATCH_TENSOR(tensor->movedim(at::ArrayRef<int64_t>(src, src_len), at::ArrayRef<int64_t>(dst, dst_len)));
}

Tensor THSTensor_count_nonzero(const Tensor tensor, const int64_t* dim, const int dim_len)
{
    CATCH_TENSOR(tensor->count_nonzero(at::ArrayRef<int64_t>(dim, dim_len)));
}


Tensor THSTensor_fft(const Tensor tensor, const int64_t n, const int64_t dim, const char* norm)
{
    auto nArg = (n == -1 ? c10::optional<int64_t>() : c10::optional<int64_t>(n));
    auto normArg = (norm == NULL ? c10::optional<std::string>() : c10::optional<std::string>(norm));
    CATCH_TENSOR(torch::fft::fft(*tensor, nArg, dim, normArg));
}

Tensor THSTensor_ifft(const Tensor tensor, const int64_t n, const int64_t dim, const char* norm)
{
    auto nArg = (n == -1 ? c10::optional<int64_t>() : c10::optional<int64_t>(n));
    auto normArg = (norm == NULL ? c10::optional<std::string>() : c10::optional<std::string>(norm));
    CATCH_TENSOR(torch::fft::ifft(*tensor, nArg, dim, normArg));
}

Tensor THSTensor_irfft(const Tensor tensor, const int64_t n, const int64_t dim, const char* norm)
{
    auto nArg = (n == -1 ? c10::optional<int64_t>() : c10::optional<int64_t>(n));
    auto normArg = (norm == NULL ? c10::optional<std::string>() : c10::optional<std::string>(norm));
    CATCH_TENSOR(torch::fft::irfft(*tensor, nArg, dim, normArg));
}

Tensor THSTensor_rfft(const Tensor tensor, const int64_t n, const int64_t dim, const char* norm)
{
    auto nArg = (n == -1 ? c10::optional<int64_t>() : c10::optional<int64_t>(n));
    auto normArg = (norm == NULL ? c10::optional<std::string>() : c10::optional<std::string>(norm));
    CATCH_TENSOR(torch::fft::rfft(*tensor, nArg, dim, normArg));
}

Tensor THSTensor_flip(const Tensor tensor, const int64_t* sizes, const int length)
{
    CATCH_TENSOR(tensor->flip(at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_fliplr(const Tensor tensor)
{
    CATCH_TENSOR(tensor->fliplr());
}

Tensor THSTensor_flipud(const Tensor tensor)
{
    CATCH_TENSOR(tensor->flipud());
}

Tensor THSTensor_fill_(const Tensor tensor, const Scalar value)
{
    CATCH_TENSOR(tensor->fill_(*value));
}

Tensor THSTensor_float_power(const Tensor input, const Tensor exponent)
{
    CATCH_TENSOR(input->float_power(*exponent));
}

Tensor THSTensor_floor(const Tensor tensor)
{
    CATCH_TENSOR(tensor->floor());
}

Tensor THSTensor_floor_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->floor_());
}

Tensor THSTensor_fmax(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->fmax(*right));
}

Tensor THSTensor_fmin(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->fmin(*right));
}

Tensor THSTensor_fmod(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->fmod(*right));
}

Tensor THSTensor_fmod_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->fmod_(*right));
}

Tensor THSTensor_fmod_scalar(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->fmod(*right));
}

Tensor THSTensor_fmod_scalar_(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->fmod_(*right));
}

Tensor THSTensor_frac(const Tensor tensor)
{
    CATCH_TENSOR(tensor->frac());
}

Tensor THSTensor_frac_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->frac_());
}

Tensor THSTensor_gather(
    const Tensor tensor,
    const int64_t dim,
    const Tensor index)
{
    CATCH_TENSOR(torch::gather(*tensor, dim, *index));
}

Tensor THSTensor_ge(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->ge(*right));
}

Tensor THSTensor_ge_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->ge_(*right));
}

Tensor THSTensor_ge_scalar(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->ge(*right));
}

Tensor THSTensor_ge_scalar_(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->ge_(*right));
}

Tensor THSTensor_gelu(const Tensor tensor)
{
    CATCH_TENSOR(torch::gelu(*tensor));
}

Tensor THSTensor_get1(const Tensor tensor, int64_t index)
{
    CATCH_TENSOR((*tensor)[index]);
}

Tensor THSTensor_get2(const Tensor tensor, int64_t index1, int64_t index2)
{
    CATCH_TENSOR((*tensor)[index1][index2]);
}

Tensor THSTensor_get3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3)
{
    CATCH_TENSOR((*tensor)[index1][index2][index3]);
}

Tensor THSTensor_get4(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4)
{
    CATCH_TENSOR((*tensor)[index1][index2][index3][index4]);
}

Tensor THSTensor_get5(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5)
{
    CATCH_TENSOR((*tensor)[index1][index2][index3][index4][index5]);
}

Tensor THSTensor_get6(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, int64_t index6)
{
    CATCH_TENSOR((*tensor)[index1][index2][index3][index4][index5][index6]);
}

Tensor THSTensor_gcd(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->gcd(*other));
}

Tensor THSTensor_gcd_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->gcd_(*other));
}

Tensor THSTensor_grad(const Tensor tensor)
{
    Tensor res;
    CATCH(
        torch::Tensor grad = tensor->grad();
    res = grad.defined() ? new torch::Tensor(grad) : NULL;
    );
    return res;
}

Tensor THSTensor_gt(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->gt(*right));
}

Tensor THSTensor_gt_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->gt_(*right));
}

Tensor THSTensor_gt_scalar(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->gt(*right));
}

Tensor THSTensor_gt_scalar_(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->gt_(*right));
}

Tensor THSTensor_hardsigmoid(const Tensor tensor)
{
    CATCH_TENSOR(torch::hardsigmoid(*tensor));
}

Tensor THSTensor_hardsigmoid_(const Tensor tensor)
{
    CATCH_TENSOR(torch::hardsigmoid_(*tensor));
}

Tensor THSTensor_hardswish(const Tensor tensor)
{
    CATCH_TENSOR(torch::hardswish(*tensor));
}

Tensor THSTensor_hardswish_(const Tensor tensor)
{
    CATCH_TENSOR(torch::hardswish_(*tensor));
}

Tensor THSTensor_hardtanh(const Tensor tensor, const Scalar min, const Scalar max)
{
    CATCH_TENSOR(torch::hardtanh(*tensor, *min, *max));
}

Tensor THSTensor_hardtanh_(const Tensor tensor, const Scalar min, const Scalar max)
{
    CATCH_TENSOR(torch::hardtanh_(*tensor, *min, *max));
}

Tensor THSTensor_heaviside(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(torch::heaviside(*left, *right));
}

Tensor THSTensor_hypot(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(torch::hypot(*left, *right));
}

Tensor THSTensor_i0(const Tensor tensor)
{
    CATCH_TENSOR(torch::i0(*tensor));
}

Tensor THSTensor_igamma(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(torch::igamma(*tensor, *other));
}

Tensor THSTensor_igammac(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(torch::igammac(*tensor, *other));
}

Tensor THSTensor_isclose(const Tensor tensor, const Tensor other, const double rtol, const double atol, const bool equal_nan)
{
    CATCH_TENSOR(torch::isclose(*tensor, *other, rtol, atol, equal_nan));
}


Tensor THSTensor_isinf(const Tensor tensor)
{
    CATCH_TENSOR(torch::isinf(*tensor));
}

Tensor THSTensor_isfinite(const Tensor tensor)
{
    CATCH_TENSOR(torch::isfinite(*tensor));
}

Tensor THSTensor_isneginf(const Tensor tensor)
{
    CATCH_TENSOR(torch::isneginf(*tensor));
}

Tensor THSTensor_isposinf(const Tensor tensor)
{
    CATCH_TENSOR(torch::isposinf(*tensor));
}

Tensor THSTensor_isreal(const Tensor tensor)
{
    CATCH_TENSOR(torch::isreal(*tensor));
}

void completeTensorIndices(const int64_t* indexStarts,
    const int64_t* indexEnds,
    const int64_t* indexSteps,
    const Tensor* indexTensors,
    at::indexing::TensorIndex* indicesArray,
    const int indicesLength)
{
    // The indexStart encodes the kind of slice being performed for each dimension
    // range INT64_MIN..INT64_MIN+5 is for various singleton cases
    // range INT64_MIN+6 is for slice with absent start
    // range INT64_MIN+7 ... INT64_MIN/4 is for start of slice centered around INT64_MIN/2
    // range INT64_MIN/4+1 ... INT64_MAX is for single (normally a positive integer)
    for (int i = 0; i < indicesLength; i++)
    {
        auto n = indexStarts[i];
        if (n == INT64_MIN) // TensorIndex 'Null'
        {
            at::indexing::TensorIndex idx(c10::nullopt);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n == INT64_MIN + 1) // TensorIndex 'False'
        {
            at::indexing::TensorIndex idx(false);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n == INT64_MIN + 2) // TensorIndex 'True'
        {
            at::indexing::TensorIndex idx(true);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n == INT64_MIN + 3) // TensorIndex '...'
        {
            at::indexing::TensorIndex idx(at::indexing::Ellipsis);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n == INT64_MIN + 4) // TensorIndex 'None'
        {
            at::indexing::TensorIndex idx(at::indexing::None);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n == INT64_MIN + 5) // TensorIndex by tensor
        {
            at::indexing::TensorIndex idx(*indexTensors[i]);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else if (n > INT64_MIN / 4) // TensorIndex by integer
        {
            at::indexing::TensorIndex idx(n);
            // The '=' copy constructor for TensorIndex doesn't work
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
        else // TensorIndex by Slice
        {
            // slice
            auto start = (n == INT64_MIN + 6) ? c10::optional<int64_t>() : c10::optional<int64_t>(n - INT64_MIN / 2);
            auto end = (indexEnds == NULL || indexEnds[i] == INT64_MIN) ? c10::optional<int64_t>() : c10::optional<int64_t>(indexEnds[i]);
            auto step = (indexSteps == NULL || indexSteps[i] == INT64_MIN) ? c10::optional<int64_t>() : c10::optional<int64_t>(indexSteps[i]);
            at::indexing::TensorIndex idx(at::indexing::Slice(start, end, step));
            memcpy(&indicesArray[i], &idx, sizeof(at::indexing::TensorIndex));
        }
    }
}

Tensor THSTensor_index(Tensor tensor,
    const int64_t* indexStarts,
    const int64_t* indexEnds,
    const int64_t* indexSteps,
    const Tensor* indexTensors,
    const int indicesLength)
{
    at::indexing::TensorIndex* indicesArray = (at::indexing::TensorIndex*)alloca(indicesLength * sizeof(at::indexing::TensorIndex));
    memset(indicesArray, 0, indicesLength * sizeof(at::indexing::TensorIndex));
    // The indexStart encodes the kind of slice being performed for each dimension
    completeTensorIndices(indexStarts, indexEnds, indexSteps, indexTensors, indicesArray, indicesLength);
    auto indices = at::ArrayRef<at::indexing::TensorIndex>(indicesArray, indicesLength);
    CATCH_TENSOR(tensor->index(indices));
}

Tensor THSTensor_index_put_(Tensor tensor,
    const int64_t* indexStarts,
    const int64_t* indexEnds,
    const int64_t* indexSteps,
    const Tensor* indexTensors,
    const int indicesLength,
    const Tensor value)
{
    at::indexing::TensorIndex* indicesArray = (at::indexing::TensorIndex*)alloca(indicesLength * sizeof(at::indexing::TensorIndex));
    memset(indicesArray, 0, indicesLength * sizeof(at::indexing::TensorIndex));
    completeTensorIndices(indexStarts, indexEnds, indexSteps, indexTensors, indicesArray, indicesLength);
    auto indices = at::ArrayRef<at::indexing::TensorIndex>(indicesArray, indicesLength);
    CATCH_TENSOR(tensor->index_put_(indices, *value));
}

Tensor THSTensor_index_put_scalar_(Tensor tensor,
    const int64_t* indexStarts,
    const int64_t* indexEnds,
    const int64_t* indexSteps,
    const Tensor* indexTensors,
    const int indicesLength,
    const Scalar value)
{
    at::indexing::TensorIndex* indicesArray = (at::indexing::TensorIndex*)alloca(indicesLength * sizeof(at::indexing::TensorIndex));
    memset(indicesArray, 0, indicesLength * sizeof(at::indexing::TensorIndex));
    completeTensorIndices(indexStarts, indexEnds, indexSteps, indexTensors, indicesArray, indicesLength);
    auto indices = at::ArrayRef<at::indexing::TensorIndex>(indicesArray, indicesLength);
    CATCH_TENSOR(tensor->index_put_(indices, *value));
}

Tensor THSTensor_index_select(Tensor tensor, int64_t dim, Tensor index)
{
    CATCH_TENSOR(tensor->index_select(dim, *index));
}

Tensor THSTensor_indices(Tensor tensor)
{
    CATCH_TENSOR(tensor->_indices());
}

Tensor THSTensor_inner(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->inner(*right));
}

Tensor THSTensor_inverse(const Tensor tensor)
{
    CATCH_TENSOR(tensor->inverse());
}

int THSTensor_is_sparse(const Tensor tensor)
{
    CATCH_RETURN(int, 0, tensor->is_sparse());
}

Scalar THSTensor_item(const Tensor tensor)
{
    CATCH_RETURN(Scalar, NULL, new torch::Scalar(tensor->item()));
}

Tensor THSTensor_kron(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->kron(*right));
}

Tensor THSTensor_lcm(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->lcm(*other));
}

Tensor THSTensor_lcm_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->lcm_(*other));
}

Tensor THSTensor_ldexp(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->ldexp(*right));
}

Tensor THSTensor_le(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->le(*right));
}

Tensor THSTensor_le_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->le_(*right));
}

Tensor THSTensor_le_scalar(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->le(*right));
}

Tensor THSTensor_le_scalar_(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->le_(*right));
}

Tensor THSTensor_leaky_relu(const Tensor tensor, const Scalar negative_slope)
{
    CATCH_TENSOR(torch::leaky_relu(*tensor, *negative_slope));
}

Tensor THSTensor_leaky_relu_(const Tensor tensor, const Scalar negative_slope)
{
    CATCH_TENSOR(torch::leaky_relu_(*tensor, *negative_slope));
}

Tensor THSTensor_lgamma(const Tensor tensor)
{
    CATCH_TENSOR(tensor->lgamma());
}

Tensor THSTensor_load(const char* location)
{
    CATCH_RETURN_Tensor(
        torch::Tensor tensor;
    torch::load(tensor, location);
    res = ResultTensor(tensor);
    );
}

Tensor THSTensor_log_sigmoid(const Tensor tensor)
{
    CATCH_TENSOR(torch::log_sigmoid(*tensor));
}

Tensor THSTensor_logcumsumexp(const Tensor tensor, const long dimension)
{
    CATCH_TENSOR(torch::logcumsumexp(*tensor, dimension));
}

Tensor THSTensor_logaddexp(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(torch::logaddexp(*tensor, *other));
}

Tensor THSTensor_logaddexp2(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(torch::logaddexp2(*tensor, *other));
}

Tensor THSTensor_logsumexp(const Tensor tensor, const long dim, const bool keepdim)
{
    CATCH_TENSOR(torch::logsumexp(*tensor, dim, keepdim));
}


//Tensor THSTensor_log_sigmoid_backward(const Tensor tensor)
//{
//    CATCH_TENSOR(torch::log_sigmoid_backward(*tensor));
//}

Tensor THSTensor_log(const Tensor tensor)
{
    CATCH_TENSOR(tensor->log());
}

Tensor THSTensor_log_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->log_());
}

Tensor THSTensor_log2(const Tensor tensor)
{
    CATCH_TENSOR(tensor->log2());
}

Tensor THSTensor_log2_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->log2_());
}

Tensor THSTensor_log10(const Tensor tensor)
{
    CATCH_TENSOR(tensor->log10());
}

Tensor THSTensor_log10_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->log10_());
}

Tensor THSTensor_log1p(const Tensor tensor)
{
    CATCH_TENSOR(tensor->log1p());
}

Tensor THSTensor_log1p_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->log1p_());
}

Tensor THSTensor_lerp(const Tensor tensor, const Tensor end, const Tensor weight)
{
    CATCH_TENSOR(tensor->lerp(*end, *weight));
}

Tensor THSTensor_lerp_(const Tensor tensor, const Tensor end, const Tensor weight)
{
    CATCH_TENSOR(tensor->lerp_(*end, *weight));
}

Tensor THSTensor_logical_and(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->logical_and(*other));
}

Tensor THSTensor_logical_and_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->logical_and_(*other));
}

Tensor THSTensor_logical_not(const Tensor tensor)
{
    CATCH_TENSOR(tensor->logical_not());
}

Tensor THSTensor_logical_not_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->logical_not_());
}

Tensor THSTensor_logical_or(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->logical_or(*other));
}

Tensor THSTensor_logical_or_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->logical_or_(*other));
}

Tensor THSTensor_logical_xor(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->logical_xor(*other));
}

Tensor THSTensor_logical_xor_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->logical_xor_(*other));
}

Tensor THSTensor_logit(const Tensor tensor, const double* eps)
{
    CATCH_TENSOR((eps == nullptr) ? tensor->logit() : tensor->logit(*eps));
}

Tensor THSTensor_lt(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->lt(*right));
}

Tensor THSTensor_lt_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->lt_(*right));
}

Tensor THSTensor_lt_scalar(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->lt(*right));
}

Tensor THSTensor_lt_scalar_(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->lt_(*right));
}

Tensor THSTensor_masked_fill(const Tensor tensor, const Tensor mask, const Scalar value)
{
    CATCH_TENSOR(tensor->masked_fill(*mask, *value));
}

Tensor THSTensor_masked_fill_(const Tensor tensor, const Tensor mask, const Scalar value)
{
    CATCH_TENSOR(tensor->masked_fill_(*mask, *value));
}

Tensor THSTensor_masked_scatter(const Tensor tensor, const Tensor mask, const Tensor value)
{
    CATCH_TENSOR(tensor->masked_scatter(*mask, *value));
}

Tensor THSTensor_masked_scatter_(const Tensor tensor, const Tensor mask, const Tensor value)
{
    CATCH_TENSOR(tensor->masked_scatter_(*mask, *value));
}

Tensor THSTensor_masked_select(const Tensor tensor, const Tensor mask)
{
    CATCH_TENSOR(tensor->masked_select(*mask));
}

Tensor THSTensor_matmul(const Tensor left, const Tensor right)
{
    return  new torch::Tensor(left->matmul(*right));
}

Tensor THSTensor_max(const Tensor tensor)
{
    CATCH_TENSOR(tensor->max());
}

Tensor THSTensor_max_elementwise(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->max(*other));
}

Tensor THSTensor_maximum(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->maximum(*other));
}

void THSTensor_max_along_dimension(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim, const bool keepdim)
{
    CATCH(
        auto max = tensor->max(dim, keepdim);
    Tensor * result = allocator(2);
    result[0] = new torch::Tensor(std::get<0>(max));
    result[1] = new torch::Tensor(std::get<1>(max));
    )
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

Tensor THSTensor_mean(const Tensor tensor)
{
    CATCH_TENSOR(tensor->mean());
}

Tensor THSTensor_mean_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim, bool has_type, const int8_t dtype)
{
    CATCH_TENSOR(
        has_type ?
        tensor->mean(at::ArrayRef<int64_t>(dimensions, length), keepdim, (c10::ScalarType)dtype)
        :
        tensor->mean(at::ArrayRef<int64_t>(dimensions, length), keepdim))
}

Tensor THSTensor_median(const Tensor tensor)
{
    CATCH_TENSOR(tensor->median());
}

void THSTensor_mode(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim, const bool keep_dim)
{
    CATCH(
        auto res = tensor->mode(dim, keep_dim);
        const size_t sz = 2;
        Tensor * result = allocator(2);
        result[0] = new torch::Tensor(std::get<0>(res));
        result[1] = new torch::Tensor(std::get<1>(res));
    )
}

void THSTensor_chunk(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t chunks, const int64_t dim)
{
    CATCH(
        auto res = tensor->chunk(chunks, dim);
        const size_t sz = res.size();
        Tensor* result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
        )
}

Tensor THSTensor_quantile(const Tensor tensor, const Tensor q, const int64_t dim, const bool keep_dim)
{
    CATCH_TENSOR(dim == -1 ? tensor->quantile(*q, c10::nullopt, keep_dim) : tensor->quantile(*q, dim, keep_dim));
}

Tensor THSTensor_nanquantile(const Tensor tensor, const Tensor q, const int64_t dim, const bool keep_dim)
{
    CATCH_TENSOR(dim == -1 ? tensor->nanquantile(*q, c10::nullopt, keep_dim) : tensor->nanquantile(*q, dim, keep_dim));
}

Tensor THSTensor_min(const Tensor tensor)
{
    CATCH_TENSOR(tensor->min());
}

Tensor THSTensor_minimmum(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->minimum(*other));
}

Tensor THSTensor_min_elementwise(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->min(*other));
}

void THSTensor_min_along_dimension(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim, const bool keepdim)
{
    CATCH(
        auto max = tensor->min(dim, keepdim);
        Tensor * result = allocator(2);
        result[0] = new torch::Tensor(std::get<0>(max));
        result[1] = new torch::Tensor(std::get<1>(max));
    )
}

//Tensor THSTensor_median_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
//{
//    CATCH_TENSOR(tensor->median(dim, keepdim));
//}

Tensor THSTensor_mm(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->mm(*right));
}

Tensor THSTensor_mv(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->mv(*right));
}

Tensor THSTensor_vdot(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->vdot(*right));
}

Tensor THSTensor_msort(const Tensor tensor)
{
    CATCH_TENSOR(tensor->msort());
}

Tensor THSTensor_mul(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->mul(*right));
}

Tensor THSTensor_mul_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->mul_(*right));
}

Tensor THSTensor_mul_scalar(const Tensor tensor, const Scalar scalar)
{
    CATCH_TENSOR(tensor->mul(*scalar));
}

Tensor THSTensor_mul_scalar_(const Tensor tensor, const Scalar scalar)
{
    CATCH_TENSOR(tensor->mul_(*scalar));
}

Tensor THSTensor_mvlgamma(const Tensor tensor, int64_t p)
{
    CATCH_TENSOR(tensor->mvlgamma(p));
}

Tensor THSTensor_mvlgamma_(const Tensor tensor, int64_t p)
{
    CATCH_TENSOR(tensor->mvlgamma_(p));
}

Tensor THSTensor_nansum(const Tensor input)
{
    CATCH_TENSOR(torch::nansum(*input));
}

Tensor THSTensor_nanmedian(const Tensor input)
{
    CATCH_TENSOR(torch::nanmedian(*input));
}

Tensor THSTensor_nan_to_num(const Tensor input, double* _nan, double* _posinf, double* _neginf)
{
    c10::optional<double> nan = (_nan != nullptr) ? *_nan : c10::optional<double>(c10::nullopt);
    c10::optional<double> posinf = (_posinf != nullptr) ? *_posinf : c10::optional<double>(c10::nullopt);
    c10::optional<double> neginf = (_neginf != nullptr) ? *_neginf : c10::optional<double>(c10::nullopt);
    CATCH_TENSOR(torch::nan_to_num(*input, nan, posinf, neginf));
}

Tensor THSTensor_ne(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->ne(*right));
}

Tensor THSTensor_narrow(const Tensor tensor, int64_t dim, int64_t start, int64_t length)
{
    CATCH_TENSOR(tensor->narrow(dim, start, length))
}

Tensor THSTensor_ne_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->ne_(*right));
}

Tensor THSTensor_ne_scalar(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->ne(*right));
}

Tensor THSTensor_ne_scalar_(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->ne_(*right));
}

Tensor THSTensor_neg(const Tensor tensor)
{
    CATCH_TENSOR(tensor->neg());
}

Tensor THSTensor_neg_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->neg_());
}

Tensor THSTensor_new(
    void* data,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    int8_t scalar_type,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::from_blob(data, at::ArrayRef<int64_t>(sizes, szlength), deleter, options));
}

Tensor THSTensor_newInt64(
    int64_t* data,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(at::kLong))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::from_blob(data, at::ArrayRef<int64_t>(sizes, szlength), deleter, options));
}

// The data is passed in as float and copied into the array of Half in the C++ code
// since .NET doesn't know about 'float16' values.
Tensor THSTensor_newFloat16(
    float* rawArray,
    c10::Half* dataArray,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    const bool requires_grad)
{
    CATCH_RETURN_Tensor(
        int64_t sz = 1;
    for (int k = 0; k < szlength; k++)
        sz *= sizes[k];
    for (int64_t i = 0; i < sz; i++)
        dataArray[i] = (c10::Half)rawArray[i];
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(at::kHalf))
        .requires_grad(requires_grad);
    res = ResultTensor(torch::from_blob(dataArray, at::ArrayRef<int64_t>(sizes, szlength), deleter, options));
    )
}

// The data is passed in as float and copied into the array of Half in the C++ code
// since .NET doesn't know about 'float16' values.
Tensor THSTensor_newBFloat16(
    float* rawArray,
    c10::BFloat16* dataArray,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    const bool requires_grad)
{
    CATCH_RETURN_Tensor(
        int64_t sz = 1;
    for (int k = 0; k < szlength; k++)
        sz *= sizes[k];
    for (int64_t i = 0; i < sz; i++)
        dataArray[i] = (c10::BFloat16)rawArray[i];
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(at::kBFloat16))
        .requires_grad(requires_grad);
    res = ResultTensor(torch::from_blob(dataArray, at::ArrayRef<int64_t>(sizes, szlength), deleter, options));
    )
}

Tensor THSTensor_newInt8Scalar(int8_t data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Char))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newByteScalar(char data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Byte))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newBoolScalar(bool  data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Bool))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newInt16Scalar(short data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Short))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newInt32Scalar(int data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Int))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newInt64Scalar(int64_t data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Long))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newFloat64Scalar(double data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Double))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newFloat32Scalar(float data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Float))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor(data, options));
}

Tensor THSTensor_newFloat16Scalar(float data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::Half))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor((c10::Half)data, options));
}

Tensor THSTensor_newBFloat16Scalar(float data, const int device_type, const int device_index, bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(c10::ScalarType::BFloat16))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);
    CATCH_TENSOR(torch::tensor((c10::BFloat16)data, options));
}

Tensor THSTensor_nextafter(const Tensor input, const Tensor other)
{
    CATCH_TENSOR(torch::nextafter(*input, *other));
}

Tensor THSTensor_norm(const Tensor tensor, float p)
{
    CATCH_TENSOR(tensor->norm(p));
}

Tensor THSTensor_norm_along_dimension(const Tensor tensor, const int64_t dim, const bool keepdim, float p)
{
    CATCH_TENSOR(tensor->norm(p, dim, keepdim));
}

int64_t THSTensor_ndimension(const Tensor tensor)
{
    CATCH_RETURN(int64_t, 0, tensor->ndimension());
}

int64_t THSTensor_numel(const Tensor tensor)
{
    CATCH_RETURN(int64_t, 0, tensor->numel());
}

Tensor THSTensor_ones(
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::ones(at::ArrayRef<int64_t>(sizes, length), options));
}


Tensor THSTensor_ones_out(const int64_t* sizes, const int length, const Tensor out)
{
    CATCH_TENSOR(torch::ones_out(*out, at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_ones_like(
    const Tensor input,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::ones_like(*input, options));
}


Tensor THSTensor_outer(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->outer(*right));
}


Tensor THSTensor_permute(const Tensor tensor, const int64_t* sizes, const int length)
{
    CATCH_TENSOR(tensor->permute(at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_polygamma(const Tensor tensor, int64_t n)
{
    CATCH_TENSOR(tensor->polygamma(n));
}

Tensor THSTensor_polygamma_(const Tensor tensor, int64_t n)
{
    CATCH_TENSOR(tensor->polygamma_(n));
}

Tensor THSTensor_pow(const Tensor tensor, const Tensor exponent)
{
    CATCH_TENSOR(tensor->pow(*exponent));
}

Tensor THSTensor_pow_(const Tensor tensor, const Tensor exponent)
{
    CATCH_TENSOR(tensor->pow_(*exponent));
}

Tensor THSTensor_pow_scalar(const Tensor tensor, const Scalar exponent)
{
    CATCH_TENSOR(tensor->pow(*exponent));
}

Tensor THSTensor_pow_scalar_(const Tensor tensor, const Scalar exponent)
{
    CATCH_TENSOR(tensor->pow_(*exponent));
}

Tensor THSTensor_prelu(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->prelu(*right));
}

//Tensor THSTensor_prelu_backward(const Tensor grad_output, const Tensor self, const Tensor weight)
//{
//    CATCH_TENSOR(torch::prelu_backward(*grad_output, *self, *weight));
//}

Tensor THSTensor_rad2deg(const Tensor tensor)
{
    CATCH_TENSOR(torch::rad2deg(*tensor));
}


Tensor THSTensor_rand(
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    Tensor tensor;
    CATCH(
        auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    tensor = new torch::Tensor(torch::rand(at::ArrayRef<int64_t>(sizes, length), options));
    )
        return tensor;
}

Tensor THSTensor_rand_out(const int64_t* sizes, const int length, const Tensor out)
{
    CATCH_TENSOR(torch::rand_out(*out, at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_rand_like(
    const Tensor input,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::rand_like(*input, options));
}

Tensor THSTensor_randint(
    const int64_t high,
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    Tensor tensor;
    CATCH(
        auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    tensor = new torch::Tensor(torch::randint(high, at::ArrayRef<int64_t>(sizes, length), options));
    )
        return tensor;
}

Tensor THSTensor_randint_out(const int64_t high, const int64_t* sizes, const int length, const Tensor out)
{
    CATCH_TENSOR(torch::randint_out(*out, high, at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_randint_like(
    const Tensor input,
    const int64_t low,
    const int64_t high,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::randint_like(*input, low, high, options));
}

Tensor THSTensor_randn(
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::randn(at::ArrayRef<int64_t>(sizes, length), options));
}

Tensor THSTensor_randn_out(const int64_t* sizes, const int length, const Tensor out)
{
    CATCH_TENSOR(torch::randn_out(*out, at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_randn_like(
    const Tensor input,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::randn_like(*input, options));
}


Tensor THSTensor_randperm(const int64_t n,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    Tensor tensor;
    CATCH(
        auto options =
        at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    tensor = new torch::Tensor(torch::randperm(n, options));
    )
        return tensor;
}

Tensor THSTensor_randperm_out(const int64_t n, const Tensor out)
{
    CATCH_TENSOR(torch::randperm_out(*out, n));
}

Tensor THSTensor_ravel(const Tensor tensor)
{
    CATCH_TENSOR(torch::ravel(*tensor));
}

Tensor THSTensor_relu(const Tensor tensor)
{
    CATCH_TENSOR(torch::relu(*tensor));
}

Tensor THSTensor_relu_(const Tensor tensor)
{
    CATCH_TENSOR(torch::relu_(*tensor));
}

Tensor THSTensor_reciprocal(const Tensor tensor)
{
    CATCH_TENSOR(torch::reciprocal(*tensor));
}

Tensor THSTensor_reciprocal_(const Tensor tensor)
{
    CATCH_TENSOR(torch::reciprocal_(*tensor));
}

Tensor THSTensor_relu6(const Tensor tensor)
{
    CATCH_TENSOR(torch::nn::functional::relu6(*tensor));
}

Tensor THSTensor_relu6_(const Tensor tensor)
{
    CATCH_TENSOR(torch::nn::functional::relu6(*tensor, torch::nn::functional::ReLU6FuncOptions().inplace(true)));
}

Tensor THSTensor_remainder(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->remainder(*right));
}

Tensor THSTensor_remainder_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->remainder_(*right));
}

Tensor THSTensor_remainder_scalar(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->remainder(*right));
}

Tensor THSTensor_remainder_scalar_(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->remainder_(*right));
}

Tensor THSTensor_renorm(const Tensor tensor, const float p, const int64_t dim, const float maxnorm)
{
    CATCH_TENSOR(tensor->renorm(p, dim, maxnorm));
}

int THSTensor_requires_grad(const Tensor tensor)
{
    CATCH_RETURN(int, 0, tensor->requires_grad());
}

Tensor THSTensor_reshape(const Tensor tensor, const int64_t* shape, const int length)
{
    CATCH_TENSOR(tensor->reshape(at::ArrayRef<int64_t>(shape, length)));
}

Tensor THSTensor_round(const Tensor tensor)
{
    CATCH_TENSOR(tensor->round());
}

Tensor THSTensor_round_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->round_());
}

Tensor THSTensor_rsqrt(const Tensor tensor)
{
    CATCH_TENSOR(tensor->rsqrt());
}

Tensor THSTensor_rsqrt_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->rsqrt_());
}

void THSTensor_save(const Tensor tensor, const char* location)
{
    CATCH(
        torch::save(*tensor, location);
    );
}

Tensor THSTensor_selu(const Tensor tensor)
{
    CATCH_TENSOR(torch::selu(*tensor));
}

Tensor THSTensor_selu_(const Tensor tensor)
{
    CATCH_TENSOR(torch::selu_(*tensor));
}

Tensor THSTensor_silu(const Tensor tensor)
{
    CATCH_TENSOR(torch::silu(*tensor));
}

Tensor THSTensor_silu_(const Tensor tensor)
{
    CATCH_TENSOR(torch::silu_(*tensor));
}

void THSTensor_set1(const Tensor tensor, int64_t index, Scalar value)
{
    CATCH(
        (*tensor)[index] = *value;
    )
}

void THSTensor_set2(const Tensor tensor, int64_t index1, int64_t index2, Scalar value)
{
    CATCH(
        (*tensor)[index1][index2] = *value;
    )
}

void THSTensor_set3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, Scalar value)
{
    CATCH(
        (*tensor)[index1][index2][index3] = *value;
    )
}

void THSTensor_set4(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, Scalar value)
{
    CATCH(
        (*tensor)[index1][index2][index3][index4] = *value;
    )
}

void THSTensor_set5(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, Scalar value)
{
    CATCH(
        (*tensor)[index1][index2][index3][index4][index5] = *value;
    )
}

void THSTensor_set6(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, int64_t index6, Scalar value)
{
    CATCH(
        (*tensor)[index1][index2][index3][index4][index5][index6] = *value;
    )
}

Tensor THSTensor_sign(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sign());
}

Tensor THSTensor_sign_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sign_());
}

Tensor THSTensor_signbit(const Tensor tensor)
{
    CATCH_TENSOR(tensor->signbit());
}

Tensor THSTensor_sqrt(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sqrt());
}

Tensor THSTensor_sqrt_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sqrt_());
}

Tensor THSTensor_softplus(const Tensor tensor)
{
    CATCH_TENSOR(torch::softplus(*tensor));
}

Tensor THSTensor_scatter(
    const Tensor tensor,
    const int64_t dim,
    const Tensor index,
    const Tensor source)
{
    CATCH_TENSOR(torch::scatter(*tensor, dim, *index, *source));
}

Tensor THSTensor_slice(const Tensor tensor, int64_t dim, int64_t start, int64_t finish, int64_t step)
{
    CATCH_TENSOR(tensor->slice(dim, start, finish, step));
}

Tensor THSTensor_sparse(
    Tensor indices,
    Tensor values,
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::sparse_coo_tensor(*indices, *values, at::ArrayRef<int64_t>(sizes, length), options));
}

Tensor THSTensor_sigmoid(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sigmoid());
}

Tensor THSTensor_sigmoid_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sigmoid_());
}

Tensor THSTensor_sin(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sin());
}

Tensor THSTensor_sin_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sin_());
}

Tensor THSTensor_sinc(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sinc());
}

Tensor THSTensor_sinc_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sinc_());
}

Tensor THSTensor_sinh(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sinh());
}

Tensor THSTensor_sinh_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sinh_());
}

void THSTensor_split_with_size(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t split_size,
    const int64_t dim)
{
    CATCH(
        auto res = tensor->split(split_size, dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_split_with_sizes(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* sizes,
    const int length,
    const int64_t dim)
{
    CATCH(
        auto res = tensor->split_with_sizes(at::ArrayRef<int64_t>(sizes, length), dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_tensor_split_with_size(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t n,
    const int64_t dim)
{
    CATCH(
        auto res = tensor->tensor_split(n, dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_tensor_split_with_sizes(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* sizes,
    const int length,
    const int64_t dim)
{
    CATCH(
        auto res = tensor->tensor_split(at::ArrayRef<int64_t>(sizes, length), dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

void THSTensor_tensor_split_with_tensor_sizes(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const Tensor sizes,
    const int64_t dim)
{
    CATCH(
        auto res = tensor->tensor_split(*sizes, dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

Tensor THSTensor_squeeze(Tensor tensor, int64_t dim)
{
    CATCH_TENSOR(tensor->squeeze(dim));
}

int64_t THSTensor_stride(const Tensor tensor, const int64_t dim)
{
    CATCH_RETURN(int64_t, 0, tensor->stride(dim));
}

int64_t THSTensor_size(const Tensor tensor, const int64_t dim)
{
    CATCH_RETURN(int64_t, 0, tensor->size(dim));
}

Tensor THSTensor_sub(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->sub(*right));
}

Tensor THSTensor_sub_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->sub_(*right));
}

Tensor THSTensor_sub_scalar(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->sub(*right));
}

Tensor THSTensor_set_requires_grad(const Tensor tensor, const bool requires_grad)
{
    CATCH_TENSOR(tensor->set_requires_grad(requires_grad));
}

Tensor THSTensor_stack(const Tensor* tensors, const int length, const int64_t dim)
{
    CATCH_TENSOR(torch::stack(toTensors<at::Tensor>((torch::Tensor**)tensors, length), dim));
}

Tensor THSTensor_hstack(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::hstack(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_vstack(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::vstack(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_dstack(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::dstack(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_column_stack(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::column_stack(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_row_stack(const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::row_stack(toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

Tensor THSTensor_std(const Tensor tensor)
{
    CATCH_TENSOR(tensor->std());
}

Tensor THSTensor_std_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool unbiased, bool keepdim)
{
    CATCH_TENSOR(tensor->std(at::ArrayRef<int64_t>(dimensions, length), unbiased, keepdim));
}

Tensor THSTensor_sub_scalar_(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->sub_(*right));
}

Tensor THSTensor_sum(const Tensor tensor, bool has_type, const int8_t dtype)
{
    CATCH_TENSOR(has_type ? tensor->sum((c10::ScalarType)dtype) : tensor->sum())
}

Tensor THSTensor_sum_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim, bool has_type, const int8_t dtype)
{
    CATCH_TENSOR(
        has_type ?
        tensor->sum(at::ArrayRef<int64_t>(dimensions, length), keepdim, (c10::ScalarType)dtype)
        :
        tensor->sum(at::ArrayRef<int64_t>(dimensions, length), keepdim))
}

Tensor THSTensor_t(const Tensor tensor)
{
    CATCH_TENSOR(tensor->t());
}

Tensor THSTensor_take(const Tensor tensor, const Tensor indices)
{
    CATCH_TENSOR(tensor->take(*indices));
}

Tensor THSTensor_tan(const Tensor tensor)
{
    CATCH_TENSOR(tensor->tan());
}

Tensor THSTensor_tan_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->tan_());
}

Tensor THSTensor_tanh(const Tensor tensor)
{
    CATCH_TENSOR(tensor->tanh());
}

Tensor THSTensor_tanh_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->tanh_());
}

Tensor THSTensor_tile(const Tensor tensor, const int64_t* rep, const int rep_length)
{
    CATCH_TENSOR(tensor->tile(at::ArrayRef<int64_t>(rep, rep_length)));
}


void THSTensor_topk(const Tensor tensor, Tensor* (*allocator)(size_t length), const int k, const int64_t dim, const bool largest, const bool sorted)
{
    CATCH(
        auto topk = tensor->topk(k, dim, largest, sorted);
    Tensor * result = allocator(2);
    result[0] = new torch::Tensor(std::get<0>(topk));
    result[1] = new torch::Tensor(std::get<1>(topk));
    )
}

Tensor THSTensor_trunc(const Tensor tensor)
{
    CATCH_TENSOR(tensor->trunc());
}

Tensor THSTensor_trunc_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->trunc_());
}

int8_t THSTensor_type(const Tensor tensor)
{
    CATCH_RETURN(int8_t, 0, (int8_t)tensor->scalar_type());
}

Tensor THSTensor_to_dense(Tensor tensor)
{
    CATCH_TENSOR(tensor->to_dense());
}

Tensor THSTensor_to_device(const Tensor tensor, const int device_type, const int device_index)
{
    CATCH_RETURN_Tensor(
        auto device = c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index);
    res = ResultTensor(tensor->to(device));
    );
}

Tensor THSTensor_to_type(const Tensor tensor, int8_t scalar_type)
{
    CATCH_TENSOR(tensor->toType(at::ScalarType(scalar_type)));
}

Tensor THSTensor_to_type_and_device(const Tensor tensor, int8_t scalar_type, const int device_type, const int device_index)
{
    CATCH_RETURN_Tensor(
        auto device = c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index);
    res = ResultTensor(tensor->to(device, at::ScalarType(scalar_type)));
    );
}

Tensor THSTensor_triu(const Tensor tensor, const int64_t diagonal)
{
    CATCH_TENSOR(tensor->triu(diagonal));
}

Tensor THSTensor_tril(const Tensor tensor, const int64_t diagonal)
{
    CATCH_TENSOR(tensor->tril(diagonal));
}

Tensor THSTensor_transpose(const Tensor tensor, const int64_t dim1, const int64_t dim2)
{
    CATCH_TENSOR(tensor->transpose(dim1, dim2));
}

Tensor THSTensor_transpose_(const Tensor tensor, const int64_t dim1, const int64_t dim2)
{
    CATCH_TENSOR(tensor->transpose_(dim1, dim2));
}

Tensor THSTensor_view(const Tensor tensor, const int64_t* shape, const int length)
{
    CATCH_TENSOR(tensor->view(at::ArrayRef<int64_t>(shape, length)));
}

void THSTensor_unbind(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim)
{
    CATCH(
        auto res = tensor->unbind(dim);
        const size_t sz = res.size();
        Tensor * result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

Tensor THSTensor_unsqueeze(Tensor tensor, int64_t dim)
{
    CATCH_TENSOR(tensor->unsqueeze(dim))
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

Tensor THSTensor_values(Tensor tensor)
{
    CATCH_TENSOR(tensor->_values());
}

Tensor THSTensor_vander(const Tensor tensor, const int64_t N, const bool increasing)
{
    CATCH_TENSOR(torch::vander(*tensor, N, increasing));
}

Tensor THSTensor_xlogy(const Tensor x, const Tensor y)
{
    CATCH_TENSOR(x->xlogy(*y));
}

Tensor THSTensor_zeros_out(const int64_t* sizes, const int length, const Tensor out)
{
    CATCH_TENSOR(torch::zeros_out(*out, at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_zeros(
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::zeros(at::ArrayRef<int64_t>(sizes, length), options));
}


Tensor THSTensor_zeros_like(
    const Tensor input,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::zeros_like(*input, options));
}



Tensor THSTensor_full_out(const int64_t* sizes, const int length, const Scalar value, const Tensor out)
{
    CATCH_TENSOR(torch::full_out(*out, at::ArrayRef<int64_t>(sizes, length), *value));
}

Tensor THSTensor_full(
    const int64_t* sizes,
    const int length,
    const Scalar value,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::full(at::ArrayRef<int64_t>(sizes, length), *value, options));
}


Tensor THSTensor_full_like(
    const Tensor input,
    const Scalar value,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::full_like(*input, *value, options));
}

// torch.lingalg:

Tensor THSLinalg_cholesky(const Tensor tensor)
{
    CATCH_TENSOR(torch::linalg::cholesky(*tensor));
}

Tensor THSLinalg_det(const Tensor tensor)
{
    CATCH_TENSOR(torch::linalg::linalg_det(*tensor));
}

Tensor THSLinalg_slogdet(const Tensor tensor, Tensor* logabsdet)
{
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg::slogdet(*tensor););
    *logabsdet = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSLinalg_eigh(const Tensor tensor, const char UPLO, Tensor* eigenvectors)
{
    std::string _uplo;
    _uplo.push_back(UPLO);
    std::tuple<at::Tensor, at::Tensor> res;
    CATCH(res = torch::linalg::eigh(*tensor, _uplo););
    *eigenvectors = ResultTensor(std::get<1>(res));
    return ResultTensor(std::get<0>(res));
}

Tensor THSLinalg_eigvalsh(const Tensor tensor, const char UPLO)
{
    std::string _uplo;
    _uplo.push_back(UPLO);
    CATCH_TENSOR(torch::linalg::eigvalsh(*tensor, _uplo));
}

Tensor THSLinalg_inv(const Tensor tensor)
{
    CATCH_TENSOR(torch::linalg::inv(*tensor));
}

Tensor THSLinalg_matrix_rank(const Tensor tensor, const double tol, const bool has_tol, const bool hermitian)
{
    if (has_tol)
    {
        CATCH_TENSOR(torch::linalg::matrix_rank(*tensor, tol, hermitian));
    }
    else
    {
        CATCH_TENSOR(torch::linalg::matrix_rank(*tensor, c10::nullopt, hermitian));
    }
}

Tensor THSLinalg_norm_str(const Tensor tensor, const char* p, const int64_t* dim, const int dim_length, const bool keepdim)
{
    c10::optional<at::IntArrayRef> dims = (dim == nullptr) ? c10::nullopt : c10::optional<at::IntArrayRef>(at::ArrayRef<int64_t>(dim, dim_length));
    CATCH_TENSOR(torch::linalg::linalg_norm(*tensor, p, dims, keepdim, c10::nullopt));
}

Tensor THSLinalg_norm_float(const Tensor tensor, const double p, const int64_t* dim, const int dim_length, const bool keepdim)
{
    c10::optional<at::IntArrayRef> dims = (dim == nullptr) ? c10::nullopt : c10::optional<at::IntArrayRef>(at::ArrayRef<int64_t>(dim, dim_length));
    CATCH_TENSOR(torch::linalg::linalg_norm(*tensor, p, dims, keepdim, c10::nullopt));
}

Tensor THSLinalg_norm_int(const Tensor tensor, const int p, const int64_t* dim, const int dim_length, const bool keepdim)
{
    c10::optional<at::IntArrayRef> dims = (dim == nullptr) ? c10::nullopt : c10::optional<at::IntArrayRef>(at::ArrayRef<int64_t>(dim, dim_length));
    CATCH_TENSOR(torch::linalg::linalg_norm(*tensor, p, dims, keepdim, c10::nullopt));
}

Tensor THSLinalg_norm_opt(const Tensor tensor, const int64_t* dim, const int dim_length, const bool keepdim)
{
    c10::optional<at::IntArrayRef> dims = (dim == nullptr) ? c10::nullopt : c10::optional<at::IntArrayRef>(at::ArrayRef<int64_t>(dim, dim_length));
    CATCH_TENSOR(torch::linalg::linalg_norm(*tensor, c10::nullopt, dims, keepdim, c10::nullopt));
}

Tensor THSLinalg_pinv(const Tensor tensor, const double rcond, const bool hermitian)
{
    CATCH_TENSOR(torch::linalg::pinv(*tensor, rcond, hermitian));
}

Tensor THSLinalg_solve(const Tensor tensor, Tensor other)
{
    CATCH_TENSOR(torch::linalg::solve(*tensor, *other));
}

Tensor THSLinalg_tensorinv(const Tensor tensor, const int64_t ind)
{
    CATCH_TENSOR(torch::linalg::tensorinv(*tensor, ind));
}

Tensor THSLinalg_tensorsolve(const Tensor tensor, Tensor other, const int64_t* dim, const int dim_length)
{
    c10::optional<at::IntArrayRef> dims = (dim == nullptr) ? c10::nullopt : c10::optional<at::IntArrayRef>(at::ArrayRef<int64_t>(dim, dim_length));
    CATCH_TENSOR(torch::linalg::tensorsolve(*tensor, *other, dims));
}

torch::nn::init::NonlinearityType get_nl_type(const int64_t nl)
{
    switch (nl)
    {
    default:
    case 0:  return torch::kLinear;
    case 1:  return torch::kConv1D;
    case 2:  return torch::kConv2D;
    case 3:  return torch::kConv3D;
    case 4:  return torch::kConvTranspose1D;
    case 5:  return torch::kConvTranspose2D;
    case 6:  return torch::kConvTranspose3D;
    case 7:  return torch::kSigmoid;
    case 8:  return torch::kTanh;
    case 9:  return torch::kReLU;
    case 10: return torch::kLeakyReLU;
    }
}

double THSInit_calculate_gain(int64_t nonlinearity, double param)
{
    CATCH_RETURN(double, 0.0, torch::nn::init::calculate_gain(get_nl_type(nonlinearity), param))
}

torch::nn::init::FanModeType get_fan_mode(int64_t mode)
{
    return mode == 0 ? torch::nn::init::FanModeType(torch::kFanIn) : torch::nn::init::FanModeType(torch::kFanOut);
}

Tensor THSInit_constant_(Tensor tensor, Scalar value)
{
    CATCH_TENSOR(torch::nn::init::constant_(*tensor, *value))
}

Tensor THSInit_dirac_(Tensor tensor)
{
    CATCH_TENSOR(torch::nn::init::dirac_(*tensor))
}

Tensor THSInit_eye_(Tensor tensor)
{
    CATCH_TENSOR(torch::nn::init::eye_(*tensor))
}

Tensor THSTensor_bernoulli(const Tensor tensor, const double p, const Generator gen)
{
    CATCH_TENSOR(gen == nullptr ? tensor->bernoulli(p) : tensor->bernoulli(p, *gen));
}

Tensor THSTensor_bernoulli_0(Tensor tensor, const double p, const Generator gen)
{
    CATCH_TENSOR(gen == nullptr ? tensor->bernoulli_(p) : tensor->bernoulli_(p, *gen));
}

Tensor THSTensor_bernoulli_1(Tensor tensor, const Tensor p_tensor, const Generator gen)
{
    CATCH_TENSOR(gen == nullptr ? tensor->bernoulli_(*p_tensor) : tensor->bernoulli_(*p_tensor, *gen));
}

Tensor THSTensor_multinomial(const Tensor tensor, const double num_samples, const bool replacement, const Generator gen)
{
    CATCH_TENSOR(gen == nullptr ? tensor->multinomial(num_samples, replacement) : tensor->multinomial(num_samples, replacement, *gen));
}

Tensor THSTensor_cauchy_(Tensor tensor, const double median, const double sigma, const Generator gen)
{
    CATCH_TENSOR(gen == nullptr ? tensor->cauchy_(median, sigma) : tensor->cauchy_(median, sigma, *gen));
}

Tensor THSTensor_exponential_(Tensor tensor, const double lambda, const Generator gen)
{
    CATCH_TENSOR(gen == nullptr ? tensor->exponential_(lambda) : tensor->exponential_(lambda, *gen));
}

Tensor THSTensor_geometric_(Tensor tensor, const double p, const Generator gen)
{
    CATCH_TENSOR(gen == nullptr ? tensor->geometric_(p) : tensor->geometric_(p, *gen));
}

Tensor THSTensor_log_normal_(Tensor tensor, double mean, double std, const Generator gen)
{
    CATCH_TENSOR(gen == nullptr ? tensor->log_normal_(mean, std) : tensor->log_normal_(mean, std, *gen));
}

Tensor THSTensor_normal_(Tensor tensor, double mean, double std, const Generator gen)
{
    CATCH_TENSOR(gen == nullptr ? tensor->normal_(mean, std) : tensor->normal_(mean, std, *gen));
}

Tensor THSTensor_random_(Tensor tensor, double low, double high, const Generator gen)
{
    CATCH_TENSOR(gen == nullptr ? tensor->random_(low, high) : tensor->random_(low, high, *gen));
}

Tensor THSTensor_uniform_(Tensor tensor, double low, double high, const Generator gen)
{
    CATCH_TENSOR(gen == nullptr ? tensor->uniform_(low, high) : tensor->uniform_(low, high, *gen));
}

Tensor THSInit_normal_(Tensor tensor, double mean, double std)
{
    CATCH_TENSOR(torch::nn::init::normal_(*tensor, mean, std))
}

Tensor THSInit_ones_(Tensor tensor)
{
    CATCH_TENSOR(torch::nn::init::ones_(*tensor))
}

Tensor THSInit_orthogonal_(Tensor tensor, double gain)
{
    CATCH_TENSOR(torch::nn::init::orthogonal_(*tensor, gain))
}

Tensor THSInit_sparse_(Tensor tensor, double sparsity, double std)
{
    CATCH_TENSOR(torch::nn::init::sparse_(*tensor, sparsity, std))
}

Tensor THSInit_uniform_(Tensor tensor, double low, double high)
{
    CATCH_TENSOR(torch::nn::init::uniform_(*tensor, low, high))
}

Tensor THSInit_kaiming_normal_(Tensor tensor, double a, const int64_t mode, const int64_t nonlinearity)
{
    CATCH_TENSOR(torch::nn::init::kaiming_normal_(*tensor, a, mode == 0 ? torch::nn::init::FanModeType(torch::kFanIn) : torch::nn::init::FanModeType(torch::kFanOut), get_nl_type(nonlinearity)))
}

Tensor THSInit_kaiming_uniform_(Tensor tensor, double a, const int64_t mode, const int64_t nonlinearity)
{
    CATCH_TENSOR(torch::nn::init::kaiming_uniform_(*tensor, a, mode == 0 ? torch::nn::init::FanModeType(torch::kFanIn) : torch::nn::init::FanModeType(torch::kFanOut), get_nl_type(nonlinearity)))
}

Tensor THSInit_xavier_normal_(Tensor tensor, double gain)
{
    CATCH_TENSOR(torch::nn::init::xavier_normal_(*tensor, gain))
}

Tensor THSInit_xavier_uniform_(Tensor tensor, double gain)
{
    CATCH_TENSOR(torch::nn::init::xavier_uniform_(*tensor, gain))
}

Tensor THSInit_zeros_(Tensor tensor)
{
    CATCH_TENSOR(torch::nn::init::zeros_(*tensor))
}

Tensor THSTensor_bartlett_window(const int64_t len, bool periodic, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::bartlett_window(len, periodic, options));
}

Tensor THSTensor_blackman_window(const int64_t len, bool periodic, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::blackman_window(len, periodic, options));
}

Tensor THSTensor_hamming_window(const int64_t len, bool periodic, double alpha, double beta, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::hamming_window(len, periodic, alpha, beta, options));
}

Tensor THSTensor_hann_window(const int64_t len, bool periodic, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::hann_window(len, periodic, options));
}

Tensor THSTensor_kaiser_window(const int64_t len, bool periodic, double beta, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad)
{
    auto options = at::TensorOptions()
        .dtype(at::ScalarType(scalar_type))
        .device(c10::Device((c10::DeviceType)device_type, (c10::DeviceIndex)device_index))
        .requires_grad(requires_grad);

    CATCH_TENSOR(torch::kaiser_window(len, periodic, beta, options));
}
