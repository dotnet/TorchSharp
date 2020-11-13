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

Tensor THSTensor_addmm(const Tensor mat, const Tensor mat1, const Tensor mat2, const float beta,const float alpha)
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

Tensor THSTensor_bernoulli(const Tensor tensor, const double p)
{
    CATCH_TENSOR(tensor->bernoulli(p));
}

Tensor THSTensor_bernoulli_(const Tensor tensor, const double p)
{
    CATCH_TENSOR(tensor->bernoulli_(p));
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

Tensor THSTensor_bmm(const Tensor batch1, const Tensor batch2)
{
    CATCH_TENSOR(batch1->bmm(*batch2));
}

Tensor THSTensor_cat(const Tensor* tensors, const int length, const int64_t dim)
{
    CATCH_TENSOR(torch::cat(toTensors<at::Tensor>((torch::Tensor**)tensors, length), dim));
}

Tensor THSTensor_cauchy_(const Tensor tensor, const double median, const double sigma)
{
    CATCH_TENSOR(tensor->cauchy_(median, sigma));
}

Tensor THSTensor_ceil(const Tensor tensor)
{
    CATCH_TENSOR(tensor->ceil());
}

Tensor THSTensor_ceil_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->ceil_());
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
    CATCH_TENSOR(tensor-> cross(*other, dim));
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

Tensor THSTensor_einsum(const char* equation, const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::einsum(equation, toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
}

int64_t THSTensor_element_size(const Tensor tensor)
{
    CATCH_RETURN(int64_t, 0, tensor->element_size());
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

Tensor THSTensor_exponential_(const Tensor tensor, const double lambd)
{
    CATCH_TENSOR(tensor->exponential_(lambd));
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

Tensor THSTensor_fft(const Tensor tensor, const int64_t dim, const bool normalized)
{
    CATCH_TENSOR(tensor->fft(dim, normalized));
}

Tensor THSTensor_ifft(const Tensor tensor, const int64_t signal_ndim, const bool normalized)
{
    CATCH_TENSOR(tensor->ifft(signal_ndim, normalized));
}

Tensor THSTensor_irfft(const Tensor tensor, const int64_t signal_ndim, const bool normalized, const bool onesided, const int64_t* signal_sizes, const int signal_sizes_length)
{
    CATCH_TENSOR(
        signal_sizes == NULL
           ? tensor->irfft(signal_ndim, normalized, onesided)
           : tensor->irfft(signal_ndim, normalized, onesided, at::ArrayRef<int64_t>(signal_sizes, signal_sizes_length)));
}

Tensor THSTensor_rfft(const Tensor tensor, const int64_t signal_ndim, const bool normalized, const bool onesided)
{
    CATCH_TENSOR(tensor->rfft(signal_ndim, normalized, onesided));
}

Tensor THSTensor_flip(const Tensor tensor, const int64_t* sizes, const int length)
{
    CATCH_TENSOR(tensor->flip(at::ArrayRef<int64_t>(sizes, length)));
}

Tensor THSTensor_fill_(const Tensor tensor, const Scalar value)
{
    CATCH_TENSOR(tensor->fill_(*value));
}

Tensor THSTensor_floor(const Tensor tensor)
{
    CATCH_TENSOR(tensor->floor());
}

Tensor THSTensor_floor_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->floor_());
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

Tensor THSTensor_geometric_(const Tensor tensor, const double p)
{
    CATCH_TENSOR(tensor->geometric_(p));
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

Tensor THSTensor_index_select(Tensor tensor, int64_t dim, Tensor index)
{
    CATCH_TENSOR(tensor->index_select(dim, *index));
}

Tensor THSTensor_indices(Tensor tensor)
{
    CATCH_TENSOR(tensor->_indices());
}

int THSTensor_is_sparse(const Tensor tensor)
{
    CATCH_RETURN(int, 0, tensor->is_sparse());
}

Scalar THSTensor_item(const Tensor tensor)
{
    CATCH_RETURN(Scalar, NULL, new torch::Scalar(tensor->item()));
}

Tensor THSTensor_lcm(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->lcm(*other));
}

Tensor THSTensor_lcm_(const Tensor tensor, const Tensor other)
{
    CATCH_TENSOR(tensor->lcm_(*other));
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

Tensor THSTensor_lgamma(const Tensor tensor)
{
    CATCH_TENSOR(tensor->lgamma());
}

Tensor THSTensor_lgamma_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->lgamma_());
}

Tensor THSTensor_load(const char* location)
{
    CATCH_RETURN_Tensor(
        torch::Tensor tensor;
    torch::load(tensor, location);
    res = ResultTensor(tensor);
    );
}

Tensor THSTensor_log_normal_(const Tensor tensor, const double mean, const double std)
{
    CATCH_TENSOR(tensor->log_normal_(mean, std));
}

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
    CATCH_TENSOR(tensor->log());
}

Tensor THSTensor_log2_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->log_());
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

Tensor THSTensor_matmul(const Tensor left, const Tensor right)
{
    return  new torch::Tensor(left->matmul(*right));
}

void THSTensor_max(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim, const bool keepdim)
{
    auto max = tensor->max(dim, keepdim);
    Tensor * result = allocator(2);
    result[0] = new torch::Tensor(std::get<0>(max));
    result[1] = new torch::Tensor(std::get<1>(max));
}

Tensor THSTensor_maxpool1d(
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

void THSTensor_maxpool1d_with_indices(
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

Tensor THSTensor_maxpool2d(
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
        at::ArrayRef<int64_t>(dilation, dilationLength)));
}

void THSTensor_maxpool2d_with_indices(
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

Tensor THSTensor_maxpool3d(
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

void THSTensor_maxpool3d_with_indices(
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

//Tensor THSTensor_median_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim)
//{
//    CATCH_TENSOR(tensor->median(dim, keepdim));
//}

Tensor THSTensor_mm(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->mm(*right));
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

Tensor THSTensor_multinomial(const Tensor tensor, const double num_samples, const bool replacement)
{
    CATCH_TENSOR(tensor->multinomial(num_samples, replacement));
}

Tensor THSTensor_mvlgamma(const Tensor tensor, int64_t p)
{
    CATCH_TENSOR(tensor->mvlgamma(p));
}

Tensor THSTensor_mvlgamma_(const Tensor tensor, int64_t p)
{
    CATCH_TENSOR(tensor->mvlgamma_(p));
}

Tensor THSTensor_ne(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->ne(*right));
}

Tensor THSTensor_narrow(const Tensor tensor, int64_t dim, int64_t start, int64_t length)
{
    CATCH_TENSOR(tensor->narrow(dim, start, length))
}

Tensor THSTensor_normal_(const Tensor tensor, const double mean, const double std)
{
    CATCH_TENSOR(tensor->normal_(mean, std));
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

Tensor THSTensor_newLong(
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

Tensor THSTensor_newSByteScalar(int8_t data, const int device_type, const int device_index, bool requires_grad)
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

Tensor THSTensor_relu(const Tensor tensor)
{
    CATCH_TENSOR(tensor->relu());
}

Tensor THSTensor_relu_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->relu_());
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

Tensor THSTensor_sinh(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sinh());
}

Tensor THSTensor_sinh_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sinh_());
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

Tensor THSTensor_t(const Tensor tensor)
{
    CATCH_TENSOR(tensor->t());
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
        Tensor* result = allocator(sz);
        for (size_t i = 0; i < sz; i++)
            result[i] = new torch::Tensor(res[i]);
    )
}

Tensor THSTensor_uniform_(const Tensor tensor, const double from, const double to)
{
    CATCH_TENSOR(tensor->uniform_(from, to));
}

Tensor THSTensor_unsqueeze(Tensor tensor, int64_t dim)
{
    CATCH_TENSOR(tensor->unsqueeze(dim))
}

Tensor THSTensor_values(Tensor tensor)
{
    CATCH_TENSOR(tensor->_values());
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

