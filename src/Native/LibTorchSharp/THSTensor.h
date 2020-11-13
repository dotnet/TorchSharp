// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "TH/THTensor.h"
#include "torch/torch.h"

#include "Utils.h"


#define _EXPORT_THS_TENSOR_API_NO_ARG(NAME)\
 EXPORT_API(Tensor) THSTensor_ ##NAME(const Tensor tensor) \
 {\
    CATCH_TENSOR(tensor-> NAME ()) ;\
 };\

#define _EXPORT_THS_TENSOR_API_SINGLE_ARG(NAME, ARG, ARG_NAME)\
 EXPORT_API(Tensor) THSTensor_ ##NAME(const Tensor tensor, ARG ARG_NAME) \
 {\
    CATCH_TENSOR(tensor-> NAME ( *ARG_NAME ));\
 };\

#define EXPORT_THS_TENSOR_API_NO_ARG(NAME) \
  _EXPORT_THS_TENSOR_API_NO_ARG(NAME) \
  _EXPORT_THS_TENSOR_API_NO_ARG(##NAME ## _)

#define EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG(NAME) \
  _EXPORT_THS_TENSOR_API_SINGLE_ARG(NAME, const Tensor, other) \
  _EXPORT_THS_TENSOR_API_SINGLE_ARG(##NAME ## _, const Tensor, other)

#define EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(NAME) \
  EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG(NAME)\
  EXPORT_API(Tensor) THSTensor_ ##NAME ## _scalar (const Tensor tensor, const Scalar other) \
  {\
     CATCH_TENSOR(tensor-> ##NAME (*other));\
  };\
  EXPORT_API(Tensor) THSTensor_##NAME ## _scalar_(const Tensor tensor, const Scalar other)\
  {\
     CATCH_TENSOR(tensor-> ##NAME ## _(*other));\
  };\


EXPORT_THS_TENSOR_API_NO_ARG(abs);

EXPORT_THS_TENSOR_API_NO_ARG(acos);

EXPORT_API(Tensor) THSTensor_add(const Tensor left, const Tensor right, const Scalar alpha);

EXPORT_API(Tensor) THSTensor_add_(const Tensor left, const Tensor right, const Scalar alpha);

EXPORT_API(Tensor) THSTensor_add_scalar(const Tensor left, const Scalar right, const Scalar alpha);

EXPORT_API(Tensor) THSTensor_add_scalar_(const Tensor left, const Scalar right, const Scalar alpha);

EXPORT_API(Tensor) THSTensor_addbmm(const Tensor left, const Tensor batch1, const Tensor batch2, const float beta, const float alpha);

EXPORT_API(Tensor) THSTensor_addbmm_(const Tensor left, const Tensor batch1, const Tensor batch2, const float beta, const float alpha);

EXPORT_API(Tensor) THSTensor_addcdiv(const Tensor left, const Tensor tensor1, const Tensor tensor2, const Scalar value);

EXPORT_API(Tensor) THSTensor_addcdiv_(const Tensor left, const Tensor tensor1, const Tensor tensor2, const Scalar value);

EXPORT_API(Tensor) THSTensor_addcmul(const Tensor left, const Tensor tensor1, const Tensor tensor2, const Scalar value);

EXPORT_API(Tensor) THSTensor_addcmul_(const Tensor left, const Tensor tensor1, const Tensor tensor2, const Scalar value);

EXPORT_API(Tensor) THSTensor_addmm(const Tensor left, const Tensor mat1, const Tensor mat2, const float beta, const float alpha);

EXPORT_API(Tensor) THSTensor_addmm_(const Tensor left, const Tensor mat1, const Tensor mat2, const float beta, const float alpha);

EXPORT_API(Tensor) THSTensor_addmv(const Tensor left, const Tensor mat1, const Tensor vec2, const float beta, const float alpha);

EXPORT_API(Tensor) THSTensor_addmv_(const Tensor left, const Tensor mat1, const Tensor vec2, const float beta, const float alpha);

EXPORT_API(int) THSTensor_allclose(const Tensor left, const Tensor right, double rtol, double atol, bool equal_nan);

EXPORT_API(Tensor) THSTensor_all(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_all_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim);

EXPORT_API(Tensor) THSTensor_any(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_any_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim);

EXPORT_API(Tensor) THSTensor_angle(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_arange(const Scalar start, const Scalar end, const Scalar step, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_arange_out(const Scalar start, const Scalar end, const Scalar step, const Tensor out);

EXPORT_API(Tensor) THSTensor_argmax(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_argmax_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim);

EXPORT_API(Tensor) THSTensor_argmin(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_argmin_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim);

EXPORT_THS_TENSOR_API_NO_ARG(asin);

EXPORT_THS_TENSOR_API_NO_ARG(atan);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG(atan2);

EXPORT_API(Tensor) THSTensor_baddbmm(const Tensor batch1, const Tensor batch2, const Tensor left, const float beta, const float alpha);

EXPORT_API(void) THSTensor_backward(Tensor tensor);

EXPORT_API(Tensor) THSTensor_bincount(const Tensor tensor, const Tensor weights, const int64_t minlength);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG(bitwise_and);

EXPORT_THS_TENSOR_API_NO_ARG(bitwise_not);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG(bitwise_or);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG(bitwise_xor);

EXPORT_API(Tensor) THSTensor_bmm(const Tensor b1wrapper, const Tensor b2wrapper);

EXPORT_API(Tensor) THSTensor_cat(const Tensor* tensor, const int length, const int64_t dim);

EXPORT_API(Tensor) THSTensor_clone(const Tensor input);

EXPORT_API(Tensor) THSTensor_contiguous(const Tensor input);

EXPORT_THS_TENSOR_API_NO_ARG(ceil);

EXPORT_API(Tensor) THSTensor_cholesky(const Tensor tensor, const bool upper);

EXPORT_API(Tensor) THSTensor_cholesky_inverse(const Tensor tensor, const bool upper);

EXPORT_API(Tensor) THSTensor_cholesky_solve(const Tensor tensor, const Tensor tensor2, const bool upper);

EXPORT_API(Tensor) THSTensor_clamp(const Tensor input, const Scalar min, const Scalar max);

EXPORT_API(Tensor) THSTensor_clamp_(const Tensor input, const Scalar min, const Scalar max);

EXPORT_API(Tensor) THSTensor_clamp_max(const Tensor input, const Scalar max);

EXPORT_API(Tensor) THSTensor_clamp_max_(const Tensor input, const Scalar max);

EXPORT_API(Tensor) THSTensor_clamp_min(const Tensor input, const Scalar min);

EXPORT_API(Tensor) THSTensor_clamp_min_(const Tensor input, const Scalar min);

EXPORT_API(Tensor) THSTensor_conv1d(const Tensor input, const Tensor weight, const Tensor bias,
    const int64_t* strides, const int strides_length,
    const int64_t* paddings, const int paddings_length,
    const int64_t* dilations, const int dilations_length,
    int64_t groups);

EXPORT_API(Tensor) THSTensor_conv2d(const Tensor input, const Tensor weight, const Tensor bias,
    const int64_t* strides, const int strides_length,
    const int64_t* paddings, const int paddings_length,
    const int64_t* dilations, const int dilations_length,
    int64_t groups);

EXPORT_API(Tensor) THSTensor_conv3d(const Tensor input, const Tensor weight, const Tensor bias,
    const int64_t* strides, const int strides_length,
    const int64_t* paddings, const int paddings_length,
    const int64_t* dilations, const int dilations_length,
    int64_t groups);

EXPORT_API(Tensor) THSTensor_conv_transpose1d(const Tensor input, const Tensor weight, const Tensor bias,
    const int64_t* strides, const int strides_length,
    const int64_t* paddings, const int paddings_length,
    const int64_t* output_padding, const int output_paddingLength,
    const int64_t* dilations, const int dilations_length,
    int64_t groups);

EXPORT_API(Tensor) THSTensor_conv_transpose2d(const Tensor input, const Tensor weight, const Tensor bias,
    const int64_t* strides, const int strides_length,
    const int64_t* paddings, const int paddings_length,
    const int64_t* output_padding, const int output_paddingLength,
    const int64_t* dilations, const int dilations_length,
    int64_t groups);

EXPORT_API(Tensor) THSTensor_conv_transpose3d(const Tensor input, const Tensor weight, const Tensor bias,
    const int64_t* strides, const int strides_length,
    const int64_t* paddings, const int paddings_length,
    const int64_t* output_padding, const int output_paddingLength,
    const int64_t* dilations, const int dilations_length,
    int64_t groups);

EXPORT_THS_TENSOR_API_NO_ARG(cos);

EXPORT_THS_TENSOR_API_NO_ARG(cosh);

EXPORT_API(Tensor) THSTensor_cpu(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_cross(const Tensor tensor, const Tensor other, const int64_t dim);

EXPORT_API(Tensor) THSTensor_cuda(const Tensor tensor);

EXPORT_API(void) THSTensor_cummax(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim);

EXPORT_API(void) THSTensor_cummin(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim);

EXPORT_API(Tensor) THSTensor_cumprod(const Tensor tensor, const int64_t dim, bool has_type, const int8_t dtype);

EXPORT_API(Tensor) THSTensor_cumsum(const Tensor tensor, const int64_t dim, bool has_type, const int8_t dtype);

EXPORT_API(void*) THSTensor_data(const Tensor tensor);

EXPORT_API(float) THSTensor_data_idx_half(const Tensor tensor, const int64_t i);

EXPORT_API(const char*) THSTensor_device_str(const Tensor tensor);

EXPORT_API(int) THSTensor_device_type(const Tensor tensor);

EXPORT_API(int) THSTensor_device_index(const Tensor tensor);

EXPORT_API(void) THSTensor_dispose(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_dist(const Tensor tensor, const Tensor other, const float p);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(div);

EXPORT_API(Tensor) THSTensor_einsum(const char* equation, const Tensor* tensors, const int length);

EXPORT_API(int64_t) THSTensor_element_size(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_empty(
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(eq);

EXPORT_API(int) THSTensor_equal(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_exp(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_exp_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_expm1(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_expm1_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_expand(const Tensor tensor, const int64_t* sizes, const int length, bool implicit);

EXPORT_API(Tensor) THSTensor_erf(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_erf_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_erfc(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_erfc_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_erfinv(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_erfinv_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_fill_(const Tensor tensor, Scalar value);

EXPORT_API(Tensor) THSTensor_fft(const Tensor tensor, const int64_t dim, const bool normalized);

EXPORT_API(Tensor) THSTensor_ifft(const Tensor tensor, const int64_t signal_ndim, const bool normalized);

EXPORT_API(Tensor) THSTensor_irfft(const Tensor tensor, const int64_t signal_ndim, const bool normalized, const bool onesided, const int64_t* signal_sizes, const int signal_sizes_length);

EXPORT_API(Tensor) THSTensor_rfft(const Tensor tensor, const int64_t signal_ndim, const bool normalized, const bool onesided);

EXPORT_API(Tensor) THSTensor_flip(const Tensor tensor, const int64_t* sizes, const int length);

EXPORT_API(Tensor) THSTensor_floor(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_floor_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_frac(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_frac_(const Tensor tensor);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(fmod)

EXPORT_API(Tensor) THSTensor_digamma(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_digamma_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_lgamma(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_lgamma_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_mvlgamma(const Tensor tensor, int64_t p);

EXPORT_API(Tensor) THSTensor_mvlgamma_(const Tensor tensor, int64_t p);

EXPORT_API(Tensor) THSTensor_polygamma(const Tensor tensor, int64_t n);

EXPORT_API(Tensor) THSTensor_polygamma_(const Tensor tensor, int64_t n);

EXPORT_API(Tensor) THSTensor_gather(const Tensor tensor, const int64_t dim, const Tensor index);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(ge);

EXPORT_API(Tensor) THSTensor_get1(const Tensor tensor, int64_t index);

EXPORT_API(Tensor) THSTensor_get2(const Tensor tensor, int64_t index1, int64_t index2);

EXPORT_API(Tensor) THSTensor_get3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3);

EXPORT_API(Tensor) THSTensor_get4(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4);

EXPORT_API(Tensor) THSTensor_get5(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5);

EXPORT_API(Tensor) THSTensor_get6(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, int64_t index6);

EXPORT_API(Tensor) THSTensor_gcd(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_gcd_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_grad(const Tensor tensor);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(gt);

EXPORT_API(Tensor) THSTensor_indices(Tensor tensor);

EXPORT_API(Tensor) THSTensor_index_select(Tensor tensor, int64_t dim, Tensor index);

EXPORT_API(int) THSTensor_is_sparse(const Tensor tensor);

EXPORT_API(Scalar) THSTensor_item(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_lcm(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_lcm_(const Tensor tensor);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(le);

EXPORT_API(Tensor) THSTensor_load(const char* location);

EXPORT_API(Tensor) THSTensor_log(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log2(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log2_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log10(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log10_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_lerp(const Tensor tensor, const Tensor end, const Tensor weight);

EXPORT_API(Tensor) THSTensor_lerp_(const Tensor tensor, const Tensor end, const Tensor weight);

EXPORT_API(Tensor) THSTensor_log1p(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log1p_(const Tensor tensor);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG(logical_and);

EXPORT_THS_TENSOR_API_NO_ARG(logical_not);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG(logical_or);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG(logical_xor);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(lt);

EXPORT_API(Tensor) THSTensor_matmul(const Tensor left, const Tensor right);

EXPORT_API(void) THSTensor_max(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim, const bool keep_dim);

EXPORT_API(Tensor) THSTensor_maxpool1d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode);

EXPORT_API(Tensor) THSTensor_maxpool2d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode);

EXPORT_API(Tensor) THSTensor_maxpool3d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode);

EXPORT_API(void) THSTensor_maxpool1d_with_indices(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode);

EXPORT_API(void) THSTensor_maxpool2d_with_indices(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode);

EXPORT_API(void) THSTensor_maxpool3d_with_indices(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode);

EXPORT_API(Tensor) THSTensor_maxunpool2d(
    const Tensor tensor,
    const Tensor indices,
    const int64_t* outputSize, const int outputSizeLength);

EXPORT_API(Tensor) THSTensor_maxunpool3d(
    const Tensor tensor,
    const Tensor indices,
    const int64_t* outputSize, const int outputSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength);

EXPORT_API(Tensor) THSTensor_mean(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_median(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_mm(const Tensor left, const Tensor right);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(mul)

EXPORT_API(Tensor) THSTensor_narrow(const Tensor tensor, int64_t dim, int64_t start, int64_t length);

EXPORT_API(int64_t) THSTensor_ndimension(const Tensor tensor);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(ne);

EXPORT_API(Tensor) THSTensor_neg(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_neg_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_new(
    void* data,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    int8_t scalar_type,
    const bool requires_grad);

EXPORT_API(Tensor) THSTensor_newLong(
    int64_t* data,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    const bool requires_grad);

EXPORT_API(Tensor) THSTensor_newHalf(
    float* rawArray,
    c10::Half* dataArray,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    const bool requires_grad);

EXPORT_API(Tensor) THSTensor_newSByteScalar(int8_t data, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newByteScalar(char data, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newBoolScalar(bool data, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newHalfScalar(float data, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newShortScalar(short data, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newIntScalar(int data, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newLongScalar(int64_t data, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newDoubleScalar(double data, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newFloatScalar(float data, bool requires_grad);

EXPORT_API(Tensor) THSTensor_norm(const Tensor tensor, float p);

EXPORT_API(Tensor) THSTensor_norm_along_dimension(const Tensor tensor, const int64_t dim, const bool keepdim, float p);

EXPORT_API(int64_t) THSTensor_numel(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_ones(const int64_t* sizes, const int length, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_ones_out(const int64_t* sizes, const int length, const Tensor out);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(pow);

EXPORT_API(Tensor) THSTensor_prelu(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_rand(const int64_t* sizes, const int length, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_rand_out(const int64_t* sizes, const int length, const Tensor out);

EXPORT_API(Tensor) THSTensor_randint(const int64_t high, const int64_t* sizes, const int length, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_randint_out(const int64_t high, const int64_t* sizes, const int length, const Tensor out);

EXPORT_API(Tensor) THSTensor_randn(const int64_t* sizes, const int length, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_randn_out(const int64_t* sizes, const int length, const Tensor out);

EXPORT_API(Tensor) THSTensor_randperm(const int64_t n, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_randperm_out(const int64_t n, const Tensor out);

EXPORT_API(Tensor) THSTensor_relu(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_relu_(const Tensor tensor);

EXPORT_API(int) THSTensor_requires_grad(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_reshape(const Tensor tensor, const int64_t* shape, const int length);

EXPORT_API(Tensor) THSTensor_round(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_round_(const Tensor tensor);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(remainder);

EXPORT_API(Tensor) THSTensor_rsqrt(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_rsqrt_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_renorm(const Tensor tensor, const float p, const int64_t dim, const float maxnorm);

EXPORT_API(Tensor) THSTensor_sigmoid(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sigmoid_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sign(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sign_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sin(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sin_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sinh(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sinh_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_softplus(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sqrt(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sqrt_(const Tensor tensor);

EXPORT_THS_TENSOR_API_OTHER_TENSOR_ARG_AND_ONE_SCALAR_ARG(sub);

EXPORT_API(Tensor) THSTensor_sum(const Tensor tensor, bool has_type, const int8_t dtype);

EXPORT_API(Tensor) THSTensor_sum_along_dimensions(const Tensor tensor, const int64_t * dimensions, int length, bool keepdim, bool has_type, const int8_t dtype);

EXPORT_API(void) THSTensor_save(const Tensor tensor, const char* location);

EXPORT_API(Tensor) THSTensor_scatter(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source);

EXPORT_API(Tensor) THSTensor_set_requires_grad(const Tensor tensor, const bool requires_grad);

EXPORT_API(void) THSTensor_set1(const Tensor tensor, int64_t index, Scalar value);

EXPORT_API(void) THSTensor_set2(const Tensor tensor, int64_t index1, int64_t index2, Scalar value);

EXPORT_API(void) THSTensor_set3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, Scalar value);

EXPORT_API(void) THSTensor_set4(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, Scalar value);

EXPORT_API(void) THSTensor_set5(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, Scalar value);

EXPORT_API(void) THSTensor_set6(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, int64_t index6, Scalar value);

EXPORT_API(int64_t) THSTensor_size(const Tensor tensor, const int64_t dim);

EXPORT_API(Tensor) THSTensor_slice(const Tensor tensor, int64_t dim, int64_t start, int64_t finish, int64_t step);

EXPORT_API(Tensor) THSTensor_sparse(
    Tensor indices,
    Tensor values,
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad);

EXPORT_API(void) THSTensor_split_with_sizes(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t* sizes, const int length, const int64_t dim);

EXPORT_API(Tensor) THSTensor_squeeze(Tensor tensor, int64_t dim);

EXPORT_API(Tensor) THSTensor_stack(const Tensor* tensor, const int length, const int64_t dim);

EXPORT_API(int64_t) THSTensor_stride(const Tensor tensor, const int64_t dim);

EXPORT_API(Tensor) THSTensor_tan(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_tan_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_tanh(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_tanh_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_t(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_transpose(const Tensor tensor, const int64_t dim1, const int64_t dim2);

EXPORT_API(Tensor) THSTensor_transpose_(const Tensor tensor, const int64_t dim1, const int64_t dim2);

EXPORT_API(Tensor) THSTensor_to_dense(Tensor tensor);

EXPORT_API(Tensor) THSTensor_to_device(const Tensor tensor, const int device_type, const int device_index);

EXPORT_API(Tensor) THSTensor_to_type(const Tensor tensor, int8_t scalar_type);

EXPORT_API(void) THSTensor_topk(const Tensor tensor, Tensor* (*allocator)(size_t length), const int k, const int64_t dim, const bool largest, const bool sorted);

EXPORT_API(int8_t) THSTensor_type(const Tensor tensor);

EXPORT_API(void) THSTensor_unbind(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim);

EXPORT_API(Tensor) THSTensor_unsqueeze(Tensor tensor, int64_t dim);

EXPORT_API(Tensor) THSTensor_values(Tensor tensor);

EXPORT_API(Tensor) THSTensor_view(const Tensor tensor, const int64_t* shape, const int length);

EXPORT_API(Tensor) THSTensor_zeros(const int64_t* sizes, const int length, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_zeros_out(const int64_t* sizes, const int length, const Tensor out);
