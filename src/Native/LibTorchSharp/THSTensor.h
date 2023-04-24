// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

EXPORT_API(Tensor) THSTensor_abs(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_abs_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_acos(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_acos_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_adaptive_avg_pool1d(const Tensor tensor, const int64_t* outputSize, const int outputSizeLength);

EXPORT_API(Tensor) THSTensor_adaptive_avg_pool2d(const Tensor tensor, const int64_t* outputSize, const int outputSizeLength);

EXPORT_API(Tensor) THSTensor_adaptive_avg_pool3d(const Tensor tensor, const int64_t* outputSize, const int outputSizeLength);

EXPORT_API(Tensor) THSTensor_adaptive_avg_pool3d_backward_out(const Tensor grad_input, const Tensor grad_output, const Tensor tensor);

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

EXPORT_API(Tensor) THSTensor_addr(const Tensor input, const Tensor mat1, const Tensor vec2, const float beta, const float alpha);

EXPORT_API(Tensor) THSTensor_addr_(const Tensor input, const Tensor mat1, const Tensor vec2, const float beta, const float alpha);

EXPORT_API(Tensor) THSTensor_adjoint(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_alias(const Tensor tensor);

EXPORT_API(int) THSTensor_allclose(const Tensor left, const Tensor right, double rtol, double atol, bool equal_nan);

EXPORT_API(Tensor) THSTensor_all(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_all_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim);

EXPORT_API(Tensor) THSTensor_amax(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim);
EXPORT_API(Tensor) THSTensor_amax_out(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim, const Tensor out);

EXPORT_API(Tensor) THSTensor_amin(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim);
EXPORT_API(Tensor) THSTensor_amin_out(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim, const Tensor out);

EXPORT_API(Tensor) THSTensor_aminmax(const Tensor tensor, const int64_t dim, bool keepdim, Tensor* max);

EXPORT_API(Tensor) THSTensor_any(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_any_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim);

EXPORT_API(Tensor) THSTensor_angle(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_arange(const Scalar start, const Scalar end, const Scalar step, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_arange_out(const Scalar start, const Scalar end, const Scalar step, const Tensor out);

EXPORT_API(Tensor) THSTensor_arccosh(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_arccosh_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_arcsinh(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_arcsinh_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_arctanh(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_arctanh_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_argmax(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_argmax_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim);

EXPORT_API(Tensor) THSTensor_argmin(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_argmin_along_dimension(const Tensor tensor, const int64_t dim, bool keepdim);

EXPORT_API(Tensor) THSTensor_argsort(const Tensor tensor, const int64_t dim, bool descending);

EXPORT_API(Tensor) THSTensor_argwhere(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_asin(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_asin_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_atan(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_atan_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_atan2(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_atan2_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_atleast_1d (const Tensor tensor);

EXPORT_API(Tensor) THSTensor_atleast_2d(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_atleast_3d(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_avg_pool1d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    bool ceil_mode,
    bool count_include_pad);

EXPORT_API(Tensor) THSTensor_avg_pool2d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    bool ceil_mode,
    bool count_include_pad);

EXPORT_API(Tensor) THSTensor_avg_pool2d_backward(
    const Tensor grad_output,
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    bool ceil_mode,
    bool count_include_pad,
    const int64_t divisor_override);

EXPORT_API(Tensor) THSTensor_avg_pool3d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    bool ceil_mode,
    bool count_include_pad);

EXPORT_API(Tensor) THSTensor_avg_pool3d_backward(
    const Tensor grad_output,
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    bool ceil_mode,
    bool count_include_pad,
    const int64_t divisor_override);

EXPORT_API(Tensor) THSTensor_baddbmm(const Tensor batch1, const Tensor batch2, const Tensor left, const float beta, const float alpha);

EXPORT_API(void) THSTensor_backward(Tensor tensor);

EXPORT_API(Tensor) THSTensor_bincount(const Tensor tensor, const Tensor weights, const int64_t minlength);

EXPORT_API(Tensor) THSTensor_bitwise_and(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_bitwise_and_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_bitwise_not(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_bitwise_not_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_bitwise_or(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_bitwise_or_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_bitwise_xor(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_bitwise_xor_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_bitwise_left_shift(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_bitwise_left_shift_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_bitwise_right_shift(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_bitwise_right_shift_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_block_diag(const Tensor* tensor, const int length);

EXPORT_API(Tensor) THSTensor_bmm(const Tensor b1wrapper, const Tensor b2wrapper);

EXPORT_API(Tensor) THSTensor_broadcast_to(const Tensor tensor, const int64_t* shape, const int shape_len);

EXPORT_API(void) THSTensor_broadcast_tensors(const Tensor* tensor, const int length, Tensor* (*allocator)(size_t length));

EXPORT_API(Tensor) THSTensor_bucketize(const Tensor tensor, const Tensor boundaries, const bool out_int32, const bool right);

EXPORT_API(Tensor) THSTensor_cartesian_prod(const Tensor* tensor, const int length);

EXPORT_API(Tensor) THSTensor_cat(const Tensor* tensor, const int length, const int64_t dim);

EXPORT_API(Tensor) THSTensor_channel_shuffle(const Tensor tensor, const int64_t groups);

EXPORT_API(Tensor) THSTensor_cdist(const Tensor x1, const Tensor x2, const double p, const int64_t compute_mode);

EXPORT_API(double) THSTensor_clip_grad_norm_(const Tensor* tensor, const int length, const double max_norm, const double norm_type);

EXPORT_API(void) THSTensor_clip_grad_value_(const Tensor* tensors, const int length, const double value);

EXPORT_API(Tensor) THSTensor_parameters_to_vector(const Tensor* tensors, const int length);

EXPORT_API(void) THSTensor_vector_to_parameters(const Tensor vec, const Tensor* tensors, const int length);

EXPORT_API(Tensor) THSTensor_clone(const Tensor input);

EXPORT_API(Tensor) THSTensor_combinations(const Tensor tensor, const int r, const bool with_replacement);

EXPORT_API(Tensor) THSTensor_contiguous(const Tensor input);

EXPORT_API(Tensor) THSTensor_ceil(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_ceil_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_celu(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_celu_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_cholesky(const Tensor tensor, const bool upper);

EXPORT_API(Tensor) THSTensor_cholesky_inverse(const Tensor tensor, const bool upper);

EXPORT_API(Tensor) THSTensor_cholesky_solve(const Tensor tensor, const Tensor tensor2, const bool upper);

EXPORT_API(void) THSTensor_chunk(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t chunks, const int64_t dim);

EXPORT_API(Tensor) THSTensor_clamp(const Tensor input, const Scalar min, const Scalar max);
EXPORT_API(Tensor) THSTensor_clamp_(const Tensor input, const Scalar min, const Scalar max);

EXPORT_API(Tensor) THSTensor_clamp_tensor(const Tensor input, const Tensor min, const Tensor max);
EXPORT_API(Tensor) THSTensor_clamp_tensor_(const Tensor input, const Tensor min, const Tensor max);

EXPORT_API(Tensor) THSTensor_clamp_max(const Tensor input, const Scalar max);

EXPORT_API(Tensor) THSTensor_clamp_max_(const Tensor input, const Scalar max);

EXPORT_API(Tensor) THSTensor_clamp_min(const Tensor input, const Scalar min);

EXPORT_API(Tensor) THSTensor_clamp_min_(const Tensor input, const Scalar min);

EXPORT_API(Tensor) THSTensor_complex(const Tensor real, const Tensor imag);

EXPORT_API(Tensor) THSTensor_conj(const Tensor tensor);

EXPORT_API(int64_t) THSTensor_is_nonzero(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_conj_physical(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_conj_physical_(const Tensor tensor);

EXPORT_API(int64_t) THSTensor_is_conj(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_resolve_conj(const Tensor tensor);


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

EXPORT_API(Tensor) THSTensor_copysign(const Tensor input, const Tensor other);

EXPORT_API(Tensor) THSTensor_copy_(const Tensor input, const Tensor other, const bool non_blocking);

EXPORT_API(Tensor) THSTensor_cos(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_corrcoef(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_cos_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_cosh(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_cosh_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_count_nonzero(const Tensor tensor, const int64_t* dim, const int dim_len);

EXPORT_API(Tensor) THSTensor_cov(const Tensor input, int64_t correction, const Tensor fweights, const Tensor aweights);

EXPORT_API(bool) THSTensor_is_cpu(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_cpu(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_cross(const Tensor tensor, const Tensor other, const int64_t dim);

EXPORT_API(Tensor) THSTensor_cuda(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_pin_memory(const Tensor tensor);

EXPORT_API(int64_t) THSTensor_is_pinned(const Tensor tensor);

EXPORT_API(void) THSTensor_cummax(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim);

EXPORT_API(void) THSTensor_cummin(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim);

EXPORT_API(Tensor) THSTensor_cumprod(const Tensor tensor, const int64_t dim, bool has_type, const int8_t dtype);

EXPORT_API(Tensor) THSTensor_cumsum(const Tensor tensor, const int64_t dim, bool has_type, const int8_t dtype);

EXPORT_API(void*) THSTensor_data(const Tensor tensor);

EXPORT_API(float) THSTensor_data_idx_float16(const Tensor tensor, const int64_t i);

EXPORT_API(float) THSTensor_data_idx_bfloat16(const Tensor tensor, const int64_t i);

EXPORT_API(Tensor) THSTensor_deg2rad(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_detach(const Tensor tensor);
EXPORT_API(Tensor) THSTensor_detach_(const Tensor tensor);

EXPORT_API(const char*) THSTensor_device_str(const Tensor tensor);

EXPORT_API(int) THSTensor_device_type(const Tensor tensor);

EXPORT_API(int) THSTensor_device_index(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_diag(const Tensor tensor, const int64_t diagonal);

EXPORT_API(Tensor) THSTensor_diag_embed(const Tensor tensor, const int64_t offset, const int64_t dim1, const int64_t dim2);

EXPORT_API(Tensor) THSTensor_trace(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_diagflat(const Tensor tensor, const int64_t offset);

EXPORT_API(Tensor) THSTensor_diagonal(const Tensor tensor, const int64_t offset, const int64_t dim1, const int64_t dim2);

EXPORT_API(Tensor) THSTensor_diff(const Tensor tensor, const int64_t n, const int64_t dim, const Tensor prepend, const Tensor append);

EXPORT_API(void) THSTensor_free(const Tensor tensor);

EXPORT_API(void) THSTensor_dispose(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_dist(const Tensor tensor, const Tensor other, const float p);

EXPORT_API(Tensor) THSTensor_div(const Tensor left, const Tensor right, const char* rounding_mode);

EXPORT_API(Tensor) THSTensor_div_(const Tensor left, const Tensor right, const char* rounding_mode);

EXPORT_API(Tensor) THSTensor_div_scalar(const Tensor left, const Scalar right, const char* rounding_mode);

EXPORT_API(Tensor) THSTensor_div_scalar_(const Tensor left, const Scalar right, const char* rounding_mode);

EXPORT_API(Tensor) THSTensor_dot(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_einsum(const char* equation, const Tensor* tensors, const int length);

EXPORT_API(int64_t) THSTensor_element_size(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_elu(const Tensor tensor, const Scalar alpha, const Scalar scale, const Scalar input_scale);

EXPORT_API(Tensor) THSTensor_elu_(const Tensor tensor, const Scalar alpha, const Scalar scale, const Scalar input_scale);

EXPORT_API(Tensor) THSTensor_empty(
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad);

EXPORT_API(Tensor) THSTensor_empty_out(const int64_t* sizes, const int length, const Tensor out);

EXPORT_API(Tensor) THSTensor_empty_like(
    const Tensor input,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad);

EXPORT_API(Tensor) THSTensor_empty_strided(
    const int64_t* sizes,
    const int sz_length,
    const int64_t* strides,
    const int str_length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad);

EXPORT_API(Tensor) THSTensor_as_strided(
    const Tensor input,
    const int64_t* sizes,
    const int sz_length,
    const int64_t* strides,
    const int str_length,
    const int64_t storage_offset);

EXPORT_API(Tensor) THSTensor_eq(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_eq_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_eq_scalar(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_eq_scalar_(const Tensor left, const Scalar right);

EXPORT_API(int) THSTensor_equal(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_exp(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_exp_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_exp2(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_expm1(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_expm1_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_expand(const Tensor tensor, const int64_t* sizes, const int length, bool implicit);

EXPORT_API(Tensor) THSTensor_erf(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_erf_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_erfc(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_erfc_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_erfinv(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_erfinv_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_eye(const int64_t n, const int64_t m, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_eye_out(const int64_t n, const int64_t m, const Tensor out);

EXPORT_API(Tensor) THSTensor_fill_(const Tensor tensor, Scalar value);

EXPORT_API(Tensor) THSTensor_flatten(const Tensor tensor, const int64_t start, const int64_t end);

EXPORT_API(Tensor) THSTensor_flip(const Tensor tensor, const int64_t* sizes, const int length);

EXPORT_API(Tensor) THSTensor_fliplr(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_flipud(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_float_power(const Tensor input, const Tensor exponent);

EXPORT_API(Tensor) THSTensor_floor(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_floor_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_floor_divide(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_floor_divide_scalar(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_floor_divide_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_floor_divide_scalar_(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_true_divide(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_true_divide_scalar(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_true_divide_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_true_divide_scalar_(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_frac(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_frac_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_fmax(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_fmin(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_fmod(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_fmod_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_fmod_scalar(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_fmod_scalar_(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_frexp(const Tensor tensor, Tensor* exponent);

EXPORT_API(Tensor) THSTensor_full(const int64_t* sizes, const int length, Scalar value, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_full_out(const int64_t* sizes, const int length, Scalar value, const Tensor out);

EXPORT_API(Tensor) THSTensor_full_like(const Tensor input, Scalar value, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_digamma(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_digamma_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_lgamma(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_lgamma_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_mvlgamma(const Tensor tensor, int64_t p);

EXPORT_API(Tensor) THSTensor_mvlgamma_(const Tensor tensor, int64_t p);

EXPORT_API(Tensor) THSTensor_gather(const Tensor tensor, const int64_t dim, const Tensor index);

EXPORT_API(Tensor) THSTensor_ge(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_ge_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_ge_scalar(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_ge_scalar_(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_gelu(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_get1(const Tensor tensor, int64_t index);

EXPORT_API(Tensor) THSTensor_get2(const Tensor tensor, int64_t index1, int64_t index2);

EXPORT_API(Tensor) THSTensor_get3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3);

EXPORT_API(Tensor) THSTensor_get4(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4);

EXPORT_API(Tensor) THSTensor_get5(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5);

EXPORT_API(Tensor) THSTensor_get6(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, int64_t index6);

EXPORT_API(Tensor) THSTensor_gcd(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_gcd_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_grad(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_gt(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_gt_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_gt_scalar(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_gt_scalar_(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_hardtanh(const Tensor tensor, Scalar min, Scalar max);

EXPORT_API(Tensor) THSTensor_hardtanh_(const Tensor tensor, Scalar min, Scalar max);

EXPORT_API(Tensor) THSTensor_heaviside(const Tensor input, const Tensor values);

EXPORT_API(Tensor) THSTensor_hypot(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_i0(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_igamma(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_igammac(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_hardsigmoid(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_hardsigmoid_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_hardswish(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_hardswish_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_histc(const Tensor tensor, const int64_t bins, const int64_t min, const int64_t max);

EXPORT_API(Tensor) THSTensor_imag(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_index_add(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source, const Scalar alpha);
EXPORT_API(Tensor) THSTensor_index_add_(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source, const Scalar alpha);

EXPORT_API(Tensor) THSTensor_index_copy(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source);
EXPORT_API(Tensor) THSTensor_index_copy_(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source);

EXPORT_API(Tensor) THSTensor_index_fill(const Tensor tensor, const int64_t dim, const Tensor index, const Scalar value);
EXPORT_API(Tensor) THSTensor_index_fill_(const Tensor tensor, const int64_t dim, const Tensor index, const Scalar value);

EXPORT_API(Tensor) THSTensor_indices(Tensor tensor);

EXPORT_API(Tensor) THSTensor_index(Tensor tensor,
    const int64_t* indexStarts,
    const int64_t* indexEnds,
    const int64_t* indexSteps,
    const Tensor* indexTensors,
    const int indicesLength);

EXPORT_API(Tensor) THSTensor_index_put_scalar_(Tensor tensor,
    const int64_t* indexStarts,
    const int64_t* indexEnds,
    const int64_t* indexSteps,
    const Tensor* indexTensors,
    const int indicesLength,
    const Scalar value);

EXPORT_API(Tensor) THSTensor_index_put_(Tensor tensor,
    const int64_t* indexStarts,
    const int64_t* indexEnds,
    const int64_t* indexSteps,
    const Tensor* indexTensors,
    const int indicesLength,
    const Tensor value);

EXPORT_API(Tensor) THSTensor_index_select(Tensor tensor, int64_t dim, Tensor index);

EXPORT_API(Tensor) THSTensor_inner(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_inverse(const Tensor tensor);

EXPORT_API(int) THSTensor_is_contiguous(const Tensor input);

EXPORT_API(int64_t) THSTensor_is_leaf(const Tensor tensor);

EXPORT_API(int) THSTensor_is_sparse(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_isclose(const Tensor tensor, const Tensor other, const double rtol, const double atol, const bool equal_nan);

EXPORT_API(Tensor) THSTensor_isin(const Tensor elements, const Tensor test_elements, bool assume_unique, bool invert);

EXPORT_API(Tensor) THSTensor_isinf(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_isfinite(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_isposinf(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_isneginf(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_isnan(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_isreal(const Tensor tensor);

EXPORT_API(Scalar) THSTensor_item(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_kron(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_kthvalue(const Tensor input, int64_t k, int64_t dim, bool keepdim, Tensor* out);

EXPORT_API(Tensor) THSTensor_lcm(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_lcm_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_ldexp(const Tensor left, const Tensor right);
EXPORT_API(Tensor) THSTensor_ldexp_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_le(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_le_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_le_scalar(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_le_scalar_(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_leaky_relu(const Tensor tensor, const Scalar negval);

EXPORT_API(Tensor) THSTensor_leaky_relu_(const Tensor tensor, const Scalar negval);

EXPORT_API(Tensor) THSTensor_lerp(const Tensor tensor, const Tensor end, const Tensor weight);

EXPORT_API(Tensor) THSTensor_lerp_(const Tensor tensor, const Tensor end, const Tensor weight);

EXPORT_API(Tensor) THSTensor_linspace(const double start, const double end, const int64_t steps, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_logspace(const double start, const double end, const int64_t steps, double base, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_load(const char* location);

EXPORT_API(Tensor) THSTensor_log(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log10(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log10_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log1p(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log1p_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log2(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log2_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log_sigmoid(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_logaddexp(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_logaddexp2(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_logcumsumexp(const Tensor tensor, const long dimension);

EXPORT_API(Tensor) THSTensor_logsumexp(const Tensor tensor, const long dimension, const bool keepdim);

EXPORT_API(Tensor) THSTensor_log1p(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_log1p_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_logical_and(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_logical_and_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_logical_not(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_logical_not_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_logical_or(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_logical_or_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_logical_xor(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_logical_xor_(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_logit(const Tensor tensor, const double* eps);
EXPORT_API(Tensor) THSTensor_logit_(const Tensor tensor, const double* eps);

EXPORT_API(Tensor) THSTensor_lt(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_lt_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_lt_scalar(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_lt_scalar_(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_logical_not(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_lu(const Tensor tensor, bool pivot, bool get_infos, Tensor* infos, Tensor* pivots);

EXPORT_API(Tensor) THSTensor_lu_solve(const Tensor tensor, const Tensor LU_data, const Tensor LU_pivots);

EXPORT_API(Tensor) THSTensor_lu_unpack(const Tensor LU_data, const Tensor LU_pivots, bool unpack_data, bool unpack_pivots, Tensor* L, Tensor* U);

EXPORT_API(Tensor) THSTensor_masked_fill(const Tensor tensor, const Tensor mask, const Scalar value);

EXPORT_API(Tensor) THSTensor_masked_fill_(const Tensor tensor, const Tensor mask, const Scalar value);

EXPORT_API(Tensor) THSTensor_masked_scatter(const Tensor tensor, const Tensor mask, const Tensor source);

EXPORT_API(Tensor) THSTensor_masked_scatter_(const Tensor tensor, const Tensor mask, const Tensor source);

EXPORT_API(Tensor) THSTensor_masked_select(const Tensor tensor, const Tensor mask);

EXPORT_API(Tensor) THSTensor_matmul(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_matrix_exp(const Tensor input);

EXPORT_API(Tensor) THSTensor_max(const Tensor tensor);

EXPORT_API(void) THSTensor_max_along_dimension(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim, const bool keep_dim);

EXPORT_API(Tensor) THSTensor_max_elementwise(const Tensor tensor, const Tensor other);

EXPORT_API(Tensor) THSTensor_max_pool1d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode);

EXPORT_API(Tensor) THSTensor_max_pool2d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode);

EXPORT_API(Tensor) THSTensor_max_pool3d(
    const Tensor tensor,
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode);

EXPORT_API(void) THSTensor_max_pool1d_with_indices(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode);

EXPORT_API(void) THSTensor_max_pool2d_with_indices(
    const Tensor tensor,
    Tensor* (*allocator)(size_t length),
    const int64_t* kernelSize, const int kernelSizeLength,
    const int64_t* stride, const int strideLength,
    const int64_t* padding, const int paddingLength,
    const int64_t* dilation, const int dilationLength,
    bool ceil_mode);

EXPORT_API(void) THSTensor_max_pool3d_with_indices(
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

EXPORT_API(Tensor) THSTensor_mean_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool keepdim, bool has_type, const int8_t dtype);

EXPORT_API(Tensor) THSTensor_median(const Tensor tensor);

EXPORT_API(void) THSTensor_mode(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim, const bool keep_dim);

EXPORT_API(Tensor) THSTensor_min(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_min_elementwise(const Tensor tensor, const Tensor other);

EXPORT_API(void) THSTensor_min_along_dimension(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim, const bool keep_dim);

EXPORT_API(Tensor) THSTensor_mm(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_mv(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_movedim(const Tensor tensor, const int64_t* src, const int src_len, const int64_t* dst, const int dst_len);

EXPORT_API(Tensor) THSTensor_msort(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_mul(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_mul_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_mul_scalar(const Tensor tensor, const Scalar scalar);

EXPORT_API(Tensor) THSTensor_mul_scalar_(const Tensor tensor, const Scalar scalar);

EXPORT_API(Tensor) THSTensor_nanmean(const Tensor input, const int64_t* dims, const int dims_len, bool keepdim, int8_t scalar_type);

EXPORT_API(Tensor) THSTensor_nanmedian(const Tensor input);

EXPORT_API(Tensor) THSTensor_nanquantile(const Tensor tensor, const Tensor q, const int64_t dim, const bool keep_dim);

EXPORT_API(Tensor) THSTensor_nansum(const Tensor input);

EXPORT_API(Tensor) THSTensor_nan_to_num(const Tensor input, double* nan, double* posinf, double* neginf);

EXPORT_API(Tensor) THSTensor_narrow(const Tensor tensor, int64_t dim, int64_t start, int64_t length);

EXPORT_API(int64_t) THSTensor_ndimension(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_ne(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_ne_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_ne_scalar(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_ne_scalar_(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_neg(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_neg_(const Tensor tensor);

EXPORT_API(int64_t) THSTensor_is_neg(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_resolve_neg(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_new(
    void* data,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    int8_t scalar_type,
    const bool requires_grad);

EXPORT_API(Tensor) THSTensor_frombuffer(
    void* data,
    void (*deleter)(void*),
    const int64_t count,
    const ptrdiff_t offset,
    int8_t scalar_type,
    const bool requires_grad);

EXPORT_API(Tensor) THSTensor_newInt64(
    int64_t* data,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    const bool requires_grad);

EXPORT_API(Tensor) THSTensor_newFloat16(
    float* rawArray,
    c10::Half* dataArray,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    const bool requires_grad);

EXPORT_API(Tensor) THSTensor_newBFloat16(
    float* rawArray,
    c10::BFloat16* dataArray,
    void (*deleter)(void*),
    const int64_t* sizes,
    const int szlength,
    const bool requires_grad);

EXPORT_API(Tensor) THSTensor_newInt8Scalar(int8_t data, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newByteScalar(char data, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newBoolScalar(bool data, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newFloat16Scalar(float data, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newBFloat16Scalar(float data, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newInt16Scalar(short data, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newInt32Scalar(int data, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newInt64Scalar(int64_t data, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newFloat32Scalar(float data, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newFloat64Scalar(double data, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newComplexFloat32Scalar(float real, float imaginary, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_newComplexFloat64Scalar(double real, double imaginary, const int device_type, const int device_index, bool requires_grad);

EXPORT_API(Tensor) THSTensor_nextafter(const Tensor input, const Tensor other);

EXPORT_API(Tensor) THSTensor_nonzero(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_norm(const Tensor tensor, float p);

EXPORT_API(Tensor) THSTensor_norm_along_dimension(const Tensor tensor, const int64_t dim, const bool keepdim, float p);

EXPORT_API(Tensor) THSLinalg_tensordot(const Tensor input1, const Tensor input2, const int64_t* dims1, const int dims1_length, const int64_t* dims2, const int dims2_length);

EXPORT_API(int64_t) THSTensor_numel(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_ones(const int64_t* sizes, const int length, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_ones_out(const int64_t* sizes, const int length, const Tensor out);

EXPORT_API(Tensor) THSTensor_ones_like(const Tensor input, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_ormqr(const Tensor input, const Tensor tau, const Tensor other, bool left, bool transpose);

EXPORT_API(Tensor) THSTensor_outer(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_mT(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_mH(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_permute(const Tensor tensor, const int64_t* sizes, const int length);

EXPORT_API(Tensor) THSTensor_polar(const Tensor abs, const Tensor angle);

EXPORT_API(Tensor) THSTensor_polygamma(const Tensor tensor, int64_t n);

EXPORT_API(Tensor) THSTensor_polygamma_(const Tensor tensor, int64_t n);

EXPORT_API(Tensor) THSTensor_positive(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_pow(const Tensor tensor, const Tensor exponent);

EXPORT_API(Tensor) THSTensor_pow_(const Tensor tensor, const Tensor exponent);

EXPORT_API(Tensor) THSTensor_pow_scalar(const Tensor tensor, const Scalar scalar);

EXPORT_API(Tensor) THSTensor_pow_scalar_(const Tensor tensor, const Scalar scalar);

EXPORT_API(Tensor) THSTensor_prelu(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_quantile(const Tensor tensor, const Tensor q, const int64_t dim, const bool keep_dim);

EXPORT_API(Tensor) THSTensor_rad2deg(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_rand(const Generator gen, const int64_t* sizes, const int length, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_rand_out(const Generator gen, const int64_t* sizes, const int length, const Tensor out);

EXPORT_API(Tensor) THSTensor_rand_like(const Tensor input, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_randint(const Generator gen, const int64_t low, const int64_t high, const int64_t* sizes, const int length, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_randint_out(const Generator gen, const int64_t low, const int64_t high, const int64_t* sizes, const int length, const Tensor out);

EXPORT_API(Tensor) THSTensor_randint_like(const Tensor input, const int64_t low, const int64_t high, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_randn(const Generator gen, const int64_t* sizes, const int length, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_randn_out(const Generator gen, const int64_t* sizes, const int length, const Tensor out);

EXPORT_API(Tensor) THSTensor_randn_like(const Tensor input, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_randperm(const Generator gen, const int64_t n, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_randperm_out(const Generator gen, const int64_t n, const Tensor out);

EXPORT_API(Tensor) THSTensor_from_file(const char* filename, const int8_t shared, const int64_t size, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_ravel(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_real(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_reciprocal(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_reciprocal_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_relu(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_relu_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_relu6(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_relu6_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_repeat(const Tensor tensor, const int64_t* sizes, const int length);

EXPORT_API(Tensor) THSTensor_repeat_interleave(const Tensor tensor, const Tensor repeats, const int64_t dim, const int64_t output_size);

EXPORT_API(Tensor) THSTensor_repeat_interleave_int64(const Tensor tensor, const int64_t repeats, const int64_t dim, const int64_t output_size);

EXPORT_API(int) THSTensor_requires_grad(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_reshape(const Tensor tensor, const int64_t* shape, const int length);

EXPORT_API(Tensor) THSTensor_roll(const Tensor tensor, const int64_t* shifts, const int shLength, const int64_t* dims, const int dimLength);

EXPORT_API(Tensor) THSTensor_rot90(const Tensor tensor, const int64_t k, const int64_t dim1, const int64_t dim2);

EXPORT_API(Tensor) THSTensor_round(const Tensor tensor, const int64_t decimals);
EXPORT_API(Tensor) THSTensor_round_(const Tensor tensor, const int64_t decimals);

EXPORT_API(Tensor) THSTensor_remainder(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_remainder_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_remainder_scalar(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_remainder_scalar_(const Tensor left, const Scalar right);

EXPORT_API(void) THSTensor_retain_grad(const Tensor tensor);

EXPORT_API(int) THSTensor_result_type(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_rsqrt(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_rsqrt_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_renorm(const Tensor tensor, const float p, const int64_t dim, const float maxnorm);

EXPORT_API(Tensor) THSTensor_select(Tensor tensor, int64_t dim, int64_t index);

EXPORT_API(Tensor) THSTensor_selu(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_selu_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sigmoid(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sigmoid_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sign(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sign_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sgn(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sgn_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_signbit(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_silu(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_silu_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sin(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sin_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sinc(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sinc_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sinh(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sinh_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_softplus(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sort(const Tensor tensor, const int64_t dim, const bool descending, const bool stable, Tensor* indices);

EXPORT_API(Tensor) THSTensor_sqrt(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_sqrt_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_std(const Tensor tensor, const bool unbiased);

EXPORT_API(Tensor) THSTensor_std_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool unbiased, bool keepdim);

EXPORT_API(Tensor) THSTensor_std_mean(const Tensor tensor, bool unbiased, Tensor* mean);

EXPORT_API(Tensor) THSTensor_std_mean_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool unbiased, bool keepdim, Tensor* mean);

EXPORT_API(Tensor) THSTensor_var(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_var_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool unbiased, bool keepdim);

EXPORT_API(Tensor) THSTensor_var_mean(const Tensor tensor, bool unbiased, Tensor* var);

EXPORT_API(Tensor) THSTensor_var_mean_along_dimensions(const Tensor tensor, const int64_t* dimensions, int length, bool unbiased, bool keepdim, Tensor* mean);

EXPORT_API(Tensor) THSTensor_sub(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_sub_(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_sub_scalar(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_sub_scalar_(const Tensor left, const Scalar right);

EXPORT_API(Tensor) THSTensor_sum(const Tensor tensor, bool has_type, const int8_t dtype);

EXPORT_API(Tensor) THSTensor_sum_along_dimensions(const Tensor tensor, const int64_t * dimensions, int length, bool keepdim, bool has_type, const int8_t dtype);

EXPORT_API(void) THSTensor_save(const Tensor tensor, const char* location);

EXPORT_API(Tensor) THSTensor_scatter(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source);
EXPORT_API(Tensor) THSTensor_scatter_(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source);

EXPORT_API(Tensor) THSTensor_diagonal_scatter(const Tensor tensor, const Tensor source, const int64_t offset, const int64_t dim1, const int64_t dim2);
EXPORT_API(Tensor) THSTensor_select_scatter(const Tensor tensor, const Tensor source, const int64_t dim, const int64_t index);
EXPORT_API(Tensor) THSTensor_slice_scatter(const Tensor tensor, const Tensor source, const int64_t dim, const int64_t* start, const int64_t* end, const int64_t step);

EXPORT_API(Tensor) THSTensor_scatter_add(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source);
EXPORT_API(Tensor) THSTensor_scatter_add_(const Tensor tensor, const int64_t dim, const Tensor index, const Tensor source);

EXPORT_API(Tensor) THSTensor_searchsorted_t(const Tensor sorted_sequence, const Tensor values, const bool out_int32, const bool right, const Tensor sorter);
EXPORT_API(Tensor) THSTensor_searchsorted_s(const Tensor sorted_sequence, const Scalar values, const bool out_int32, const bool right, const Tensor sorter);

EXPORT_API(Tensor) THSTensor_histogram_t(const Tensor input, const Tensor bins, const Tensor weight, const bool density, Tensor* r_bin_edges);
EXPORT_API(Tensor) THSTensor_histogram_i(const Tensor input, const int64_t bins, const double* range, const int length, const Tensor weight, const bool density, Tensor* r_bin_edges);
EXPORT_API(Tensor) THSTensor_histogram_out_t(const Tensor input, const Tensor bins, const Tensor weight, const bool density, Tensor* hist, Tensor* bin_edges, Tensor* r_bin_edges);
EXPORT_API(Tensor) THSTensor_histogram_out_i(const Tensor input, const int64_t bins, const double* range, const int length, const Tensor weight, const bool density, Tensor* hist, Tensor* bin_edges, Tensor* r_bin_edges);

EXPORT_API(Tensor) THSTensor_set_(Tensor tensor, const Tensor source);

EXPORT_API(Tensor) THSTensor_set_requires_grad(const Tensor tensor, const bool requires_grad);

EXPORT_API(void) THSTensor_set1(const Tensor tensor, int64_t index, const Tensor value);

EXPORT_API(void) THSTensor_set2(const Tensor tensor, int64_t index1, int64_t index2, const Tensor value);

EXPORT_API(void) THSTensor_set3(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, const Tensor value);

EXPORT_API(void) THSTensor_set4(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, const Tensor value);

EXPORT_API(void) THSTensor_set5(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, const Tensor value);

EXPORT_API(void) THSTensor_set6(const Tensor tensor, int64_t index1, int64_t index2, int64_t index3, int64_t index4, int64_t index5, int64_t index6, const Tensor value);

EXPORT_API(int64_t) THSTensor_size(const Tensor tensor, const int64_t dim);

EXPORT_API(void) THSTensor_sizes(const Tensor tensor, int64_t* (*allocator)(size_t length));

EXPORT_API(Tensor) THSTensor_slice(const Tensor tensor, int64_t dim, int64_t start, int64_t finish, int64_t step);

EXPORT_API(Tensor) THSTensor_sparse(
    Tensor indices,
    Tensor values,
    const int64_t* sizes,
    const int length,
    const int8_t scalar_type,
    const int device_type, const int device_index,
    const bool requires_grad);

EXPORT_API(void) THSTensor_split_with_size(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t split_size, const int64_t dim);
EXPORT_API(void) THSTensor_split_with_sizes(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t* sizes, const int length, const int64_t dim);

EXPORT_API(Tensor) THSTensor_squeeze(Tensor tensor, int64_t dim);
EXPORT_API(Tensor) THSTensor_squeeze_no_dim(Tensor tensor);
EXPORT_API(Tensor) THSTensor_squeeze_(Tensor tensor, int64_t dim);
EXPORT_API(Tensor) THSTensor_squeeze_no_dim_(Tensor tensor);

EXPORT_API(Tensor) THSTensor_stack(const Tensor* tensor, const int length, const int64_t dim);

EXPORT_API(Tensor) THSTensor_hstack(const Tensor* tensor, const int length);

EXPORT_API(Tensor) THSTensor_vstack(const Tensor* tensor, const int length);

EXPORT_API(Tensor) THSTensor_dstack(const Tensor* tensor, const int length);

EXPORT_API(Tensor) THSTensor_column_stack(const Tensor* tensor, const int length);

EXPORT_API(Tensor) THSTensor_row_stack(const Tensor* tensor, const int length);

EXPORT_API(void) THSTensor_meshgrid(const Tensor* tensors, const int length, const char* indexing, Tensor* (*allocator)(size_t length));

EXPORT_API(int64_t) THSTensor_stride(const Tensor tensor, const int64_t dim);

EXPORT_API(void) THSTensor_strides(const Tensor tensor, int64_t* (*allocator)(size_t length));

EXPORT_API(Tensor) THSTensor_take(const Tensor tensor, const Tensor indices);

EXPORT_API(Tensor) THSTensor_take_along_dim_dflt(const Tensor tensor, const Tensor indices);
EXPORT_API(Tensor) THSTensor_take_along_dim(const Tensor tensor, const Tensor indices, const int64_t dim);

EXPORT_API(Tensor) THSTensor_tan(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_tan_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_tanh(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_tanh_(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_t(const Tensor tensor);

EXPORT_API(void) THSTensor_tensor_split_with_size(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t split_size, const int64_t dim);
EXPORT_API(void) THSTensor_tensor_split_with_sizes(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t* sizes, const int length, const int64_t dim);
EXPORT_API(void) THSTensor_tensor_split_with_tensor_sizes(const Tensor tensor, Tensor* (*allocator)(size_t length), const Tensor sizes, const int64_t dim);

EXPORT_API(void) THSTensor_vsplit_with_size(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t split_size);
EXPORT_API(void) THSTensor_vsplit_with_sizes(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t* sizes, const int length);

EXPORT_API(void) THSTensor_hsplit_with_size(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t split_size);
EXPORT_API(void) THSTensor_hsplit_with_sizes(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t* sizes, const int length);

EXPORT_API(void) THSTensor_dsplit_with_size(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t split_size);
EXPORT_API(void) THSTensor_dsplit_with_sizes(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t* sizes, const int length);

EXPORT_API(Tensor) THSTensor_tile(const Tensor tensor, const int64_t* rep, const int rep_length);

EXPORT_API(Tensor) THSTensor_tril(const Tensor tensor, const int64_t diagonal);

EXPORT_API(Tensor) THSTensor_triu(const Tensor tensor, const int64_t diagonal);

EXPORT_API(Tensor) THSTensor_tril_indices(const int64_t row, const int64_t col, const int64_t offset, const int8_t scalar_type, const int device_type, const int device_index);
EXPORT_API(Tensor) THSTensor_triu_indices(const int64_t row, const int64_t col, const int64_t offset, const int8_t scalar_type, const int device_type, const int device_index);

EXPORT_API(Tensor) THSTensor_transpose(const Tensor tensor, const int64_t dim1, const int64_t dim2);

EXPORT_API(Tensor) THSTensor_transpose_(const Tensor tensor, const int64_t dim1, const int64_t dim2);

EXPORT_API(Tensor) THSTensor_cumulative_trapezoid_x(const Tensor y, const Tensor x, int64_t dim);
EXPORT_API(Tensor) THSTensor_cumulative_trapezoid_dx(const Tensor y, const double dx, int64_t dim);

EXPORT_API(Tensor) THSTensor_trapezoid_x(const Tensor y, const Tensor x, int64_t dim);
EXPORT_API(Tensor) THSTensor_trapezoid_dx(const Tensor y, const double dx, int64_t dim);

EXPORT_API(Tensor) THSTensor_cumulative_trapezoid_x(const Tensor y, const Tensor x, int64_t dim);
EXPORT_API(Tensor) THSTensor_cumulative_trapezoid_dx(const Tensor y, const double dx, int64_t dim);

EXPORT_API(Tensor) THSTensor_to_dense(Tensor tensor);

EXPORT_API(Tensor) THSTensor_to_device(const Tensor tensor, const int device_type, const int device_index, const bool copy);

EXPORT_API(Tensor) THSTensor_to_type(const Tensor tensor, int8_t scalar_type, const bool copy);

EXPORT_API(Tensor) THSTensor_to_type_and_device(const Tensor tensor, int8_t scalar_type, const int device_type, const int device_index, const bool copy);

EXPORT_API(void) THSTensor_topk(const Tensor tensor, Tensor* (*allocator)(size_t length), const int k, const int64_t dim, const bool largest, const bool sorted);

EXPORT_API(Tensor) THSTensor_trunc(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_trunc_(const Tensor tensor);

EXPORT_API(int8_t) THSTensor_type(const Tensor tensor);

EXPORT_API(void) THSTensor_unbind(const Tensor tensor, Tensor* (*allocator)(size_t length), const int64_t dim);

EXPORT_API(Tensor) THSTensor_unique(const Tensor tensor, const bool sorted, const bool return_inverse, const bool return_counts, Tensor* inverse_indices, Tensor* counts);
EXPORT_API(Tensor) THSTensor_unique_dim(const Tensor tensor, const int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts, Tensor* inverse_indices, Tensor* counts);
EXPORT_API(Tensor) THSTensor_unique_consecutive(const Tensor tensor, const bool return_inverse, const bool return_counts, Tensor* inverse_indices, Tensor* counts);
EXPORT_API(Tensor) THSTensor_unique_dim_consecutive(const Tensor tensor, const int64_t dim, const bool return_inverse, const bool return_counts, Tensor* inverse_indices, Tensor* counts);

EXPORT_API(Tensor) THSTensor_unflatten(const Tensor tensor, const int64_t dimension, const int64_t* shape, const int length);

EXPORT_API(Tensor) THSTensor_unfold(const Tensor tensor, const int64_t dimension, const int64_t size, const int64_t step);

EXPORT_API(Tensor) THSTensor_unsqueeze(Tensor tensor, int64_t dim);
EXPORT_API(Tensor) THSTensor_unsqueeze_(Tensor tensor, int64_t dim);

EXPORT_API(Tensor) THSTensor_upsample_nearest1d(
    const Tensor tensor,
    const int64_t* outputSize, const int outputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength);

EXPORT_API(Tensor) THSTensor_upsample_nearest1d_backward(
    const Tensor grad_output,
    const int64_t* outputSize, const int outputSizeLength,
    const int64_t* inputSize, const int inputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength);

EXPORT_API(Tensor) THSTensor_upsample_nearest2d(
    const Tensor tensor,
    const int64_t* outputSize, const int outputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength);

EXPORT_API(Tensor) THSTensor_upsample_nearest2d_backward(
    const Tensor grad_output,
    const int64_t* outputSize, const int outputSizeLength,
    const int64_t* inputSize, const int inputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength);

EXPORT_API(Tensor) THSTensor_upsample_nearest3d(
    const Tensor tensor,
    const int64_t* outputSize, const int outputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength);

EXPORT_API(Tensor) THSTensor_upsample_nearest3d_backward(
    const Tensor grad_output,
    const int64_t* outputSize, const int outputSizeLength,
    const int64_t* inputSize, const int inputSizeLength,
    const double* scaleFactors, const int scaleFactorsLength);

EXPORT_API(Tensor) THSTensor_values(Tensor tensor);

EXPORT_API(bool) THSTensor_has_names(Tensor tensor);
EXPORT_API(void) THSTensor_names(Tensor tensor, const char** (*allocator)(size_t length));
EXPORT_API(Tensor) THSTensor_rename(Tensor tensor, const char** names, int64_t nLength);
EXPORT_API(Tensor) THSTensor_rename_(Tensor tensor, const char** names, int64_t nLength);

EXPORT_API(Tensor) THSTensor_refine_names(Tensor tensor, const char** names, int64_t nLength);

EXPORT_API(Tensor) THSTensor_flatten_names(Tensor tensor, const char** names, int64_t nLength);
EXPORT_API(Tensor) THSTensor_unflatten_names(Tensor tensor, const char** names, const int64_t* sizes, int64_t nLength);

EXPORT_API(Tensor) THSTensor_align_to(Tensor tensor, const char** names, int64_t nLength);

EXPORT_API(Tensor) THSTensor_vander(const Tensor tensor, const int64_t N, const bool increasing);

EXPORT_API(Tensor) THSTensor_vdot(const Tensor left, const Tensor right);

EXPORT_API(Tensor) THSTensor_view(const Tensor tensor, const int64_t* shape, const int length);

EXPORT_API(Tensor) THSTensor_view_as_complex(const Tensor tensor);
EXPORT_API(Tensor) THSTensor_view_as_real(const Tensor tensor);

EXPORT_API(Tensor) THSTensor_where(const Tensor condition, const Tensor x, const Tensor y);
EXPORT_API(void) THSTensor_where_list(const Tensor condition, Tensor* (*allocator)(size_t length));

EXPORT_API(Tensor) THSTensor_xlogy(const Tensor x, const Tensor y);
EXPORT_API(Tensor) THSTensor_xlogy_(const Tensor x, const Tensor y);

EXPORT_API(Tensor) THSTensor_xlogy_scalar(const Tensor x, const Scalar y);
EXPORT_API(Tensor) THSTensor_xlogy_scalar_(const Tensor x, const Scalar y);

EXPORT_API(Tensor) THSTensor_zeros(const int64_t* sizes, const int length, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_zeros_out(const int64_t* sizes, const int length, const Tensor out);

EXPORT_API(Tensor) THSTensor_zeros_like(const Tensor input, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

// Random numbers:

EXPORT_API(Tensor) THSTensor_bernoulli(const Tensor tensor, const Generator gen);

EXPORT_API(Tensor) THSTensor_binomial(const Tensor count, const Tensor prob, const Generator gen);

EXPORT_API(Tensor) THSTensor_multinomial(const Tensor tensor, const int64_t num_samples, const bool replacement, const Generator gen);

EXPORT_API(Tensor) THSTensor_poisson(const Tensor tensor, const Generator gen);

EXPORT_API(Tensor) THSTensor_sample_dirichlet_(Tensor tensor, const Generator gen);

EXPORT_API(Tensor) THSTensor_standard_gamma_(Tensor tensor, const Generator gen);


// Random number in-place:

EXPORT_API(Tensor) THSTensor_bernoulli_0(Tensor tensor, const double p, const Generator gen);

EXPORT_API(Tensor) THSTensor_bernoulli_1(Tensor tensor, const Tensor p_tensor, const Generator gen);

EXPORT_API(Tensor) THSTensor_cauchy_(Tensor tensor, const double median, const double sigma, const Generator gen);

EXPORT_API(Tensor) THSTensor_exponential_(Tensor tensor, const double lambda, const Generator gen);

EXPORT_API(Tensor) THSTensor_geometric_(Tensor tensor, const double p, const Generator gen);

EXPORT_API(Tensor) THSTensor_normal_(Tensor tensor, double mean, double std, const Generator gen);

EXPORT_API(Tensor) THSTensor_log_normal_(Tensor tensor, double mean, double std, const Generator gen);

EXPORT_API(Tensor) THSTensor_random_(Tensor tensor, double low, double high, const Generator gen);

EXPORT_API(Tensor) THSTensor_uniform_(Tensor tensor, double low, double high, const Generator gen);


// torch.linalg:

EXPORT_API(Tensor) THSLinalg_cond_int(const Tensor tensor, const int p);
EXPORT_API(Tensor) THSLinalg_cond_float(const Tensor tensor, const double p);
EXPORT_API(Tensor) THSLinalg_cond_str(const Tensor tensor, const char *p);
EXPORT_API(Tensor) THSLinalg_cond_none(const Tensor tensor);

EXPORT_API(Tensor) THSLinalg_cholesky(const Tensor tensor);
EXPORT_API(Tensor) THSLinalg_cholesky_ex(const Tensor tensor, bool check_errors, Tensor* info);

EXPORT_API(Tensor) THSLinalg_cross(const Tensor input, const Tensor other, const int64_t dim);

EXPORT_API(Tensor) THSLinalg_det(const Tensor tensor);
EXPORT_API(Tensor) THSTensor_logdet(const Tensor tensor);

EXPORT_API(Tensor) THSLinalg_slogdet(const Tensor tensor, Tensor *logabsdet);

EXPORT_API(Tensor) THSLinalg_eig(const Tensor tensor, Tensor* eigenvectors);
EXPORT_API(Tensor) THSLinalg_eigh(const Tensor tensor, const char UPLO, Tensor* eigenvectors);

EXPORT_API(Tensor) THSTensor_eig(const Tensor tensor, bool vectors, Tensor* eigenvectors);

EXPORT_API(Tensor) THSLinalg_eigvals(const Tensor tensor);
EXPORT_API(Tensor) THSLinalg_eigvalsh(const Tensor tensor, const char UPLO);

EXPORT_API(Tensor) THSTensor_geqrf(const Tensor tensor, Tensor* tau);

EXPORT_API(Tensor) THSLinalg_householder_product(const Tensor tensor, const Tensor tau);

EXPORT_API(Tensor) THSLinalg_inv(const Tensor tensor);
EXPORT_API(Tensor) THSLinalg_inv_ex(const Tensor tensor, bool check_errors, Tensor* info);

EXPORT_API(Tensor) THSLinalg_lstsq_none(const Tensor A, const Tensor B, Tensor* residuals, Tensor* rank, Tensor* singular_values);
EXPORT_API(Tensor) THSLinalg_lstsq_rcond(const Tensor A, const Tensor B, const double rcond, Tensor* residuals, Tensor* rank, Tensor* singular_values);

EXPORT_API(Tensor) THSLinalg_lu(const Tensor A, const bool pivot, Tensor* L, Tensor* U);
EXPORT_API(Tensor) THSLinalg_lu_factor(const Tensor A, const bool pivot, Tensor* pivots);

EXPORT_API(Tensor) THSLinalg_ldl_factor(const Tensor A, const bool hermitian, Tensor* pivots);
EXPORT_API(Tensor) THSLinalg_ldl_factor_ex(const Tensor A, const bool hermitian, const bool check_errors, Tensor* pivots, Tensor* info);
EXPORT_API(Tensor) THSLinalg_ldl_solve(const Tensor LD, const Tensor pivots, const Tensor B, const bool hermitian);

EXPORT_API(Tensor) THSLinalg_matrix_power(const Tensor target, const int64_t n);

EXPORT_API(Tensor) THSLinalg_matrix_norm(const Tensor tensor, const Scalar ord, const int64_t* dim, const int dim_length, const bool keepdim);
EXPORT_API(Tensor) THSLinalg_matrix_norm_fronuc(const Tensor tensor, const int8_t fronuc, const int64_t* dim, const int dim_length, const bool keepdim);

EXPORT_API(Tensor) THSLinalg_matrix_rank(const Tensor tensor, const double atol, const bool has_atol, const double rtol, const bool has_rtol, const bool hermitian);
EXPORT_API(Tensor) THSLinalg_matrix_rank_tensor(const Tensor tensor, const Tensor atol, const Tensor rtol, const bool hermitian);

EXPORT_API(Tensor) THSLinalg_multi_dot(const Tensor* tensors, const int length);

EXPORT_API(Tensor) THSLinalg_norm_str(const Tensor tensor, const char* p, const int64_t* dim, const int dim_length, const bool keepdim);
EXPORT_API(Tensor) THSLinalg_norm_float(const Tensor tensor, const double p, const int64_t* dim, const int dim_length, const bool keepdim);
EXPORT_API(Tensor) THSLinalg_norm_int(const Tensor tensor, const int p, const int64_t* dim, const int dim_length, const bool keepdim);
EXPORT_API(Tensor) THSLinalg_norm_opt(const Tensor tensor, const int64_t* dim, const int dim_length, const bool keepdim);

EXPORT_API(Tensor) THSLinalg_pinverse(const Tensor tensor, const double rcond, const bool hermitian);

EXPORT_API(Tensor) THSLinalg_pinv(const Tensor tensor, const double atol, const bool has_atol, const double rtol, const bool has_rtol, const bool hermitian);
EXPORT_API(Tensor) THSLinalg_pinv_tensor(const Tensor tensor, const Tensor atol, const Tensor rtol, const bool hermitian);

EXPORT_API(Tensor) THSLinalg_qr(const Tensor tensor, const char mode, Tensor* R);

EXPORT_API(Tensor) THSLinalg_solve(const Tensor tensor, Tensor other, bool left);
EXPORT_API(Tensor) THSLinalg_solve_ex(const Tensor tensor, Tensor other, bool left, bool check_errors, Tensor* S);

EXPORT_API(Tensor) THSLinalg_svd(const Tensor tensor, const bool full_matrices, Tensor* S, Tensor* Vh);

EXPORT_API(Tensor) THSLinalg_svdvals(const Tensor tensor);

EXPORT_API(Tensor) THSLinalg_tensorinv(const Tensor tensor, const int64_t ind);

EXPORT_API(Tensor) THSLinalg_tensorsolve(const Tensor tensor, Tensor other, const int64_t* dim, const int dim_length);

EXPORT_API(Tensor) THSLinalg_vector_norm(const Tensor tensor, const Scalar ord, const int64_t* dim, const int dim_length, const bool keepdim);

EXPORT_API(Tensor) THSLinalg_vander(const Tensor tensor, const int64_t N);

EXPORT_API(Tensor) THSLinalg_vecdot(const Tensor x, const Tensor y, const int64_t dim, Tensor out);

EXPORT_API(Tensor) THSLinalg_lu_solve(const Tensor B, const Tensor LU, const Tensor pivots, bool left, bool adjoint, Tensor out);



// torch.special:

EXPORT_API(Tensor) THSSpecial_airy_ai(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_airy_ai_out(const Tensor tensor, Tensor out);

EXPORT_API(Tensor) THSSpecial_bessel_j0(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_bessel_j0_out(const Tensor tensor, Tensor out);
EXPORT_API(Tensor) THSSpecial_bessel_j1(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_bessel_j1_out(const Tensor tensor, Tensor out);
EXPORT_API(Tensor) THSSpecial_bessel_y0(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_bessel_y0_out(const Tensor tensor, Tensor out);
EXPORT_API(Tensor) THSSpecial_bessel_y1(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_bessel_y1_out(const Tensor tensor, Tensor out);

EXPORT_API(Tensor) THSSpecial_modified_bessel_i0(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_modified_bessel_i0_out(const Tensor tensor, Tensor out);
EXPORT_API(Tensor) THSSpecial_modified_bessel_i1(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_modified_bessel_i1_out(const Tensor tensor, Tensor out);
EXPORT_API(Tensor) THSSpecial_modified_bessel_k0(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_modified_bessel_k0_out(const Tensor tensor, Tensor out);
EXPORT_API(Tensor) THSSpecial_modified_bessel_k1(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_modified_bessel_k1_out(const Tensor tensor, Tensor out);

EXPORT_API(Tensor) THSSpecial_scaled_modified_bessel_k0(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_scaled_modified_bessel_k0_out(const Tensor tensor, Tensor out);
EXPORT_API(Tensor) THSSpecial_scaled_modified_bessel_k1(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_scaled_modified_bessel_k1_out(const Tensor tensor, Tensor out);

EXPORT_API(Tensor) THSSpecial_spherical_bessel_j0(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_spherical_bessel_j0_out(const Tensor tensor, Tensor out);

EXPORT_API(Tensor) THSSpecial_chebyshev_polynomial_t(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_chebyshev_polynomial_t_out(const Tensor x, const Tensor n, Tensor out);
EXPORT_API(Tensor) THSSpecial_chebyshev_polynomial_u(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_chebyshev_polynomial_u_out(const Tensor x, const Tensor n, Tensor out);
EXPORT_API(Tensor) THSSpecial_chebyshev_polynomial_v(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_chebyshev_polynomial_v_out(const Tensor x, const Tensor n, Tensor out);
EXPORT_API(Tensor) THSSpecial_chebyshev_polynomial_w(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_chebyshev_polynomial_w_out(const Tensor x, const Tensor n, Tensor out);

EXPORT_API(Tensor) THSSpecial_shifted_chebyshev_polynomial_t(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_shifted_chebyshev_polynomial_t_out(const Tensor x, const Tensor n, Tensor out);
EXPORT_API(Tensor) THSSpecial_shifted_chebyshev_polynomial_u(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_shifted_chebyshev_polynomial_u_out(const Tensor x, const Tensor n, Tensor out);
EXPORT_API(Tensor) THSSpecial_shifted_chebyshev_polynomial_v(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_shifted_chebyshev_polynomial_v_out(const Tensor x, const Tensor n, Tensor out);
EXPORT_API(Tensor) THSSpecial_shifted_chebyshev_polynomial_w(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_shifted_chebyshev_polynomial_w_out(const Tensor x, const Tensor n, Tensor out);

EXPORT_API(Tensor) THSSpecial_hermite_polynomial_h(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_hermite_polynomial_h_out(const Tensor x, const Tensor n, Tensor out);
EXPORT_API(Tensor) THSSpecial_hermite_polynomial_he(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_hermite_polynomial_he_out(const Tensor x, const Tensor n, Tensor out);

EXPORT_API(Tensor) THSSpecial_laguerre_polynomial_l(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_laguerre_polynomial_l_out(const Tensor x, const Tensor n, Tensor out);
EXPORT_API(Tensor) THSSpecial_legendre_polynomial_p(const Tensor x, const Tensor n);
EXPORT_API(Tensor) THSSpecial_legendre_polynomial_p_out(const Tensor x, const Tensor n, Tensor out);

EXPORT_API(Tensor) THSSpecial_entr(const Tensor tensor);

EXPORT_API(Tensor) THSSpecial_erf(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_erfc(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_erfcx(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_erfinv(const Tensor tensor);

EXPORT_API(Tensor) THSSpecial_expit(const Tensor tensor);

EXPORT_API(Tensor) THSSpecial_expm1(const Tensor tensor);

EXPORT_API(Tensor) THSSpecial_exp2(const Tensor tensor);

EXPORT_API(Tensor) THSSpecial_gammaln(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_gammainc(const Tensor input, const Tensor other);
EXPORT_API(Tensor) THSSpecial_gammaincc(const Tensor input, const Tensor other);

EXPORT_API(Tensor) THSSpecial_polygamma(const int64_t n, const Tensor tensor);

EXPORT_API(Tensor) THSSpecial_digamma(const Tensor tensor);

EXPORT_API(Tensor) THSSpecial_multigammaln(const Tensor tensor, const int64_t p);

EXPORT_API(Tensor) THSSpecial_i0(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_i0e(const Tensor tensor);

EXPORT_API(Tensor) THSSpecial_i1(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_i1e(const Tensor tensor);

EXPORT_API(Tensor) THSSpecial_logit(const Tensor tensor);

EXPORT_API(Tensor) THSSpecial_log_softmax(const Tensor tensor, int64_t dim, int8_t scalar_type);
EXPORT_API(Tensor) THSSpecial_softmax(const Tensor tensor, int64_t dim, int8_t scalar_type);

EXPORT_API(Tensor) THSSpecial_ndtr(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_ndtri(const Tensor tensor);
EXPORT_API(Tensor) THSSpecial_sinc(const Tensor tensor);

EXPORT_API(Tensor) THSSpecial_xlog1py(const Tensor input, const Tensor other);

EXPORT_API(Tensor) THSSpecial_zeta(const Tensor input, const Tensor other);


// torch.nn.init:

EXPORT_API(double) THSInit_calculate_gain(int64_t nonlinearity, double param);

EXPORT_API(Tensor) THSInit_constant_(Tensor tensor, Scalar value);

EXPORT_API(Tensor) THSInit_dirac_(Tensor tensor);

EXPORT_API(Tensor) THSInit_eye_(Tensor matrix);

EXPORT_API(Tensor) THSInit_normal_(Tensor tensor, double mean, double std);

EXPORT_API(Tensor) THSInit_trunc_normal_(Tensor tensor, double mean, double std, double a, double b);

EXPORT_API(Tensor) THSInit_ones_(Tensor tensor);

EXPORT_API(Tensor) THSInit_orthogonal_(Tensor tensor, double gain);

EXPORT_API(Tensor) THSInit_sparse_(Tensor tensor, double sparsity, double std);

EXPORT_API(Tensor) THSInit_uniform_(Tensor tensor, double low, double high);

EXPORT_API(Tensor) THSInit_kaiming_normal_(Tensor tensor, double a, const int64_t mode, const int64_t nonlinearity);

EXPORT_API(Tensor) THSInit_kaiming_uniform_(Tensor tensor, double a, const int64_t mode, const int64_t nonlinearity);

EXPORT_API(Tensor) THSInit_xavier_normal_(Tensor tensor, double gain);

EXPORT_API(Tensor) THSInit_xavier_uniform_(Tensor tensor, double gain);

EXPORT_API(Tensor) THSInit_zeros_(Tensor tensor);


// torch::fft:

EXPORT_API(Tensor) THSTensor_fft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm);

EXPORT_API(Tensor) THSTensor_ifft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm);

EXPORT_API(Tensor) THSTensor_hfft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm);

EXPORT_API(Tensor) THSTensor_ihfft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm);

EXPORT_API(Tensor) THSTensor_fft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm);

EXPORT_API(Tensor) THSTensor_ifft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm);

EXPORT_API(Tensor) THSTensor_hfft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm);
EXPORT_API(Tensor) THSTensor_ihfft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm);

EXPORT_API(Tensor) THSTensor_hfftn(const Tensor tensor, const int64_t* s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm);
EXPORT_API(Tensor) THSTensor_ihfftn(const Tensor tensor, const int64_t* s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm);

EXPORT_API(Tensor) THSTensor_fftn(const Tensor tensor, const int64_t *s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm);

EXPORT_API(Tensor) THSTensor_ifftn(const Tensor tensor, const int64_t* s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm);

EXPORT_API(Tensor) THSTensor_rfft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm);

EXPORT_API(Tensor) THSTensor_irfft(const Tensor tensor, const int64_t n, const int64_t dim, int8_t norm);

EXPORT_API(Tensor) THSTensor_rfft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm);

EXPORT_API(Tensor) THSTensor_irfft2(const Tensor tensor, const int64_t* s, const int64_t* dim, int8_t norm);

EXPORT_API(Tensor) THSTensor_rfftn(const Tensor tensor, const int64_t* s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm);

EXPORT_API(Tensor) THSTensor_irfftn(const Tensor tensor, const int64_t* s, const int s_length, const int64_t* dim, const int dim_length, int8_t norm);

EXPORT_API(Tensor) THSTensor_fftfreq(const int64_t n, const double d, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_rfftfreq(const int64_t n, const double d, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_fftshift(const Tensor tensor, const int64_t* dim, const int dim_length);

EXPORT_API(Tensor) THSTensor_ifftshift(const Tensor tensor, const int64_t* dim, const int dim_length);


// Spectral Ops

EXPORT_API(Tensor) THSTensor_bartlett_window(const int64_t len, bool periodic, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);
EXPORT_API(Tensor) THSTensor_blackman_window(const int64_t len, bool periodic, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);
EXPORT_API(Tensor) THSTensor_hamming_window(const int64_t len, bool periodic, double alpha, double beta, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);
EXPORT_API(Tensor) THSTensor_hann_window(const int64_t len, bool periodic, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);
EXPORT_API(Tensor) THSTensor_kaiser_window(const int64_t len, bool periodic, double beta, const int8_t scalar_type, const int device_type, const int device_index, const bool requires_grad);

EXPORT_API(Tensor) THSTensor_stft(const Tensor x, int64_t n_fft, int64_t hop_length, int64_t win_length, const Tensor window, bool normalized, int64_t onesided, bool return_complex);
EXPORT_API(Tensor) THSTensor_istft(const Tensor x, int64_t n_fft, int64_t hop_length, int64_t win_length, const Tensor window, bool center, bool normalized, int64_t onesided, int64_t length, bool return_complex);
