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

Tensor THSTensor_baddbmm(
    const Tensor batch1,
    const Tensor batch2,
    const Tensor mat,
    const float beta,
    const float alpha)
{
    CATCH_TENSOR(mat->baddbmm(*batch1, *batch2, beta, alpha));
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

Tensor THSTensor_ceil(const Tensor tensor)
{
    CATCH_TENSOR(tensor->ceil());
}

Tensor THSTensor_ceil_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->ceil_());
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

Tensor THSTensor_deg2rad(const Tensor tensor)
{
    CATCH_TENSOR(torch::deg2rad(*tensor));
}
Tensor THSTensor_div(const Tensor left, const Tensor right, const char* rounding_mode)
{
    CATCH_TENSOR(rounding_mode == nullptr ? left->div(*right) : left->div(*right, rounding_mode));
}

Tensor THSTensor_div_(const Tensor left, const Tensor right, const char* rounding_mode)
{
    CATCH_TENSOR(rounding_mode == nullptr ? left->div_(*right) : left->div_(*right, rounding_mode));
}

Tensor THSTensor_div_scalar(const Tensor left, const Scalar right, const char* rounding_mode)
{
    CATCH_TENSOR(rounding_mode == nullptr ? left->div(*right) : left->div(*right, rounding_mode));
}

Tensor THSTensor_div_scalar_(const Tensor left, const Scalar right, const char* rounding_mode)
{
    CATCH_TENSOR(rounding_mode == nullptr ? left->div_(*right) : left->div_(*right, rounding_mode));
}

Tensor THSTensor_dot(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->dot(*right));
}

Tensor THSTensor_einsum(const char* equation, const Tensor* tensors, const int length)
{
    CATCH_TENSOR(torch::einsum(equation, toTensors<at::Tensor>((torch::Tensor**)tensors, length)));
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

Tensor THSTensor_histc(const Tensor tensor, const int64_t bins, const int64_t min, const int64_t max)
{
    CATCH_TENSOR(tensor->histc(bins, min, max));
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

Tensor THSTensor_matmul(const Tensor left, const Tensor right)
{
    return  new torch::Tensor(left->matmul(*right));
}

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

Tensor THSTensor_ne(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->ne(*right));
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

Tensor THSTensor_rad2deg(const Tensor tensor)
{
    CATCH_TENSOR(torch::rad2deg(*tensor));
}

Tensor THSTensor_reciprocal(const Tensor tensor)
{
    CATCH_TENSOR(torch::reciprocal(*tensor));
}

Tensor THSTensor_reciprocal_(const Tensor tensor)
{
    CATCH_TENSOR(torch::reciprocal_(*tensor));
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

Tensor THSTensor_sqrt(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sqrt());
}

Tensor THSTensor_sqrt_(const Tensor tensor)
{
    CATCH_TENSOR(tensor->sqrt_());
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

Tensor THSTensor_sub(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->sub(*right));
}

Tensor THSTensor_sub_(const Tensor left, const Tensor right)
{
    CATCH_TENSOR(left->sub_(*right));
}

Tensor THSTensor_sub_scalar_(const Tensor left, const Scalar right)
{
    CATCH_TENSOR(left->sub_(*right));
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

Tensor THSTensor_xlogy(const Tensor x, const Tensor y)
{
    CATCH_TENSOR(x->xlogy(*y));
}

Tensor THSTensor_xlogy_(const Tensor x, const Tensor y)
{
    CATCH_TENSOR(x->xlogy_(*y));
}

Tensor THSTensor_xlogy_scalar(const Tensor x, const Scalar y)
{
    CATCH_TENSOR(x->xlogy(*y));
}

Tensor THSTensor_xlogy_scalar_(const Tensor x, const Scalar y)
{
    CATCH_TENSOR(x->xlogy_(*y));
}
