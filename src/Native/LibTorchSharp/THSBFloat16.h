// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"
#include "Utils.h"

#include "c10/util/BFloat16.h"
//#include "c10/util/BFloat16-inl.h"

EXPORT_API(c10::BFloat16) THSBFloat16_ctor(float value);
EXPORT_API(float) THSBFloat16_op_float(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) THSBFloat16_op_add(c10::BFloat16 a, c10::BFloat16 b);
EXPORT_API(c10::BFloat16) THSBFloat16_op_sub(c10::BFloat16 a, c10::BFloat16 b);
EXPORT_API(c10::BFloat16) THSBFloat16_op_mul(c10::BFloat16 a, c10::BFloat16 b);
EXPORT_API(c10::BFloat16) THSBFloat16_op_div(c10::BFloat16 a, c10::BFloat16 b);

EXPORT_API(float) THSBFloat16_op_add_float(c10::BFloat16 a, float b);
EXPORT_API(float) THSBFloat16_op_sub_float(c10::BFloat16 a, float b);
EXPORT_API(float) THSBFloat16_op_mul_float(c10::BFloat16 a, float b);
EXPORT_API(float) THSBFloat16_op_div_float(c10::BFloat16 a, float b);
EXPORT_API(float) THSBFloat16_op_add_lfloat(float a, c10::BFloat16 b);
EXPORT_API(float) THSBFloat16_op_sub_lfloat(float a, c10::BFloat16 b);
EXPORT_API(float) THSBFloat16_op_mul_lfloat(float a, c10::BFloat16 b);
EXPORT_API(float) THSBFloat16_op_div_lfloat(float a, c10::BFloat16 b);

EXPORT_API(double) THSBFloat16_op_add_double(c10::BFloat16 a, double b);
EXPORT_API(double) THSBFloat16_op_sub_double(c10::BFloat16 a, double b);
EXPORT_API(double) THSBFloat16_op_mul_double(c10::BFloat16 a, double b);
EXPORT_API(double) THSBFloat16_op_div_double(c10::BFloat16 a, double b);
EXPORT_API(double) THSBFloat16_op_add_ldouble(double a, c10::BFloat16 b);
EXPORT_API(double) THSBFloat16_op_sub_ldouble(double a, c10::BFloat16 b);
EXPORT_API(double) THSBFloat16_op_mul_ldouble(double a, c10::BFloat16 b);
EXPORT_API(double) THSBFloat16_op_div_ldouble(double a, c10::BFloat16 b);

EXPORT_API(c10::BFloat16) THSBFloat16_min(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) THSBFloat16_lowest(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) THSBFloat16_max(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) THSBFloat16_epsilon(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) THSBFloat16_round_error(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) THSBFloat16_infinity(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) THSBFloat16_quiet_NaN(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) THSBFloat16_signaling_NaN(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) THSBFloat16_denorm_min(c10::BFloat16 bf16);