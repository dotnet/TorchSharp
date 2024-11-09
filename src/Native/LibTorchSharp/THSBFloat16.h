// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"
#include "Utils.h"

#include "c10/util/BFloat16.h"
//#include "c10/util/BFloat16-inl.h"

EXPORT_API(c10::BFloat16) bfloat16_ctor(float value);
EXPORT_API(float) op_float(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) op_add(c10::BFloat16 a, c10::BFloat16 b);
EXPORT_API(c10::BFloat16) op_sub(c10::BFloat16 a, c10::BFloat16 b);
EXPORT_API(c10::BFloat16) op_mul(c10::BFloat16 a, c10::BFloat16 b);
EXPORT_API(c10::BFloat16) op_div(c10::BFloat16 a, c10::BFloat16 b);

EXPORT_API(float) op_add_float(c10::BFloat16 a, float b);
EXPORT_API(float) op_sub_float(c10::BFloat16 a, float b);
EXPORT_API(float) op_mul_float(c10::BFloat16 a, float b);
EXPORT_API(float) op_div_float(c10::BFloat16 a, float b);
EXPORT_API(float) op_add_lfloat(float a, c10::BFloat16 b);
EXPORT_API(float) op_sub_lfloat(float a, c10::BFloat16 b);
EXPORT_API(float) op_mul_lfloat(float a, c10::BFloat16 b);
EXPORT_API(float) op_div_lfloat(float a, c10::BFloat16 b);

EXPORT_API(double) op_add_double(c10::BFloat16 a, double b);
EXPORT_API(double) op_sub_double(c10::BFloat16 a, double b);
EXPORT_API(double) op_mul_double(c10::BFloat16 a, double b);
EXPORT_API(double) op_div_double(c10::BFloat16 a, double b);
EXPORT_API(double) op_add_ldouble(double a, c10::BFloat16 b);
EXPORT_API(double) op_sub_ldouble(double a, c10::BFloat16 b);
EXPORT_API(double) op_mul_ldouble(double a, c10::BFloat16 b);
EXPORT_API(double) op_div_ldouble(double a, c10::BFloat16 b);

EXPORT_API(c10::BFloat16) bfloat16_min(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) bfloat16_lowest(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) bfloat16_max(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) bfloat16_epsilon(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) bfloat16_round_error(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) bfloat16_infinity(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) bfloat16_quiet_NaN(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) bfloat16_signaling_NaN(c10::BFloat16 bf16);
EXPORT_API(c10::BFloat16) bfloat16_denorm_min(c10::BFloat16 bf16);