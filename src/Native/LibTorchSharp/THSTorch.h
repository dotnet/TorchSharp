// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "Utils.h"

// API.

// Sets manually the seed.
EXPORT_API(void) THSTorch_seed(const int64_t seed);

// Sets manually the seed.
EXPORT_API(int) THSTorch_isCudaAvailable();

// Returns the latest error. This is thread-local.
EXPORT_API(const char *) THSTorch_get_and_reset_last_err();

EXPORT_API(Scalar) THSTorch_int8_to_scalar(int8_t value);
EXPORT_API(Scalar) THSTorch_uint8_to_scalar(uint8_t value);
EXPORT_API(Scalar) THSTorch_short_to_scalar(short value);
EXPORT_API(Scalar) THSTorch_int32_to_scalar(int value);
EXPORT_API(Scalar) THSTorch_long_to_scalar(long value);
EXPORT_API(Scalar) THSTorch_float32_to_scalar(float value);
EXPORT_API(Scalar) THSTorch_float64_to_scalar(double value);
EXPORT_API(Scalar) THSTorch_bool_to_scalar(bool value);
EXPORT_API(Scalar) THSTorch_half_to_scalar(float value);

// Dispose the scalar.
EXPORT_API(void) THSTorch_dispose_scalar(Scalar scalar);
