// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "Utils.h"

// API.

// Sets manually the seed.
EXPORT_API(void)      THSTorch_manual_seed(const int64_t seed);
EXPORT_API(void) THSCuda_manual_seed(const int64_t seed);
EXPORT_API(void) THSCuda_manual_seed_all(const int64_t seed);

EXPORT_API(Generator) THSGenerator_manual_seed(const int64_t seed);
EXPORT_API(void) THSGenerator_gen_manual_seed(const Generator gen, const int64_t seed);

EXPORT_API(Tensor) THSGenerator_get_rng_state(const Generator gen);
EXPORT_API(void)  THSGenerator_set_rng_state(const Generator gen, const Tensor tensor);

EXPORT_API(Generator) THSGenerator_default_generator();
EXPORT_API(Generator) THSGenerator_new(uint64_t seed, int64_t device, int64_t index);
EXPORT_API(int64_t)   THSGenerator_initial_seed(const Generator gen);
EXPORT_API(void)      THSGenerator_dispose(const Generator generator);

EXPORT_API(int) THSTorchCuda_is_available();
EXPORT_API(int) THSTorchCuda_cudnn_is_available();
EXPORT_API(int) THSTorchCuda_device_count();
EXPORT_API(void) THSTorchCuda_synchronize(const int64_t device);

// Returns the latest error. This is thread-local.
EXPORT_API(const char *) THSTorch_get_and_reset_last_err();

EXPORT_API(Scalar) THSTorch_int8_to_scalar(int8_t value);
EXPORT_API(Scalar) THSTorch_uint8_to_scalar(uint8_t value);
EXPORT_API(Scalar) THSTorch_int16_to_scalar(short value);
EXPORT_API(Scalar) THSTorch_int32_to_scalar(int value);
EXPORT_API(Scalar) THSTorch_int64_to_scalar(long value);
EXPORT_API(Scalar) THSTorch_float32_to_scalar(float value);
EXPORT_API(Scalar) THSTorch_float64_to_scalar(double value);
EXPORT_API(Scalar) THSTorch_bool_to_scalar(bool value);
EXPORT_API(Scalar) THSTorch_float16_to_scalar(float value);
EXPORT_API(Scalar) THSTorch_bfloat16_to_scalar(float value);

EXPORT_API(Scalar) THSTorch_complex32_to_scalar(float real, float imaginary);
EXPORT_API(Scalar) THSTorch_complex64_to_scalar(double real, double imaginary);

EXPORT_API(int8_t) THSTorch_scalar_to_int8(Scalar value);
EXPORT_API(uint8_t) THSTorch_scalar_to_uint8(Scalar value);
EXPORT_API(int16_t) THSTorch_scalar_to_int16(Scalar value);
EXPORT_API(int32_t) THSTorch_scalar_to_int32(Scalar value);
EXPORT_API(int64_t) THSTorch_scalar_to_int64(Scalar value);
EXPORT_API(float) THSTorch_scalar_to_float32(Scalar value);
EXPORT_API(double) THSTorch_scalar_to_float64(Scalar value);
EXPORT_API(bool) THSTorch_scalar_to_bool(Scalar value);

EXPORT_API(void) THSTorch_scalar_to_complex32(Scalar value, float* (*allocator)(size_t length));
EXPORT_API(void) THSTorch_scalar_to_complex64(Scalar value, double* (*allocator)(size_t length));

EXPORT_API(Tensor) THSTorch_lstsq(const Tensor input, const Tensor A, Tensor* qr);

EXPORT_API(int8_t) THSTorch_scalar_type(Scalar value);

// Dispose the scalar.
EXPORT_API(void) THSTorch_dispose_scalar(Scalar scalar);
