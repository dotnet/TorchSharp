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

// Returns a Scalar object from a char value.
EXPORT_API(Scalar) THSTorch_sbtos(int8_t value);

// Returns a Scalar object from a byte value.
EXPORT_API(Scalar) THSTorch_btos(uint8_t value);

// Returns a Scalar object from a short value.
EXPORT_API(Scalar) THSTorch_stos(short value);

// Returns a Scalar object from an int value.
EXPORT_API(Scalar) THSTorch_itos(int value);

// Returns a Scalar object from a long value.
EXPORT_API(Scalar) THSTorch_ltos(long value);

// Returns a Scalar object from a float value.
EXPORT_API(Scalar) THSTorch_ftos(float value);

// Returns a Scalar object from a double value.
EXPORT_API(Scalar) THSTorch_dtos(double value);

// Dispose the scalar.
EXPORT_API(void) THSThorch_dispose_scalar(Scalar scalar);
