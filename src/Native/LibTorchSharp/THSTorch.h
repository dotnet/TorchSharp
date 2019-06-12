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
