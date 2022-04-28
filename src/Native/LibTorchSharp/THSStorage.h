// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

EXPORT_API(int64_t) THSTensor_storage_offset(const Tensor tensor);

EXPORT_API(size_t) THSStorage_nbytes(const Tensor tensor);

EXPORT_API(void) THSStorage_set_nbytes(const Tensor tensor, size_t nbytes);

EXPORT_API(void*) THSStorage_data_ptr(const Tensor tensor);
