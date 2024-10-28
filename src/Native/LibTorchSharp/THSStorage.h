// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "torch/torch.h"

#include "Utils.h"

EXPORT_API(int64_t) THSTensor_storage_offset(const Tensor tensor);

EXPORT_API(size_t) THSStorage_nbytes(const Tensor tensor);

EXPORT_API(void) THSStorage_set_nbytes(const Tensor tensor, size_t nbytes);

EXPORT_API(void*) THSStorage_data_ptr(const Tensor tensor);
/*
template<typename T>
T* THSStorage_tensor_array(const Tensor tensor)
{
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 4
    return tensor->data_ptr<T>();
#else
    return tensor->data<T>();
#endif
}

EXPORT_API(int*) THSStorage_tensor_to_array_int(const Tensor tensor);
EXPORT_API(long*) THSStorage_tensor_to_array_long(const Tensor tensor);
EXPORT_API(float*) THSStorage_tensor_to_array_float(const Tensor tensor);
EXPORT_API(double*) THSStorage_tensor_to_array_double(const Tensor tensor);
EXPORT_API(char*) THSStorage_tensor_to_array_char(const Tensor tensor);*/