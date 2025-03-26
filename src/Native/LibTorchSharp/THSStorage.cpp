//// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#include "THSStorage.h"

int64_t THSTensor_storage_offset(const Tensor tensor)
{
    return tensor->storage_offset();
}

size_t THSStorage_nbytes(const Tensor tensor)
{
    return tensor->storage().nbytes();
}

void THSStorage_set_nbytes(const Tensor tensor, size_t nbytes)
{
    tensor->storage().set_nbytes(nbytes);
}

void* THSStorage_data_ptr(const Tensor tensor)
{
    auto &st = tensor->storage();
    auto &dp = st.data_ptr();
    return dp.get();
}

/*
int* THSStorage_tensor_to_array_int(const Tensor tensor)
{
    return THSStorage_tensor_array<int>(tensor);
}
long* THSStorage_tensor_to_array_long(const Tensor tensor)
{
    return THSStorage_tensor_array<long>(tensor);
}

float* THSStorage_tensor_to_array_float(const Tensor tensor)
{
    return THSStorage_tensor_array<float>(tensor);
}

double* THSStorage_tensor_to_array_double(const Tensor tensor)
{
    return THSStorage_tensor_array<double>(tensor);
}
char* THSStorage_tensor_to_array_char(const Tensor tensor)
{
    return THSStorage_tensor_array<char>(tensor);
}*/