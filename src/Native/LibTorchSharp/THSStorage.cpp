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

