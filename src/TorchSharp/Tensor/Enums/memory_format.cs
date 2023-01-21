// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
namespace TorchSharp
{
    // https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format
    public enum memory_format
    {
        preserve_format,
        contiguous_format,
        channels_last
    }
}