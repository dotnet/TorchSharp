// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
namespace TorchSharp
{
    // https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout
    public enum layout
    {
        /// <summary>
        /// dense Tensors
        /// </summary>
        strided,

        /// <summary>
        /// sparse COO Tensors
        /// </summary>
        sparse_coo
    }
}