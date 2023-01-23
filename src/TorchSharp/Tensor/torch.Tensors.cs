// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Diagnostics.Contracts;

namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#tensors
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.is_tensor
        [Pure]public static bool is_tensor(object obj) => obj is Tensor;

        // https://pytorch.org/docs/stable/generated/torch.is_storage
        [Pure]public static bool is_storage(object obj) => obj is Storage;

        // https://pytorch.org/docs/stable/generated/torch.is_complex
        [Pure, Obsolete("not implemented", true)]
        public static bool is_complex(object input) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.is_conj
        [Pure, Obsolete("not implemented", true)]
        public static bool is_conj(object input) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.is_floating_point
        [Pure, Obsolete("not implemented", true)]
        public static bool is_floating_point(object input) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.is_nonzero
        [Pure, Obsolete("not implemented", true)]
        public static bool is_nonzero(object input) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.set_default_dtype
        /// <summary>
        /// Sets the default floating point dtype to d. This dtype is:
        /// 1. The inferred dtype for python floats in torch.tensor().
        /// 2. Used to infer dtype for python complex numbers.
        /// The default complex dtype is set to torch.complex128 if default floating point dtype is torch.float64, otherwise it’s set to torch.complex64
        /// The default floating point dtype is initially torch.float32.
        /// </summary>
        /// <param name="dtype"></param>
        public static void set_default_dtype(ScalarType dtype) { default_dtype = dtype; }

        // https://pytorch.org/docs/stable/generated/torch.get_default_dtype
        /// <summary>
        /// Get the current default floating point torch.dtype.
        /// </summary>
        /// <returns></returns>
        [Pure]public static ScalarType get_default_dtype() => default_dtype;

        // https://pytorch.org/docs/stable/generated/torch.set_default_tensor_type
        public static void set_default_tensor_type(Tensor t) => set_default_dtype(t.dtype);

        // https://pytorch.org/docs/stable/generated/torch.numel
        /// <summary>
        /// Get the number of elements in the input tensor.
        /// </summary>
        [Pure]public static long numel(Tensor input) => input.numel();

        // https://pytorch.org/docs/stable/generated/torch.set_printoptions
        [Obsolete("not implemented", true)]
        public static void set_printoptions(
            int precision = 4,
            int threshold = 1000,
            int edgeitems = 3,
            int linewidth = 80,
            PrintOptionsProfile profile = PrintOptionsProfile.@default,
            bool? sci_mode = null) => throw new NotImplementedException();
    }
}