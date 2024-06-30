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
        /// <summary>
        /// Returns True if the data type of input is a complex data type i.e., one of torch.complex64, and torch.complex128.
        /// </summary>
        /// <param name="input">The input tensor</param>
        public static bool is_complex(Tensor input) => is_complex(input.dtype);

        // https://pytorch.org/docs/stable/generated/torch.is_floating_point
        /// <summary>
        /// Returns True if the data type of input is a floating point data type.
        /// </summary>
        /// <param name="input">The input tensor</param>
        public static bool is_floating_point(Tensor input) => is_floating_point(input.dtype);

        // https://pytorch.org/docs/stable/generated/torch.is_nonzero
        /// <summary>
        /// Returns True if the input is a single element tensor which is not equal to zero after type conversions,
        /// i.e. not equal to torch.tensor([0.]) or torch.tensor([0]) or torch.tensor([False]).
        /// Throws an InvalidOperationException if torch.numel() != 1.
        /// </summary>
        /// <param name="input">The input tensor</param>
        public static bool is_nonzero(Tensor input) => input.is_nonzero();

        // https://pytorch.org/docs/stable/generated/torch.set_default_dtype
        /// <summary>
        /// Sets the default floating point dtype to d. This dtype is:
        /// 1. The inferred dtype for python floats in torch.tensor().
        /// 2. Used to infer dtype for python complex numbers.
        /// The default complex dtype is set to torch.complex128 if default floating point dtype is torch.float64, otherwise it’s set to torch.complex64
        /// The default floating point dtype is initially torch.float32.
        /// </summary>
        /// <param name="dtype"></param>
        public static void set_default_dtype(ScalarType dtype)
        {
            if (!dtype.IsFloatingPoint()) {
                throw new ArgumentException("only floating-point types are supported as the default type");
            }
            default_dtype = dtype;
        }

        // https://pytorch.org/docs/stable/generated/torch.get_default_dtype
        /// <summary>
        /// Get the current default floating point torch.dtype.
        /// </summary>
        /// <returns></returns>
        [Pure]public static ScalarType get_default_dtype() => default_dtype;

        [Pure]public static Device get_default_device() => default_device;

        // https://pytorch.org/docs/stable/generated/torch.set_default_tensor_type
        public static void set_default_tensor_type(Tensor t) => set_default_dtype(t.dtype);

        public static void set_default_device(Device device) 
        {
            if (device == null) 
                throw new ArgumentNullException(nameof(device));
            default_device = device;
        }

        public static void set_default_device(string device)
        {
            set_default_device(new Device(device));
        }

        // https://pytorch.org/docs/stable/generated/torch.numel
        /// <summary>
        /// Get the number of elements in the input tensor.
        /// </summary>
        [Pure]public static long numel(Tensor input) => input.numel();
    }
}