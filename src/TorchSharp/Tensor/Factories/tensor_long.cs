// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(long scalar, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newInt64Scalar(scalar, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return InstantiateTensorWithLeakSafeTypeChange(handle, dtype);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(IList<long> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_i64(rawArray.ToArray(), dimensions, dtype, device, requires_grad, false, names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(long[] dataArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_i64(dataArray, stackalloc long[] { dataArray.LongLength }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(long[] dataArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_i64(dataArray, dimensions, dtype, device, requires_grad, names: names);
        }

        [Pure]
        private static Tensor _tensor_i64(Array rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype, Device? device, bool requires_grad, bool clone = false, string[]? names = null) =>
            _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Int64, dtype, device, requires_grad, names: names);

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<long> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { (long)rawArray.Count }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>
        /// The Torch runtime does not take ownership of the data, so there is no device argument.
        /// The input array must have rows * columns elements.
        /// </remarks>
        [Pure]
        public static Tensor tensor(IList<long> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>
        /// The Torch runtime does not take ownership of the data, so there is no device argument.
        /// The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        [Pure]
        public static Tensor tensor(IList<long> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>
        /// The Torch runtime does not take ownership of the data, so there is no device argument.
        /// The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        [Pure]
        public static Tensor tensor(IList<long> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(long[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_i64(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(long[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_i64(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(long[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_i64(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(Memory<long> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Int64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Cast a tensor to a 64-bit integer tensor.
        /// </summary>
        /// <param name="t">The input tensor</param>
        /// <returns>'this' if the tensor is already a 64-bit integer tensor; otherwise, a new tensor.</returns>
        public static Tensor LongTensor(Tensor t) => t.to(ScalarType.Int64);
    }
}
