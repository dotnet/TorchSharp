// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<BFloat16> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), dimensions, (sbyte)ScalarType.BFloat16, dtype, device, requires_grad, false, names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(BFloat16[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.BFloat16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(BFloat16[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.BFloat16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<BFloat16> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<BFloat16> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<BFloat16> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<BFloat16> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(BFloat16[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.BFloat16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(BFloat16[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.BFloat16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(BFloat16[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.BFloat16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(Memory<BFloat16> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.BFloat16, dtype, device, requires_grad, names: names);
        }
    }
}
