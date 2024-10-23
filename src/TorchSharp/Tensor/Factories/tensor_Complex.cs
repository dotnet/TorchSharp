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
        public static Tensor tensor(System.Numerics.Complex scalar, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newComplexFloat64Scalar(scalar.Real, scalar.Imaginary, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return InstantiateTensorWithLeakSafeTypeChange(handle, dtype);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<(float Real, float Imaginary)> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray.ToArray(), dimensions, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor((float Real, float Imaginary)[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            var dataArray = new float[rawArray.Length * 2];
            for (var i = 0; i < rawArray.Length; i++) {
                dataArray[i * 2] = rawArray[i].Real;
                dataArray[i * 2 + 1] = rawArray[i].Imaginary;
            }

            return _tensor_generic(dataArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.ComplexFloat32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor((float Real, float Imaginary)[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            var dataArray = new float[rawArray.Length * 2];
            for (var i = 0; i < rawArray.Length; i++) {
                dataArray[i * 2] = rawArray[i].Real;
                dataArray[i * 2 + 1] = rawArray[i].Imaginary;
            }

            return _tensor_generic(dataArray, dimensions, (sbyte)ScalarType.ComplexFloat32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<(float Real, float Imaginary)> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<(float Real, float Imaginary)> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<(float Real, float Imaginary)> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<(float Real, float Imaginary)> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor((float Real, float Imaginary)[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            var dataArray = new float[rawArray.GetLongLength(0), rawArray.GetLongLength(1) * 2];
            for (var i = 0; i < rawArray.GetLongLength(0); i++) {
                for (var j = 0; j < rawArray.GetLongLength(1); j++) {
                    dataArray[i, j * 2] = rawArray[i, j].Real;
                    dataArray[i, j * 2 + 1] = rawArray[i, j].Imaginary;
                }
            }

            return _tensor_generic(dataArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.ComplexFloat32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor((float Real, float Imaginary)[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            var dataArray = new float[rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) * 2];
            for (var i = 0; i < rawArray.GetLongLength(0); i++) {
                for (var j = 0; j < rawArray.GetLongLength(1); j++) {
                    for (var k = 0; k < rawArray.GetLongLength(2); k++) {
                        dataArray[i, j, k * 2] = rawArray[i, j, k].Real;
                        dataArray[i, j, k * 2 + 1] = rawArray[i, j, k].Imaginary;
                    }
                }
            }

            return _tensor_generic(dataArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.ComplexFloat32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor((float Real, float Imaginary)[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            var dataArray = new float[rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) * 2];
            for (var i = 0; i < rawArray.GetLongLength(0); i++) {
                for (var j = 0; j < rawArray.GetLongLength(1); j++) {
                    for (var k = 0; k < rawArray.GetLongLength(2); k++) {
                        for (var l = 0; l < rawArray.GetLongLength(3); l++) {
                            dataArray[i, j, k, l * 2] = rawArray[i, j, k, l].Real;
                            dataArray[i, j, k, l * 2 + 1] = rawArray[i, j, k, l].Imaginary;
                        }
                    }
                }
            }

            return _tensor_generic(dataArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.ComplexFloat32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<System.Numerics.Complex> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray.ToArray(), dimensions, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(System.Numerics.Complex[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.ComplexFloat64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(System.Numerics.Complex[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.ComplexFloat64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<System.Numerics.Complex> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<System.Numerics.Complex> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<System.Numerics.Complex> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<System.Numerics.Complex> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(System.Numerics.Complex[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.ComplexFloat64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(System.Numerics.Complex[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.ComplexFloat64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(System.Numerics.Complex[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.ComplexFloat64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(Memory<System.Numerics.Complex> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.ComplexFloat64, dtype, device, requires_grad, names: names);
        }
    }
}
