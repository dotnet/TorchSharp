// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Diagnostics.Contracts;
using static TorchSharp.PInvoke.LibTorchSharp;
using System.Text;

namespace TorchSharp
{
    using Utils;

    public static partial class torch
    {
        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
        /// common difference step, starting from start
        /// </summary>
        /// <param name="start">The starting value for the set of points.</param>
        /// <param name="stop">The ending value for the set of points</param>
        /// <param name="step">The gap between each pair of adjacent points.</param>
        /// <param name="dtype">the desired data type of returned tensor.
        /// Default: if null, uses a global default (see torch.set_default_tensor_type()).
        /// If dtype is not given, infer the data type from the other input arguments.
        /// If any of start, end, or stop are floating-point, the dtype is inferred to be the default dtype, see get_default_dtype().
        /// Otherwise, the dtype is inferred to be torch.int64.</param>
        /// <param name="device"></param>
        /// <param name="requires_grad"> If autograd should record operations on the returned tensor. Default: false.</param>
        public static Tensor arange(Scalar start, Scalar stop, Scalar step, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);

            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                if (start.Type.IsIntegral() && stop.Type.IsIntegral() && step.Type.IsIntegral()) {
                    dtype = ScalarType.Int64;
                } else {
                    dtype = get_default_dtype();
                }
            }

            if (dtype == ScalarType.ComplexFloat32) {
                return ComplexFloat32Tensor.arange(start, stop, step, device, requires_grad);
            } else if (dtype == ScalarType.ComplexFloat64) {
                return ComplexFloat64Tensor.arange(start, stop, step, device, requires_grad);
            }

            var handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange(start.Handle, stop.Handle, step.Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            }
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        public static Tensor arange(Scalar start, Scalar stop, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            return arange(start, stop, 1, dtype, device, requires_grad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        public static Tensor arange(Scalar stop, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            return arange(0, stop, 1, dtype, device, requires_grad);
        }

        // as_tensor()

        public static Tensor as_tensor(Tensor data, ScalarType? dtype = null, Device? device = null)
        {
            if (dtype != null && device != null && (data.dtype != dtype || data.device != device)) {
                return data.to(dtype.Value, device).requires_grad_(data.requires_grad);
            } else if (dtype != null && data.dtype != dtype) {
                return data.to(dtype.Value).requires_grad_(data.requires_grad);
            } else if (device != null && data.device != device) {
                return data.to(device).requires_grad_(data.requires_grad);
            } else {
                return data.alias();
            }
        }

        public static Tensor as_tensor(IList<bool> rawArray, ScalarType? dtype = null, Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(bool[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<byte> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(byte[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<sbyte> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(sbyte[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<short> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(short[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<int> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(int[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<long> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(long[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<float> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(float[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<double> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(double[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<(float, float)> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor((float, float)[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }

        public static Tensor as_tensor(IList<System.Numerics.Complex> rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray.ToArray(), dtype, device);
        }

        public static Tensor as_tensor(System.Numerics.Complex[] rawArray, torch.ScalarType? dtype = null, torch.Device? device = null)
        {
            return torch.from_array(rawArray, dtype, device);
        }





        private static Tensor _zeros(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {

                IntPtr handle = IntPtr.Zero;

                fixed (long* psizes = size) {

                    handle = THSTensor_zeros((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);

                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    }

                    if (handle == IntPtr.Zero) { CheckForErrors(); }

                    var result = new Tensor(handle);

                    if (names != null && names.Length > 0) {

                        result.rename_(names);
                    }

                    return result;
                }
            }
        }

        /// <summary>
        /// Create a new tensor filled with zeros
        /// </summary>
        public static Tensor zeros(long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(size, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new tensor filled with zeros
        /// </summary>
        public static Tensor zeros(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(size, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(long size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { size }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(int size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { size }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(int rows, int columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(int dim0, int dim1, int dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with zeros
        /// </summary>
        public static Tensor zeros(int dim0, int dim1, int dim2, int dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _zeros(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Returns a tensor filled with the scalar value 0, with the same size as input.
        /// </summary>
        public static Tensor zeros_like(Tensor input, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null) => input.zeros_like(dtype, device, requires_grad);


        /// <summary>
        /// Create a new tensor filled with ones
        /// </summary>
        private static Tensor _ones(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_ones((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }
                    var result = new Tensor(handle);

                    if (names != null && names.Length > 0) {

                        result.rename_(names);
                    }

                    return result;
                }
            }
        }

        /// <summary>
        /// Create a new tensor filled with ones
        /// </summary>
        public static Tensor ones(long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(size, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new tensor filled with ones
        /// </summary>
        public static Tensor ones(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(size, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with ones
        /// </summary>
        public static Tensor ones(long size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { size }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with ones
        /// </summary>
        public static Tensor ones(long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with ones
        /// </summary>
        public static Tensor ones(long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with ones
        /// </summary>
        public static Tensor ones(long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with ones
        /// </summary>
        public static Tensor ones(int size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { size }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with ones
        /// </summary>
        public static Tensor ones(int rows, int columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with ones
        /// </summary>
        public static Tensor ones(int dim0, int dim1, int dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with ones
        /// </summary>
        public static Tensor ones(int dim0, int dim1, int dim2, int dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _ones(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Returns a tensor filled with the scalar value 1, with the same size as input.
        /// </summary>
        public static Tensor ones_like(Tensor input, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null) => input.ones_like(dtype, device, requires_grad);


        /// <summary>
        /// Create a new tensor filled with empty
        /// </summary>
        private static Tensor _empty(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }
                    var result = new Tensor(handle);

                    if (names != null && names.Length > 0) {

                        result.rename_(names);
                    }

                    return result;
                }
            }
        }

        /// <summary>
        /// Create a new tensor filled with empty
        /// </summary>
        public static Tensor empty(long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _empty(size, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new tensor filled with empty
        /// </summary>
        public static Tensor empty(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _empty(size, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with empty
        /// </summary>
        public static Tensor empty(long size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _empty(stackalloc long[] { size }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with empty
        /// </summary>
        public static Tensor empty(long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _empty(stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with empty
        /// </summary>
        public static Tensor empty(long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _empty(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with empty
        /// </summary>
        public static Tensor empty(long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _empty(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with empty
        /// </summary>
        public static Tensor empty(int size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _empty(stackalloc long[] { size }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with empty
        /// </summary>
        public static Tensor empty(int rows, int columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _empty(stackalloc long[] { rows, columns }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with empty
        /// </summary>
        public static Tensor empty(int dim0, int dim1, int dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _empty(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with empty
        /// </summary>
        public static Tensor empty(int dim0, int dim1, int dim2, int dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _empty(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Returns a tensor filled with uninitialized data, with the same size as input.
        /// </summary>
        public static Tensor empty_like(Tensor input, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null) => input.empty_like(dtype, device, requires_grad);


        /// <summary>
        /// Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        public static Tensor empty_strided(long[] size, long[] strides, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            unsafe {
                fixed (long* psizes = size, pstrides = strides) {
                    var handle = THSTensor_empty_strided((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }
                    var result = new Tensor(handle);

                    if (names != null && names.Length > 0) {

                        result.rename_(names);
                    }

                    return result;
                }
            }
        }

        /// <summary>
        /// Create a new tensor filled with a given value
        /// </summary>
        private static Tensor _full(ReadOnlySpan<long> size, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                if (value.Type.IsIntegral()) {
                    dtype = ScalarType.Int64;
                } else {
                    dtype = get_default_dtype();
                }
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_full((IntPtr)psizes, size.Length, value.Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full((IntPtr)psizes, size.Length, value.Handle, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }
                    var result = new Tensor(handle);

                    if (names != null && names.Length > 0) {

                        result.rename_(names);
                    }

                    return result;
                }
            }
        }

        /// <summary>
        /// Create a new tensor filled with a given value
        /// </summary>
        public static Tensor full(long[] size, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(size, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new tensor filled with a given value
        /// </summary>
        public static Tensor full(ReadOnlySpan<long> size, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(size, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with given value
        /// </summary>
        public static Tensor full(long size, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { size }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with given value
        /// </summary>
        public static Tensor full(long rows, long columns, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { rows, columns }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with given value
        /// </summary>
        public static Tensor full(long dim0, long dim1, long dim2, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { dim0, dim1, dim2 }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with given value
        /// </summary>
        public static Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { dim0, dim1, dim2, dim3 }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with given value
        /// </summary>
        public static Tensor full(int size, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { size }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with given value
        /// </summary>
        public static Tensor full(int rows, int columns, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { rows, columns }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with given value
        /// </summary>
        public static Tensor full(int dim0, int dim1, int dim2, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { dim0, dim1, dim2 }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with given value
        /// </summary>
        public static Tensor full(int dim0, int dim1, int dim2, int dim3, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _full(stackalloc long[] { dim0, dim1, dim2, dim3 }, value, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Returns a tensor with the same size as input filled with 'value.'
        /// </summary>
        public static Tensor full_like(Tensor input, Scalar value, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null) => input.full_like(value, dtype, device, requires_grad);

        /// <summary>
        /// Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        public static Tensor eye(long rows, long columns = -1L, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye(rows, columns, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye(rows, columns, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            }
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            var result = new Tensor(handle);

            if (names != null && names.Length > 0) {

                result.rename_(names);
            }

            return result;
        }


        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [low, max).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        public static Tensor randint(long low, long high, Size size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            var shape = size.Shape;

            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = ScalarType.Int64;
            }

            ValidateIntegerRange(low, dtype.Value, nameof(low));
            ValidateIntegerRange(high - 1, dtype.Value, nameof(high));

            var genHandle = (generator is null) ? IntPtr.Zero : generator.Handle;

            Tensor? result = null;

            if (dtype == ScalarType.ComplexFloat32) {
                result = randint_c32(genHandle, low, high, shape, device, requires_grad, names: names);
            } else if (dtype == ScalarType.ComplexFloat64) {
                result = randint_c64(genHandle, low, high, shape, device, requires_grad, names: names);
            }

            if (result is null) {
                unsafe {
                    fixed (long* psizes = shape) {
                        var handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, shape.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                        if (handle == IntPtr.Zero) {
                            GC.Collect();
                            GC.WaitForPendingFinalizers();
                            handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, shape.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                        }
                        if (handle == IntPtr.Zero) { CheckForErrors(); }
                        result = new Tensor(handle);
                    }
                }
            }

            if (names != null && names.Length > 0) {

                result.rename_(names);
            }

            return result;
        }

        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        public static Tensor randint(long high, Size size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randint(0, high, size, dtype, device, requires_grad, generator, names);
        }

        // Note: Once F# implicit conversion support is broadly available, all the following overloads will be redundant.

        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        public static Tensor randint(long high, long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randint(0, high, new Size(size), dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [low, max).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        /// <remarks>
        /// The array-size and 'int' overloads are necessary for F#, which doesn't implicitly convert types.
        /// Once implicit conversion support is broadly available, some of these overloads can be removed.
        /// </remarks>
        public static Tensor randint(long low, long high, long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randint(low, high, new Size(size), dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [low, max).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        /// <remarks>
        /// The array-size and 'int' overloads are necessary for F#, which doesn't implicitly convert types.
        /// Once implicit conversion support is broadly available, some of these overloads can be removed.
        /// </remarks>
        public static Tensor randint(int low, int high, int[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randint(low, high, new Size(size), dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="dtype">The desired data type of the tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="generator">An optional random number genertor object.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        public static Tensor randint(int high, int[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randint(0, high, new Size(size), dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 32-bit complex tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        /// <param name="genHandle"></param>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        private static Tensor randint_c32(IntPtr genHandle, long low, long high, long[] size, Device device, bool requires_grad, string[]? names)
        {
            var sz = new List<long>();
            sz.AddRange(size);
            sz.Add(2);
            var size2 = sz.ToArray();

            unsafe {
                fixed (long* psizes = size2) {
                    //
                    // This is a little roundabout -- the native library doesn't support 'randint' for complex types,
                    // but we can get around that by adding another dimension, creating a float tensor, and then
                    // converting it to a complex tensor in the end.
                    //
                    var handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float32, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float32, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }

                    var cmplx = THSTensor_view_as_complex(handle);
                    if (cmplx == IntPtr.Zero)
                        CheckForErrors();

                    //
                    // view_as_complex() creates a view, but we want an independent tensor, so we have to create one and then copy the view's data into it.
                    //
                    var res = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int)device.type, device.index, requires_grad);
                    if (res == IntPtr.Zero)
                        CheckForErrors();

                    res = THSTensor_copy_(res, cmplx, false);
                    if (res == IntPtr.Zero)
                        CheckForErrors();

                    THSTensor_dispose(handle);
                    THSTensor_dispose(cmplx);

                    var result = new Tensor(res);

                    if (names != null && names.Length > 0) {

                        result.rename_(names);
                    }

                    return result;
                }
            }
        }

        /// <summary>
        /// Create a new 64-bit complex tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        /// <param name="genHandle"></param>
        /// <param name="low">Lowest integer to be drawn from the distribution.</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="size">The shape of the output tensor.</param>
        /// <param name="device">The desired device of returned tensor.</param>
        /// <param name="requires_grad">If autograd should record operations on the returned tensor.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        private static Tensor randint_c64(IntPtr genHandle, long low, long high, long[] size, Device device, bool requires_grad, string[]? names)
        {
            var sz = new List<long>();
            sz.AddRange(size);
            sz.Add(2);
            var size2 = sz.ToArray();

            unsafe {
                fixed (long* psizes = size2) {
                    //
                    // This is a little roundabout -- the native library doesn't support 'randint' for complex types,
                    // but we can get around that by adding another dimension, creating a float tensor, and then
                    // converting it to a complex tensor in the end.
                    //
                    var handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float64, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint(genHandle, low, high, (IntPtr)psizes, size2.Length, (sbyte)ScalarType.Float64, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }

                    var cmplx = THSTensor_view_as_complex(handle);
                    if (cmplx == IntPtr.Zero)
                        CheckForErrors();

                    //
                    // view_as_complex() creates a view, but we want an independent tensor, so we have to create one and then copy the view's data into it.
                    //
                    var res = THSTensor_empty((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int)device.type, device.index, requires_grad);
                    if (res == IntPtr.Zero)
                        CheckForErrors();

                    res = THSTensor_copy_(res, cmplx, false);
                    if (res == IntPtr.Zero)
                        CheckForErrors();

                    THSTensor_dispose(handle);
                    THSTensor_dispose(cmplx);

                    var result = new Tensor(res);
                    if (names != null && names.Length > 0) {
                        result.rename_(names);
                    }
                    return result;
                }
            }
        }

        /// <summary>
        /// Similar to the function above, but the means and standard deviations are shared among all drawn elements. The resulting tensor has size given by size.
        /// </summary>
        /// <param name="mean">The mean for all distributions</param>
        /// <param name="std">The standard deviation for all distributions</param>
        /// <param name="size">A sequence of integers defining the shape of the output tensor.</param>
        /// <param name="dtype"></param>
        /// <param name="device"></param>
        /// <param name="requires_grad"></param>
        /// <param name="generator">An optional random number generator</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        /// <returns></returns>
        public static Tensor normal(double mean, double std, ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randn(size, dtype: dtype, device: device, requires_grad: requires_grad, generator: generator) * std + mean;
        }

        /// <summary>
        /// Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        private static Tensor _rand(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            if (dtype.HasValue && torch.is_integral(dtype.Value))
                throw new ArgumentException($"torch.rand() was passed a bad dtype: {dtype}. It must be floating point or complex.", "dtype");

            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            var genHandle = (generator is null) ? IntPtr.Zero : generator.Handle;

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_rand(genHandle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand(genHandle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }
                    var result = new Tensor(handle);
                    if (names != null && names.Length > 0) {
                        result.rename_(names);
                    }
                    return result;
                }
            }
        }

        /// <summary>
        /// Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(size, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(size, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(long size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { size }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { rows, columns }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(int size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { size }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(int rows, int columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { rows, columns }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(int dim0, int dim1, int dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand(int dim0, int dim1, int dim2, int dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _rand(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        private static Tensor _randn(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            if (dtype.HasValue && torch.is_integral(dtype.Value))
                throw new ArgumentException($"torch.randn() was passed a bad dtype: {dtype}. It must be floating point or complex.", "dtype");

            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            var genHandle = (generator is null) ? IntPtr.Zero : generator.Handle;

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_randn(genHandle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn(genHandle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }
                    var result = new Tensor(handle);
                    if (names != null && names.Length > 0) {
                        result.rename_(names);
                    }
                    return result;
                }
            }
        }

        /// <summary>
        /// Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(size, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(size, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(long size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { size }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { rows, columns }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 1-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(int size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { size }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 2-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(int rows, int columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { rows, columns }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 3-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(int dim0, int dim1, int dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { dim0, dim1, dim2 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a new 4-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn(int dim0, int dim1, int dim2, int dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return _randn(stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, generator, names);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(bool scalar, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newBoolScalar(scalar, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            var tensor = new Tensor(handle);
            tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            return tensor;
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(byte scalar, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newByteScalar(scalar, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(sbyte scalar, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newInt8Scalar(scalar, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(short scalar, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newInt16Scalar(scalar, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(int scalar, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newInt32Scalar(scalar, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(long scalar, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newInt64Scalar(scalar, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            var tensor = new Tensor(handle);
            if (device is { }) {
                tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            } else if (dtype.HasValue) {
                tensor = tensor.to_type(dtype.Value);
            }
            return tensor;
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(float scalar, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newFloat32Scalar(scalar, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(double scalar, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newFloat64Scalar(scalar, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            var tensor = new Tensor(handle);
            tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            return tensor;
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(float real, float imaginary, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newComplexFloat32Scalar(real, imaginary, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            var tensor = new Tensor(handle);
            if (device is { }) {
                tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            } else if (dtype.HasValue) {
                tensor = tensor.to_type(dtype.Value);
            }
            return tensor;
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor((float Real, float Imaginary) scalar, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newComplexFloat32Scalar(scalar.Real, scalar.Imaginary, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            var tensor = new Tensor(handle);
            if (device is { }) {
                tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            } else if (dtype.HasValue) {
                tensor = tensor.to_type(dtype.Value);
            }
            return tensor;
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor((double Real, double Imaginary) scalar, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newComplexFloat64Scalar(scalar.Real, scalar.Imaginary, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            var tensor = new Tensor(handle);
            if (device is { }) {
                tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            } else if (dtype.HasValue) {
                tensor = tensor.to_type(dtype.Value);
            }
            return tensor;
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(double real, double imaginary, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newComplexFloat64Scalar(real, imaginary, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            var tensor = new Tensor(handle);
            if (device is { }) {
                tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            } else if (dtype.HasValue) {
                tensor = tensor.to_type(dtype.Value);
            }
            return tensor;
        }

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor tensor(System.Numerics.Complex scalar, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            var handle = THSTensor_newComplexFloat64Scalar(scalar.Real, scalar.Imaginary, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            var tensor = new Tensor(handle);
            tensor = dtype.HasValue ? tensor.to(dtype.Value, device) : tensor.to(device);
            return tensor;
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor tensor(IList<bool> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), dimensions, (sbyte)ScalarType.Bool, dtype, device, requires_grad, false, names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(bool[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.Bool, dtype, device, requires_grad, names: names);
        }

        public static Tensor tensor(bool[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Bool, dtype, device, requires_grad, names: names);
        }

        private static Tensor _tensor_generic(Array rawArray, ReadOnlySpan<long> dimensions, sbyte origType, ScalarType? dtype, Device? device, bool requires_grad, bool clone = true, string[]? names = null)
        {
            {
                // Validate the sizes before handing over storage to native code...
                var prod = 1L;
                foreach (var sz in dimensions) prod *= sz;

                if (origType == (sbyte)ScalarType.ComplexFloat32)
                    prod *= 2;

                if (prod != rawArray.LongLength)
                    throw new ArgumentException($"mismatched total size creating a tensor from an array: {prod} vs. {rawArray.LongLength}");
            }

            device = InitializeDevice(device);

            if (clone) { rawArray = (Array)rawArray.Clone(); }

            var dataHandle = GCHandle.Alloc(rawArray, GCHandleType.Pinned);
            var dataArrayAddr = dataHandle.AddrOfPinnedObject();
            var gchp = GCHandle.ToIntPtr(dataHandle);
            TorchSharp.PInvoke.GCHandleDeleter deleter = null!;
            deleter = new TorchSharp.PInvoke.GCHandleDeleter((IntPtr ptr) => {
                GCHandle.FromIntPtr(gchp).Free();
                deleters.TryRemove(deleter, out deleter!);
            });
            deleters.TryAdd(deleter, deleter); // keep the delegate alive

            unsafe {
                fixed (long* shape = dimensions) {
                    var handle = THSTensor_new(dataArrayAddr, deleter, (IntPtr)shape, dimensions.Length, origType, requires_grad);

                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_new(dataArrayAddr, deleter, (IntPtr)shape, dimensions.Length, origType, requires_grad);
                    }

                    if (handle == IntPtr.Zero) { CheckForErrors(); }
                    var tensor = new Tensor(handle);

                    var needsConversion = dtype.HasValue && dtype.Value != (ScalarType)origType;

                    if (device is not null) {
                        tensor = needsConversion ? tensor.to(dtype!.Value, device) : tensor.to(device);
                    } else if (needsConversion) {
                        tensor = tensor.to_type(dtype!.Value);
                    }
                    if (names != null && names.Length > 0) {
                        tensor.rename_(names);
                    }
                    return tensor;
                }
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<bool> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<bool> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<bool> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<bool> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }
        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(bool[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.Bool, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(bool[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.Bool, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(bool[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.Bool, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<byte> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), dimensions, (sbyte)ScalarType.Byte, dtype, device, requires_grad, false, names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(byte[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.Byte, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the array passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(byte[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Byte, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<byte> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<byte> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<byte> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<byte> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(byte[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.Byte, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(byte[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.Byte, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(byte[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.Byte, dtype, device, requires_grad, names: names);
        }


        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<sbyte> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), dimensions, (sbyte)ScalarType.Int8, dtype, device, requires_grad, false, names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(sbyte[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.Int8, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(sbyte[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Int8, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<sbyte> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<sbyte> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<sbyte> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<sbyte> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(sbyte[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.Int8, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(sbyte[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.Int8, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(sbyte[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.Int8, dtype, device, requires_grad, names: names);
        }


        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<short> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), dimensions, (sbyte)ScalarType.Int16, dtype, device, requires_grad, false, names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(short[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.Int16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(short[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Int16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<short> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<short> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<short> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<short> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(short[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.Int16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(short[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.Int16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(short[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.Int16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<int> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), dimensions, (sbyte)ScalarType.Int32, dtype, device, requires_grad, false, names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(int[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.Int32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(int[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Int32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<int> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<int> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<int> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(int[] rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(int[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.Int32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(int[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.Int32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(int[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.Int32, dtype, device, requires_grad, names: names);
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
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<float> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), dimensions, (sbyte)ScalarType.Float32, dtype, device, requires_grad, false, names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(float[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.Float32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(float[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Float32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<float> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<float> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<float> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<float> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(float[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.Float32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(float[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.Float32, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(float[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.Float32, dtype, device, requires_grad, names: names);
        }


        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<double> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), stackalloc long[] { rawArray.Count }, (sbyte)ScalarType.Float64, dtype: dtype, device: device, requires_grad: requires_grad, false, names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(double[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.Float64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(double[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Float64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>
        /// The Torch runtime does not take ownership of the data, so there is no device argument.
        /// The input array must have rows * columns elements.
        /// </remarks>
        [Pure]
        public static Tensor tensor(IList<double> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), stackalloc long[] { rows, columns }, (sbyte)ScalarType.Float64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>
        /// The Torch runtime does not take ownership of the data, so there is no device argument.
        /// The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        [Pure]
        public static Tensor tensor(IList<double> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), stackalloc long[] { dim0, dim1, dim2 }, (sbyte)ScalarType.Float64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        /// The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        [Pure]
        public static Tensor tensor(IList<double> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), stackalloc long[] { dim0, dim1, dim2, dim3 }, (sbyte)ScalarType.Float64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(double[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.Float64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(double[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.Float64, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.from
        /// </summary>
        [Pure]
        public static Tensor tensor(double[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.Float64, dtype, device, requires_grad, names: names);
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
        /// Cast a tensor to a Boolean tensor.
        /// </summary>
        /// <param name="t">The input tensor</param>
        /// <returns>'this' if the tensor is already a Boolean tensor; otherwise, a new tensor.</returns>
        public static Tensor BoolTensor(Tensor t) => t.to(ScalarType.Bool);

        /// <summary>
        /// Cast a tensor to a byte tensor.
        /// </summary>
        /// <param name="t">The input tensor</param>
        /// <returns>'this' if the tensor is already a byte tensor; otherwise, a new tensor.</returns>
        public static Tensor ByteTensor(Tensor t) => t.to(ScalarType.Byte);

        /// <summary>
        /// Cast a tensor to a 16-bit integer tensor.
        /// </summary>
        /// <param name="t">The input tensor</param>
        /// <returns>'this' if the tensor is already a 16-bit integer tensor; otherwise, a new tensor.</returns>
        public static Tensor ShortTensor(Tensor t) => t.to(ScalarType.Int16);

        /// <summary>
        /// Cast a tensor to a 32-bit integer tensor.
        /// </summary>
        /// <param name="t">The input tensor</param>
        /// <returns>'this' if the tensor is already a 32-bit integer tensor; otherwise, a new tensor.</returns>
        public static Tensor IntTensor(Tensor t) => t.to(ScalarType.Int32);

        /// <summary>
        /// Cast a tensor to a 64-bit integer tensor.
        /// </summary>
        /// <param name="t">The input tensor</param>
        /// <returns>'this' if the tensor is already a 64-bit integer tensor; otherwise, a new tensor.</returns>
        public static Tensor LongTensor(Tensor t) => t.to(ScalarType.Int64);

        /// <summary>
        /// Cast a tensor to a 32-bit floating point tensor.
        /// </summary>
        /// <param name="t">The input tensor</param>
        /// <returns>'this' if the tensor is already a 32-bit floating point; otherwise, a new tensor.</returns>
        public static Tensor FloatTensor(Tensor t) => t.to(ScalarType.Float32);

        /// <summary>
        /// Cast a tensor to a 64-bit floating point tensor.
        /// </summary>
        /// <param name="t">The input tensor</param>
        /// <returns>'this' if the tensor is already a 64-bit floating point tensor; otherwise, a new tensor.</returns>
        public static Tensor DoubleTensor(Tensor t) => t.to(ScalarType.Float64);

        /// <summary>
        /// Creates a <see cref="torch.Tensor">torch tensor</see> from an arbitrary <see cref="Array">array</see>.
        /// </summary>
        /// <param name="rawArray">The arbitrary array to create the tensor from.</param>
        /// <param name="device">The device where the tensor is to be located. Defaults to 'cpu'.</param>
        /// <returns>A <see cref="torch.Tensor">torch tensor</see></returns>
        /// <remarks>
        /// This function roughly corresponds to torch.from_numpy(). It shares the underlying buffer between the input and output.
        /// torch.tensor() always makes a copy, which can be orders of magnitude slower.
        /// </remarks>
        /// <exception cref="InvalidOperationException">
        /// When <see cref="Type.GetElementType()">Array.GetType().GetElementType()</see> does not return the .NET element type.
        /// </exception>
        /// <exception cref="NotSupportedException">
        /// When <see cref="Type.GetElementType()">Array.GetType().GetElementType()</see> returns an unsupported .NET element type.
        /// Supported element types are <see cref="bool" />, <see cref="byte" />, <see cref="sbyte" />, <see cref="short" />,
        /// <see cref="int" />, <see cref="long" />, <see cref="float" />, <see cref="double" />,
        /// and <see cref="System.Numerics.Complex" />.
        /// </exception>
        /// <example>
        /// Tensor from array of rank 1
        /// <code>
        /// var array = new double[] { { 1, 2, 3, 4, 5, 6, 7, 8 } };
        /// var tensor = torch.from_array(rawArray: array, dtype: ScalarType.Float64, device: Device.CPU, requires_grad: false);
        /// </code>
        /// Tensor from array of rank 2
        /// <code>
        /// var array = new double[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } };
        /// var tensor = torch.from_array(rawArray: array, dtype: ScalarType.Float64, device: Device.CPU, requires_grad: false);
        /// </code>
        /// Tensor from array of rank 3
        /// <code>
        /// var array = new double[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } };
        /// var tensor = torch.from_array(rawArray: array, dtype: ScalarType.Float64, device: Device.CPU, requires_grad: false);
        /// </code>
        /// </example>
        [Pure]
        public static Tensor from_array(Array rawArray, Device? device = null)
        {
            var t = rawArray.GetType().GetElementType();
            if (t is null) throw new InvalidOperationException($"{nameof(rawArray)}.GetType().GetElementType() returned null.");
            if (t == typeof((float, float))) throw new NotImplementedException("from_array() for (float,float) elements.");

            var dtype = ToScalarType(t!);

            return from_array(rawArray, dtype, device is null ? CPU : device);
        }

        /// <summary>
        /// Creates a <see cref="torch.Tensor">torch tensor</see> from an arbitrary <see cref="Array">array</see>.
        /// </summary>
        /// <param name="rawArray">The arbitrary array to create the tensor from.</param>
        /// <param name="dtype">The element type to use in the created tensor. This can be different from the element type of the input.</param>
        /// <param name="device">The device where the tensor is to be located. Defaults to 'cpu'.</param>
        /// <param name="requires_grad"></param>
        /// <param name="names"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        [Pure]
        public static Tensor from_array(Array rawArray, ScalarType? dtype, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            // enumerates over all dimensions of the arbitrary array
            // and returns the length of the dimension
            [Pure]
            static long[] GetShape(Array arr)
            {
                var shape = new long[arr.Rank];
                for (var dim = 0; dim < arr.Rank; dim++)
                    shape[dim] = arr.GetLength(dim);
                return shape;
            }
            var shape = GetShape(rawArray);

            var t = rawArray.GetType().GetElementType();
            if (t is null) throw new InvalidOperationException($"{nameof(rawArray)}.GetType().GetElementType() returned null.");

            // call the existing factory methods to construct the tensor

            if (t == typeof((float, float))) return tensor(rawArray.Cast<(float, float)>().ToArray(), shape, dtype, device, requires_grad, names: names);

            var origType = ToScalarType(t);
            return _tensor_generic(rawArray, shape, (sbyte)origType, dtype, device, requires_grad, false, names);
        }

        // https://pytorch.org/docs/stable/generated/torch.as_tensor
        public static Tensor as_tensor(Array data, ScalarType? dtype = null, Device? device = null) => from_array(data, dtype, device, false);

        /// <summary>
        /// Creates a 1-dimensional Tensor from an array of n dimensions.
        ///
        /// Skips the first offset bytes in the buffer, and interprets the rest of the raw bytes as a 1-dimensional tensor of type dtype with count elements.
        /// </summary>
        /// <param name="rawArray">The input array</param>
        /// <param name="dtype">The torch data type.</param>
        /// <param name="count"></param>
        /// <param name="offset"></param>
        /// <param name="requires_grad">Set <value>true</value> if gradients need to be computed for this Tensor; <value>false</value> otherwise.</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <remarks>The returned tensor and buffer share the same memory. Modifications to the tensor will be reflected in the buffer and vice versa.</remarks>
        public static Tensor frombuffer(Array rawArray, ScalarType dtype, long count = -1, long offset = 0, bool requires_grad = false)
        {
            InitializeDeviceType(DeviceType.CPU);

            var lLength = rawArray.LongLength;

            if (offset < 0 || offset >= lLength) {
                throw new ArgumentException($"invalid value for 'offset': {offset}");
            }

            if (count < 0) { count = lLength - offset; }

            if (count > lLength - offset) {
                throw new IndexOutOfRangeException($"element count is too large: {count}");
            }

            var t = rawArray.GetType().GetElementType();
            ScalarType origType = ToScalarType(t!);

            switch (origType) {
            case ScalarType.Int16:
                offset *= 2;
                break;
            case ScalarType.Int32:
            case ScalarType.Float32:
                offset *= 4;
                break;
            case ScalarType.Int64:
            case ScalarType.Float64:
                offset *= 8;
                break;
            case ScalarType.ComplexFloat64:
                offset *= 16;
                break;
            case ScalarType.ComplexFloat32:
                // Since we are not allowed to make a copy of the buffer, it's currently not
                // feasible to support complex types, which GCHandle.Alloc() doesn't like.
                throw new NotImplementedException("frombuffer(ComplexFloat32)");
            }

            var dataHandle = GCHandle.Alloc(rawArray, GCHandleType.Pinned);
            var dataArrayAddr = dataHandle.AddrOfPinnedObject();
            var gchp = GCHandle.ToIntPtr(dataHandle);
            TorchSharp.PInvoke.GCHandleDeleter deleter = null!;
            deleter = new TorchSharp.PInvoke.GCHandleDeleter((IntPtr ptr) => {
                GCHandle.FromIntPtr(gchp).Free();
                deleters.TryRemove(deleter, out deleter!);
            });
            deleters.TryAdd(deleter, deleter); // keep the delegate alive

            unsafe {
                var handle = THSTensor_frombuffer(dataArrayAddr, deleter, count, offset, (sbyte)origType, requires_grad);

                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_frombuffer(dataArrayAddr, deleter, count, offset, (sbyte)origType, requires_grad);
                }

                if (handle == IntPtr.Zero) { CheckForErrors(); }
                var tensor = new Tensor(handle);

                var needsConversion = dtype != origType;

                if (needsConversion) {
                    tensor = tensor.to_type(dtype);
                }
                return tensor;
            }
        }

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = values.dtype;
            }

            unsafe {
                fixed (long* psizes = size) {
                    var handle = THSTensor_sparse(indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse(indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)dtype, (int)device.type, device.index, requires_grad);
                    }
                    if (handle == IntPtr.Zero) { CheckForErrors(); }
                    var tensor = new Tensor(handle);
                    if (names != null && names.Length > 0) {
                        tensor.rename_(names);
                    }
                    return tensor;
                }
            }
        }

        /// <summary>
        /// Constructs a complex tensor with its real part equal to real and its imaginary part equal to imag.
        /// </summary>
        public static Tensor complex(Tensor real, Tensor imag)
        {
            var res = THSTensor_complex(real.Handle, imag.Handle);
            if (res == IntPtr.Zero)
                CheckForErrors();
            return new Tensor(res);
        }

        /// <summary>
        /// Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value 'abs' and angle 'angle'.
        /// </summary>
        public static Tensor polar(Tensor abs, Tensor angle)
        {
            var res = THSTensor_polar(abs.Handle, angle.Handle);
            if (res == IntPtr.Zero)
                CheckForErrors();
            return new Tensor(res);
        }

        public static Tensor from_file(string filename, bool? shared = null, long? size = 0, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            var handle = THSTensor_from_file(StringEncoder.GetNullTerminatedUTF8ByteArray(filename), (sbyte)(!shared.HasValue ? -1 : shared.Value ? 1 : 0), size.HasValue ? size.Value : -1, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Create a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        /// </summary>
        public static Tensor linspace(double start, double end, long steps, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            var handle = THSTensor_linspace(start, end, steps, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_linspace(start, end, steps, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            }
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        /// <summary>
        /// Creates a one-dimensional tensor of size steps whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with base 'base.'
        /// </summary>
        public static Tensor logspace(double start, double end, long steps, double @base = 10, ScalarType? dtype = null, Device? device = null, bool requires_grad = false)
        {
            device = InitializeDevice(device);
            if (!dtype.HasValue) {
                // Determine the element type dynamically.
                dtype = get_default_dtype();
            }

            var handle = THSTensor_logspace(start, end, steps, @base, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_logspace(start, end, steps, @base, (sbyte)dtype, (int)device.type, device.index, requires_grad);
            }
            if (handle == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(handle);
        }

        private static void ValidateIntegerRange(long value, ScalarType dtype, string argument)
        {
            switch (dtype) {
            case ScalarType.Byte:
                if (value < byte.MinValue || value > byte.MaxValue)
                    throw new ArgumentOutOfRangeException(argument, value, $"The value is outside the range of {dtype}");
                break;
            case ScalarType.Int8:
                if (value < sbyte.MinValue || value > sbyte.MaxValue)
                    throw new ArgumentOutOfRangeException(argument, value, $"The value is outside the range of {dtype}");
                break;
            case ScalarType.Int16:
                if (value < short.MinValue || value > short.MaxValue)
                    throw new ArgumentOutOfRangeException(argument, value, $"The value is outside the range of {dtype}");
                break;
            case ScalarType.Int32:
                if (value < int.MinValue || value > int.MaxValue)
                    throw new ArgumentOutOfRangeException(argument, value, $"The value is outside the range of {dtype}");
                break;
            default:
                break;
            }
        }

        private static ConcurrentDictionary<TorchSharp.PInvoke.GCHandleDeleter, TorchSharp.PInvoke.GCHandleDeleter> deleters;
        private static ScalarType default_dtype = ScalarType.Float32;

        static torch()
        {
            deleters = new ConcurrentDictionary<TorchSharp.PInvoke.GCHandleDeleter, TorchSharp.PInvoke.GCHandleDeleter>();
        }
    }
}
