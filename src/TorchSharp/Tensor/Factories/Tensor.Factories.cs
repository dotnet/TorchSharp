// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using Utils;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
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
#if NET6_0_OR_GREATER
        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<Half> rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray.ToArray(), dimensions, (sbyte)ScalarType.Float16, dtype, device, requires_grad, false, names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(Half[] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.LongLength }, (sbyte)ScalarType.Float16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        [Pure]
        public static Tensor tensor(Half[] rawArray, ReadOnlySpan<long> dimensions, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, dimensions, (sbyte)ScalarType.Float16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        [Pure]
        public static Tensor tensor(IList<Half> rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<Half> rawArray, long rows, long columns, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<Half> rawArray, long dim0, long dim1, long dim2, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        public static Tensor tensor(IList<Half> rawArray, long dim0, long dim1, long dim2, long dim3, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return tensor(rawArray, stackalloc long[] { dim0, dim1, dim2, dim3 }, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a two-dimensional tensor from a two-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(Half[,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1) }, (sbyte)ScalarType.Float16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a three-dimensional tensor from a three-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(Half[,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2) }, (sbyte)ScalarType.Float16, dtype, device, requires_grad, names: names);
        }

        /// <summary>
        /// Create a four-dimensional tensor from a four-dimensional array of values.
        /// </summary>
        [Pure]
        public static Tensor tensor(Half[,,,] rawArray, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
        {
            return _tensor_generic(rawArray, stackalloc long[] { rawArray.GetLongLength(0), rawArray.GetLongLength(1), rawArray.GetLongLength(2), rawArray.GetLongLength(3) }, (sbyte)ScalarType.Float16, dtype, device, requires_grad, names: names);
        }
#endif
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
