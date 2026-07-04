// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Collections.Concurrent;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Utils;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
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
        /// <param name="dtype">The element type of the tensor to be created.</param>
        /// <param name="device">The device where the tensor should be located.</param>
        /// <param name="requires_grad">Whether the tensor should be tracking gradients.</param>
        /// <param name="generator">An optional random number generator</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        /// <returns></returns>
        public static Tensor normal(double mean, double std, ReadOnlySpan<long> size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, Generator? generator = null, string[]? names = null)
        {
            return randn(size, dtype, device, requires_grad: false, generator, names)
                .mul_(std).add_(mean).requires_grad_(requires_grad);
        }

        /// <summary>
        /// This private method handles most tensor creation, centrally.
        /// </summary>
        /// <param name="rawArray">An array of contiguous elements, to be used for creating the tensor.</param>
        /// <param name="dimensions">The dimensions of the tensor.</param>
        /// <param name="origType">The element type of the input array.</param>
        /// <param name="dtype">The element type of the tensor to be created.</param>
        /// <param name="device">The device where the tensor should be located.</param>
        /// <param name="requires_grad">Whether the tensor should be tracking gradients.</param>
        /// <param name="clone">Whether to clone the input array or use it as the backing storage for the tensor.</param>
        /// <param name="names">Names of the dimensions of the tensor.</param>
        /// <returns>A constructed tensor with elements of `dtype`</returns>
        /// <exception cref="ArgumentException"></exception>
        private static Tensor _tensor_generic(Array rawArray, ReadOnlySpan<long> dimensions, sbyte origType, ScalarType? dtype, Device? device, bool requires_grad, bool clone = true, string[]? names = null)
        {
            {
                // Validate the sizes before handing over storage to native code...
                var prod = 1L;
                foreach (var sz in dimensions) prod *= sz;

                if (origType == (sbyte)ScalarType.ComplexFloat32)
                    prod *= 2;

                if (prod > rawArray.LongLength)
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

            dtype = dtype.HasValue ? dtype : (ScalarType)origType;

            unsafe {
                void *ptr = null;
                IntPtr iPtr = (IntPtr)ptr;

                fixed (long* shape = dimensions) {
                    var handle = THSTensor_new(dataArrayAddr, deleter, (IntPtr)shape, dimensions.Length, origType, (sbyte)dtype.Value, (int)device.type, device.index, requires_grad);

                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_new(dataArrayAddr, deleter, (IntPtr)shape, dimensions.Length, origType, (sbyte)dtype.Value, (int)device.type, device.index, requires_grad);
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

        private static Tensor _tensor_generic<T>(Memory<T> rawArray, ReadOnlySpan<long> dimensions, sbyte origType, ScalarType? dtype, Device? device, bool requires_grad, bool clone = true, string[]? names = null)
        {
            if (clone)
            {
                return _tensor_generic(rawArray.ToArray(), dimensions, origType, dtype, device, requires_grad, false, names);
            }

            {
                // Validate the sizes before handing over storage to native code...
                var prod = 1L;
                foreach (var sz in dimensions) prod *= sz;

                if (origType == (sbyte)ScalarType.ComplexFloat32)
                    prod *= 2;

                if (prod > rawArray.Length)
                    throw new ArgumentException($"mismatched total size creating a tensor from an array: {prod} vs. {rawArray.Length}");
            }

            device = InitializeDevice(device);

            dtype = dtype.HasValue ? dtype : (ScalarType)origType;

            unsafe {

                var dataHandle = rawArray.Pin();
                var dataArrayAddr = (IntPtr)dataHandle.Pointer;

                TorchSharp.PInvoke.GCHandleDeleter deleter = null!;
                deleter = new TorchSharp.PInvoke.GCHandleDeleter((IntPtr ptr) => {
                    dataHandle.Dispose();
                    deleters.TryRemove(deleter, out deleter!);
                });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive

                void *ptr = null;
                IntPtr iPtr = (IntPtr)ptr;

                fixed (long* shape = dimensions) {
                    var handle = THSTensor_new(dataArrayAddr, deleter, (IntPtr)shape, dimensions.Length, origType, (sbyte)dtype.Value, (int)device.type, device.index, requires_grad);

                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_new(dataArrayAddr, deleter, (IntPtr)shape, dimensions.Length, origType, (sbyte)dtype.Value, (int)device.type, device.index, requires_grad);
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
            var lLength = rawArray.LongLength;

            if (offset < 0 || offset >= lLength) {
                throw new ArgumentException($"invalid value for 'offset': {offset}");
            }

            if (count < 0) { count = lLength - offset; }

            if (count > lLength - offset) {
                throw new IndexOutOfRangeException($"element count is too large: {count}");
            }

            var device = InitializeDevice(torch.CPU);

            var t = rawArray.GetType().GetElementType();
            ScalarType origType = ToScalarType(t!);

            switch (origType) {
            case ScalarType.Int16:
            case ScalarType.Float16:
            case ScalarType.BFloat16:
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
                var handle = THSTensor_frombuffer(dataArrayAddr, deleter, count, offset, (sbyte)origType, (sbyte)dtype, (int)device.type, device.index, requires_grad);

                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_frombuffer(dataArrayAddr, deleter, count, offset, (sbyte)origType, (sbyte)dtype, (int)device.type, device.index, requires_grad);
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
        /// Constructs a sparse tensor in COO(rdinate) format with specified values at the given indices.
        /// </summary>
        /// <param name="indices">
        /// Initial data for the tensor. The indices are the coordinates of the non-zero values in the matrix,
        /// and thus should be two-dimensional where the first dimension is the number of tensor dimensions
        /// and the second dimension is the number of non-zero values.
        /// </param>
        /// <param name="values">nitial values for the tensor.</param>
        /// <param name="size">Size of the sparse tensor.</param>
        /// <param name="dtype">The element type to use in the created tensor. This can be different from the element type of the input.</param>
        /// <param name="device">The device where the tensor is to be located. Defaults to 'cpu'.</param>
        /// <param name="requires_grad">>Set <value>true</value> if gradients need to be computed for this Tensor; <value>false</value> otherwise.</param>
        /// <param name="names">Names for the dimensions of the tensor.</param>
        /// <returns></returns>
        public static Tensor sparse_coo_tensor(Tensor indices, Tensor values, long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
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
        /// Constructs a sparse tensor in COO(rdinate) format with specified values at the given indices.
        /// </summary>
        /// <param name="indices">
        /// Initial data for the tensor. The indices are the coordinates of the non-zero values in the matrix,
        /// and thus should be two-dimensional where the first dimension is the number of tensor dimensions
        /// and the second dimension is the number of non-zero values.
        /// </param>
        /// <param name="values">nitial values for the tensor.</param>
        /// <param name="size">Size of the sparse tensor.</param>
        /// <param name="dtype">The element type to use in the created tensor. This can be different from the element type of the input.</param>
        /// <param name="device">The device where the tensor is to be located. Defaults to 'cpu'.</param>
        /// <param name="requires_grad">>Set <value>true</value> if gradients need to be computed for this Tensor; <value>false</value> otherwise.</param>
        /// <param name="names">Names for the dimensions of the tensor.</param>
        [Obsolete("This method had the wrong name when it was added. Use torch.sparse_coo_tensor instead.")]
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, ScalarType? dtype = null, Device? device = null, bool requires_grad = false, string[]? names = null)
            => sparse(indices, values, size, dtype, device, requires_grad, names);

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

        #region Loading a tensor from a stream
        public partial class Tensor
        {
            /// <summary>
            /// Load a tensor using a .NET-specific format.
            /// </summary>
            /// <param name="reader">A BinaryReader instance</param>
            public static Tensor Load(System.IO.BinaryReader reader)
            {
                // First, read the type
                var type = (ScalarType)reader.Decode();

                // Then, the shape
                var shLen = reader.Decode();
                long[] loadedShape = new long[shLen];

                long totalSize = 1;
                for (int i = 0; i < shLen; ++i) {
                    loadedShape[i] = reader.Decode();
                    totalSize *= loadedShape[i];
                }

                //
                // TODO: Fix this so that you can read large tensors. Right now, they are limited to 2GB
                //
                if (totalSize > int.MaxValue)
                    throw new NotImplementedException("Loading tensors larger than 2GB");

                var tensor = torch.empty(loadedShape, dtype: type);
                tensor.ReadBytesFromStream(reader.BaseStream);

                return tensor;
            }

            /// <summary>
            /// Load a tensor using a .NET-specific format.
            /// </summary>
            /// <param name="stream">A stream opened for reading binary data.</param>
            public static Tensor Load(System.IO.Stream stream)
            {
                using var reader = new System.IO.BinaryReader(stream);
                return Load(reader);
            }

            /// <summary>
            /// Load a tensor using a .NET-specific format.
            /// </summary>
            /// <param name="location">A file name.</param>
            public static Tensor Load(string location)
            {
                if (!System.IO.File.Exists(location))
                    throw new System.IO.FileNotFoundException(location);

                using var stream = System.IO.File.OpenRead(location);
                return Load(stream);
            }
        }
        #endregion

        #region Private declarations
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
        private static Device default_device = new Device(DeviceType.CPU, -1);

        static torch()
        {
            deleters = new ConcurrentDictionary<TorchSharp.PInvoke.GCHandleDeleter, TorchSharp.PInvoke.GCHandleDeleter>();
            TryInitializeDeviceType(DeviceType.CUDA);
        }
        #endregion
    }
}
