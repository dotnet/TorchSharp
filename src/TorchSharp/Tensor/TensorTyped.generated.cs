using System;
using System.Runtime.InteropServices;
using System.Collections.Concurrent;

// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
namespace TorchSharp {

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void GCHandleDeleter(IntPtr memory);


    public static partial class torch
    {

    /// <summary>
    ///   Tensor of type Byte.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class ByteTensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static ByteTensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }




        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newByteScalar(byte scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor from(byte scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newByteScalar(scalar, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(byte[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray;
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Byte, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Byte, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(byte[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from(byte[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from(byte[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from(byte[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Int8.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class Int8Tensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static Int8Tensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }




        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt8Scalar(sbyte scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor from(sbyte scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newInt8Scalar(scalar, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(sbyte[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray;
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int8, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int8, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(sbyte[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from(sbyte[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from(sbyte[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from(sbyte[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Int16.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class Int16Tensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static Int16Tensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }




        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt16Scalar(short scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor from(short scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newInt16Scalar(scalar, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(short[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray;
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int16, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int16, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(short[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from(short[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from(short[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from(short[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Int32.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class Int32Tensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static Int32Tensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }




        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt32Scalar(int scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor from(int scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newInt32Scalar(scalar, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(int[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray;
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int32, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int32, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(int[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from(int[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from(int[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from(int[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Int64.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class Int64Tensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static Int64Tensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }




        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt64Scalar(long scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor from(long scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newInt64Scalar(scalar, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt64(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(long[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray;
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_newInt64(dataArrayAddr, deleter, dimensions, dimensions.Length, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_newInt64(dataArrayAddr, deleter, dimensions, dimensions.Length, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(long[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from(long[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from(long[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from(long[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Float16.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class Float16Tensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static Float16Tensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_fftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Computes the discrete Fourier Transform sample frequencies for a signal of size n.
        /// </summary>
        static public Tensor fftfreq(long n, double d = 1.0, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_fftfreq (n, d, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_fftfreq (n, d, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rfftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Computes the sample frequencies for rfft() with a signal of size n.
        /// </summary>
        static public Tensor rfftfreq(long n, double d = 1.0, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_rfftfreq (n, d, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_rfftfreq (n, d, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_linspace(double start, double end, long steps, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        /// </summary>
        static public Tensor linspace(float start, float end, long steps, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_logspace(double start, double end, long steps, double @base, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Creates a one-dimensional tensor of size steps whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with base 'base.'
        /// </summary>
        static public Tensor logspace(float start, float end, long steps, float @base = 10, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_bartlett_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Bartlett window function.
        /// </summary>
        static public Tensor bartlett_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_bartlett_window (len, periodic, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_bartlett_window (len, periodic, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_blackman_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Blackman window function.
        /// </summary>
        static public Tensor blackman_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_blackman_window (len, periodic, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_blackman_window (len, periodic, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hamming_window(long len, bool periodic, double alpha, double beta, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Hamming window function.
        /// </summary>
        static public Tensor hamming_window(long len, bool periodic = true, float alpha = 0.54f, float beta = 0.46f, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_hamming_window (len, periodic, alpha, beta, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hamming_window (len, periodic, alpha, beta, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hann_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Hann window function.
        /// </summary>
        static public Tensor hann_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_hann_window (len, periodic, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hann_window (len, periodic, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_kaiser_window(long len, bool periodic, double beta, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Computes the Kaiser window with window length window_length and shape parameter beta.
        /// </summary>
        static public Tensor kaiser_window(long len, bool periodic = true, float beta = 12.0f, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_kaiser_window (len, periodic, beta, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_kaiser_window (len, periodic, beta, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat16Scalar(float scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor from(float scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newFloat16Scalar(scalar, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat16(IntPtr rawArray, IntPtr dataArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(float[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = new Int16[rawArray.Length];
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                fixed (float* pRawArray = rawArray)
                {
                    var handle = THSTensor_newFloat16((IntPtr)pRawArray, dataArrayAddr, deleter, dimensions, dimensions.Length, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_newFloat16((IntPtr)pRawArray, dataArrayAddr, deleter, dimensions, dimensions.Length, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(handle);
                }
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(float[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from(float[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from(float[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from(float[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type BFloat16.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class BFloat16Tensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static BFloat16Tensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_fftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Computes the discrete Fourier Transform sample frequencies for a signal of size n.
        /// </summary>
        static public Tensor fftfreq(long n, double d = 1.0, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_fftfreq (n, d, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_fftfreq (n, d, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rfftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Computes the sample frequencies for rfft() with a signal of size n.
        /// </summary>
        static public Tensor rfftfreq(long n, double d = 1.0, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_rfftfreq (n, d, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_rfftfreq (n, d, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_linspace(double start, double end, long steps, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        /// </summary>
        static public Tensor linspace(float start, float end, long steps, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_logspace(double start, double end, long steps, double @base, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Creates a one-dimensional tensor of size steps whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with base 'base.'
        /// </summary>
        static public Tensor logspace(float start, float end, long steps, float @base = 10, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_bartlett_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Bartlett window function.
        /// </summary>
        static public Tensor bartlett_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_bartlett_window (len, periodic, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_bartlett_window (len, periodic, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_blackman_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Blackman window function.
        /// </summary>
        static public Tensor blackman_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_blackman_window (len, periodic, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_blackman_window (len, periodic, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hamming_window(long len, bool periodic, double alpha, double beta, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Hamming window function.
        /// </summary>
        static public Tensor hamming_window(long len, bool periodic = true, float alpha = 0.54f, float beta = 0.46f, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_hamming_window (len, periodic, alpha, beta, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hamming_window (len, periodic, alpha, beta, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hann_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Hann window function.
        /// </summary>
        static public Tensor hann_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_hann_window (len, periodic, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hann_window (len, periodic, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_kaiser_window(long len, bool periodic, double beta, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Computes the Kaiser window with window length window_length and shape parameter beta.
        /// </summary>
        static public Tensor kaiser_window(long len, bool periodic = true, float beta = 12.0f, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_kaiser_window (len, periodic, beta, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_kaiser_window (len, periodic, beta, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newBFloat16Scalar(float scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor from(float scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newBFloat16Scalar(scalar, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newBFloat16(IntPtr rawArray, IntPtr dataArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(float[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = new Int16[rawArray.Length];
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                fixed (float* pRawArray = rawArray)
                {
                    var handle = THSTensor_newBFloat16((IntPtr)pRawArray, dataArrayAddr, deleter, dimensions, dimensions.Length, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_newBFloat16((IntPtr)pRawArray, dataArrayAddr, deleter, dimensions, dimensions.Length, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(handle);
                }
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(float[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from(float[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from(float[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from(float[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Float32.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class Float32Tensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static Float32Tensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_fftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Computes the discrete Fourier Transform sample frequencies for a signal of size n.
        /// </summary>
        static public Tensor fftfreq(long n, double d = 1.0, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_fftfreq (n, d, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_fftfreq (n, d, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rfftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Computes the sample frequencies for rfft() with a signal of size n.
        /// </summary>
        static public Tensor rfftfreq(long n, double d = 1.0, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_rfftfreq (n, d, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_rfftfreq (n, d, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_linspace(double start, double end, long steps, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        /// </summary>
        static public Tensor linspace(float start, float end, long steps, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_logspace(double start, double end, long steps, double @base, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Creates a one-dimensional tensor of size steps whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with base 'base.'
        /// </summary>
        static public Tensor logspace(float start, float end, long steps, float @base = 10, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_bartlett_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Bartlett window function.
        /// </summary>
        static public Tensor bartlett_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_bartlett_window (len, periodic, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_bartlett_window (len, periodic, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_blackman_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Blackman window function.
        /// </summary>
        static public Tensor blackman_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_blackman_window (len, periodic, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_blackman_window (len, periodic, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hamming_window(long len, bool periodic, double alpha, double beta, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Hamming window function.
        /// </summary>
        static public Tensor hamming_window(long len, bool periodic = true, float alpha = 0.54f, float beta = 0.46f, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_hamming_window (len, periodic, alpha, beta, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hamming_window (len, periodic, alpha, beta, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hann_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Hann window function.
        /// </summary>
        static public Tensor hann_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_hann_window (len, periodic, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hann_window (len, periodic, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_kaiser_window(long len, bool periodic, double beta, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Computes the Kaiser window with window length window_length and shape parameter beta.
        /// </summary>
        static public Tensor kaiser_window(long len, bool periodic = true, float beta = 12.0f, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_kaiser_window (len, periodic, beta, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_kaiser_window (len, periodic, beta, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat32Scalar(float scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor from(float scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newFloat32Scalar(scalar, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(float[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray;
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Float32, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Float32, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(float[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from(float[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from(float[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from(float[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Float64.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class Float64Tensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static Float64Tensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_fftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Computes the discrete Fourier Transform sample frequencies for a signal of size n.
        /// </summary>
        static public Tensor fftfreq(long n, double d = 1.0, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_fftfreq (n, d, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_fftfreq (n, d, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rfftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Computes the sample frequencies for rfft() with a signal of size n.
        /// </summary>
        static public Tensor rfftfreq(long n, double d = 1.0, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_rfftfreq (n, d, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_rfftfreq (n, d, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_linspace(double start, double end, long steps, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        /// </summary>
        static public Tensor linspace(double start, double end, long steps, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_logspace(double start, double end, long steps, double @base, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Creates a one-dimensional tensor of size steps whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with base 'base.'
        /// </summary>
        static public Tensor logspace(double start, double end, long steps, double @base = 10, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_bartlett_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Bartlett window function.
        /// </summary>
        static public Tensor bartlett_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_bartlett_window (len, periodic, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_bartlett_window (len, periodic, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_blackman_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Blackman window function.
        /// </summary>
        static public Tensor blackman_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_blackman_window (len, periodic, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_blackman_window (len, periodic, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hamming_window(long len, bool periodic, double alpha, double beta, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Hamming window function.
        /// </summary>
        static public Tensor hamming_window(long len, bool periodic = true, double alpha = 0.54f, double beta = 0.46f, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_hamming_window (len, periodic, alpha, beta, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hamming_window (len, periodic, alpha, beta, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hann_window(long len, bool periodic, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Hann window function.
        /// </summary>
        static public Tensor hann_window(long len, bool periodic = true, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_hann_window (len, periodic, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_hann_window (len, periodic, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_kaiser_window(long len, bool periodic, double beta, sbyte scalar_type, int device_type, int device_index, bool requires_grad);

        /// <summary>
        /// Computes the Kaiser window with window length window_length and shape parameter beta.
        /// </summary>
        static public Tensor kaiser_window(long len, bool periodic = true, double beta = 12.0f, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_kaiser_window (len, periodic, beta, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_kaiser_window (len, periodic, beta, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat64Scalar(double scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor from(double scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newFloat64Scalar(scalar, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(double[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray;
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Float64, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Float64, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(double[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from(double[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from(double[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from(double[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type ComplexFloat32.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public partial class ComplexFloat32Tensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static ComplexFloat32Tensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }


        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }



        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from((float Real, float Imaginary)[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = new float[rawArray.Length * 2];
            for (var i = 0; i < rawArray.Length; i++) {
                dataArray[i*2] = rawArray[i].Real;
                dataArray[i*2 + 1] = rawArray[i].Imaginary;
            }
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.ComplexFloat32, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.ComplexFloat32, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from((float Real, float Imaginary)[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from((float Real, float Imaginary)[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from((float Real, float Imaginary)[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from((float Real, float Imaginary)[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat32, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type ComplexFloat64.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public partial class ComplexFloat64Tensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static ComplexFloat64Tensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }


        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public Tensor rand(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return rand(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public Tensor randn(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return randn(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }



        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(System.Numerics.Complex[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = new double[rawArray.Length * 2];
            for (var i = 0; i < rawArray.Length; i++) {
                dataArray[i*2] = rawArray[i].Real;
                dataArray[i*2 + 1] = rawArray[i].Imaginary;
            }
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.ComplexFloat64, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.ComplexFloat64, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(System.Numerics.Complex[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from(System.Numerics.Complex[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from(System.Numerics.Complex[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from(System.Numerics.Complex[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.ComplexFloat64, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Bool.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class BoolTensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static BoolTensor()
        {
            deleters = new ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter>();
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, Scalar step, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - start) / step] with values from interval [start, stop) and
		/// common difference step, starting from start
        /// </summary>
        static public Tensor arange(Scalar start, Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(start, stop, 1, device, requiresGrad);
        }

        /// <summary>
        /// Creates 1-D tensor of size [(stop - 0)] with values from interval [0, stop), starting from 0
        /// </summary>
        static public Tensor arange(Scalar stop, torch.Device device = null, bool requiresGrad = false)
        {
            return arange(0, stop, 1, device, requiresGrad);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public Tensor randperm(long n, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with zeros
        /// </summary>
        static public Tensor zeros(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return zeros(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public Tensor ones(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with ones
        /// </summary>
        static public Tensor ones(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return ones(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long size, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { size }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long rows, long columns, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { rows, columns }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2 }, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with uninitialized data
        /// </summary>
        static public Tensor empty(long dim0, long dim1, long dim2, long dim3, torch.Device device = null, bool requiresGrad = false)
        {
            return empty(new long[] { dim0, dim1, dim2, dim3 }, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public Tensor empty_strided(long[] size, long[] strides, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with a given value
        /// </summary>
        static public Tensor full(long[] size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }

        /// <summary>
        ///  Create a new 1-D tensor filled with given value
        /// </summary>
        static public Tensor full(long size, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { size }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 2-D tensor filled with given value
        /// </summary>
        static public Tensor full(long rows, long columns, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { rows, columns }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 3-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2 }, value, device, requiresGrad);
        }

        /// <summary>
        ///  Create a new 4-D tensor filled with given value
        /// </summary>
        static public Tensor full(long dim0, long dim1, long dim2, long dim3, Scalar value, torch.Device device = null, bool requiresGrad = false)
        {
            return full(new long[] { dim0, dim1, dim2, dim3 }, value, device, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public Tensor eye(long rows, long columns = -1L, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public Tensor randint(long max, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }




        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newBoolScalar(bool scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static Tensor from(bool scalar, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);
            var handle = THSTensor_newBoolScalar(scalar, (int) device.type, device.index, requiresGrad);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(bool[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            var dataArray = rawArray;
            unsafe
            {
                var dataHandle = GCHandle.Alloc(dataArray, GCHandleType.Pinned);
                var dataArrayAddr = dataHandle.AddrOfPinnedObject();
                var gchp = GCHandle.ToIntPtr(dataHandle);
                GCHandleDeleter deleter = null;
                deleter =
                    new GCHandleDeleter(delegate (IntPtr ptr) {
                        GCHandle.FromIntPtr(gchp).Free();
                        deleters.TryRemove(deleter, out deleter);
                        });
                deleters.TryAdd(deleter, deleter); // keep the delegate alive
                var handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Bool, requiresGrad);
                if (handle == IntPtr.Zero) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    handle = THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Bool, requiresGrad);
                }
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(handle);
            }
        }

        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the input array.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static Tensor from(bool[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a two-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have rows * columns elements.
        /// </remarks>
        public static Tensor from(bool[] rawArray, long rows, long columns, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { rows, columns }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a three-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2 elements.
        /// </remarks>
        public static Tensor from(bool[] rawArray, long dim0, long dim1, long dim2, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2 }, requiresGrad);
        }

        /// <summary>
        /// Create a tensor from an array of values, organizing it as a four-dimensional tensor.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.
        ///          The input array must have dim0*dim1*dim2*dim3 elements.
        /// </remarks>
        public static Tensor from(bool[] rawArray, long dim0, long dim1, long dim2, long dim3, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { dim0, dim1, dim2, dim3 }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static Tensor sparse(Tensor indices, Tensor values, long[] size, torch.Device device = null, bool requiresGrad = false)
        {
            device = torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.type, device.index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor (handle);
                }
            }
        }
    }
}
}