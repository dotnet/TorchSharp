using System;
using System.Runtime.InteropServices;
using System.Collections.Concurrent;

// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
namespace TorchSharp.Tensor {

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void GCHandleDeleter(IntPtr memory);

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
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor randperm(long n, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor zeros(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor ones(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor empty(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public TorchTensor empty_strided(long[] size, long[] strides, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor full(long[] size, TorchScalar value, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public TorchTensor eye(long rows, long columns = -1L, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newByteScalar(byte scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(byte scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newByteScalar(scalar, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(byte[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
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
                if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(handle);
            }
        }
        
        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(byte[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static TorchTensor sparse(TorchTensor indices, TorchTensor values, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
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
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor randperm(long n, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor zeros(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor ones(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor empty(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public TorchTensor empty_strided(long[] size, long[] strides, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor full(long[] size, TorchScalar value, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public TorchTensor eye(long rows, long columns = -1L, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt8Scalar(sbyte scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(sbyte scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newInt8Scalar(scalar, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(sbyte[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
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
                if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(handle);
            }
        }
        
        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(sbyte[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static TorchTensor sparse(TorchTensor indices, TorchTensor values, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
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
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor randperm(long n, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor zeros(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor ones(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor empty(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public TorchTensor empty_strided(long[] size, long[] strides, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor full(long[] size, TorchScalar value, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public TorchTensor eye(long rows, long columns = -1L, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt16Scalar(short scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(short scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newInt16Scalar(scalar, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(short[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
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
                if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(handle);
            }
        }
        
        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(short[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static TorchTensor sparse(TorchTensor indices, TorchTensor values, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
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
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor randperm(long n, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor zeros(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor ones(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor empty(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public TorchTensor empty_strided(long[] size, long[] strides, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor full(long[] size, TorchScalar value, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public TorchTensor eye(long rows, long columns = -1L, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt32Scalar(int scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(int scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newInt32Scalar(scalar, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(int[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
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
                if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(handle);
            }
        }
        
        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(int[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static TorchTensor sparse(TorchTensor indices, TorchTensor values, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
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
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor randperm(long n, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor zeros(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor ones(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor empty(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public TorchTensor empty_strided(long[] size, long[] strides, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor full(long[] size, TorchScalar value, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public TorchTensor eye(long rows, long columns = -1L, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt64Scalar(long scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(long scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newInt64Scalar(scalar, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt64(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(long[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
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
                if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(handle);
            }
        }
        
        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(long[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static TorchTensor sparse(TorchTensor indices, TorchTensor values, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
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
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor randperm(long n, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor zeros(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor ones(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor empty(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public TorchTensor empty_strided(long[] size, long[] strides, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor full(long[] size, TorchScalar value, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public TorchTensor eye(long rows, long columns = -1L, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_linspace(double start, double end, long steps, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        /// </summary>
        static public TorchTensor linspace(float start, float end, long steps, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_logspace(double start, double end, long steps, double @base, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Creates a one-dimensional tensor of size steps whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with base 'base.'
        /// </summary>
        static public TorchTensor logspace(float start, float end, long steps, float @base = 10, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor rand(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor randn(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat16Scalar(float scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(float scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newFloat16Scalar(scalar, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat16(IntPtr rawArray, IntPtr dataArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(float[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
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
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(handle);
                }
            }
        }
        
        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(float[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static TorchTensor sparse(TorchTensor indices, TorchTensor values, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
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
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor randperm(long n, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor zeros(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor ones(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor empty(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public TorchTensor empty_strided(long[] size, long[] strides, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor full(long[] size, TorchScalar value, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public TorchTensor eye(long rows, long columns = -1L, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_linspace(double start, double end, long steps, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        /// </summary>
        static public TorchTensor linspace(float start, float end, long steps, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_logspace(double start, double end, long steps, double @base, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Creates a one-dimensional tensor of size steps whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with base 'base.'
        /// </summary>
        static public TorchTensor logspace(float start, float end, long steps, float @base = 10, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor rand(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor randn(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newBFloat16Scalar(float scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(float scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newBFloat16Scalar(scalar, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newBFloat16(IntPtr rawArray, IntPtr dataArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(float[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
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
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(handle);
                }
            }
        }
        
        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(float[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static TorchTensor sparse(TorchTensor indices, TorchTensor values, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
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
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor randperm(long n, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor zeros(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor ones(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor empty(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public TorchTensor empty_strided(long[] size, long[] strides, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor full(long[] size, TorchScalar value, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public TorchTensor eye(long rows, long columns = -1L, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_linspace(double start, double end, long steps, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        /// </summary>
        static public TorchTensor linspace(float start, float end, long steps, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_logspace(double start, double end, long steps, double @base, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Creates a one-dimensional tensor of size steps whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with base 'base.'
        /// </summary>
        static public TorchTensor logspace(float start, float end, long steps, float @base = 10, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor rand(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor randn(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat32Scalar(float scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(float scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newFloat32Scalar(scalar, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(float[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
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
                if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(handle);
            }
        }
        
        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(float[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static TorchTensor sparse(TorchTensor indices, TorchTensor values, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
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
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor randperm(long n, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor zeros(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor ones(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor empty(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public TorchTensor empty_strided(long[] size, long[] strides, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor full(long[] size, TorchScalar value, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public TorchTensor eye(long rows, long columns = -1L, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_linspace(double start, double end, long steps, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        /// </summary>
        static public TorchTensor linspace(double start, double end, long steps, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_linspace (start, end, steps, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_logspace(double start, double end, long steps, double @base, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Creates a one-dimensional tensor of size steps whose values are evenly spaced from base^start to base^end, inclusive, on a logarithmic scale with base 'base.'
        /// </summary>
        static public TorchTensor logspace(double start, double end, long steps, double @base = 10, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_logspace (start, end, steps, @base, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor rand(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor randn(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat64Scalar(double scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(double scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newFloat64Scalar(scalar, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(double[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
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
                if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(handle);
            }
        }
        
        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(double[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static TorchTensor sparse(TorchTensor indices, TorchTensor values, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
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
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor arange(TorchScalar start, TorchScalar stop, TorchScalar step, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor randperm(long n, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor zeros(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor ones(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor empty(long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Returns a tensor filled with uninitialized data. The shape and strides of the tensor is defined by the variable argument size and stride respectively.
        /// </summary>
        static public TorchTensor empty_strided(long[] size, long[] strides, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size, pstrides = strides)
                {
                    var handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty_strided ((IntPtr)psizes, size.Length, (IntPtr)pstrides, strides.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with uninitialized data
        /// </summary>
        static public TorchTensor full(long[] size, TorchScalar value, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_full ((IntPtr)psizes, size.Length, value.Handle, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// </summary>
        static public TorchTensor eye(long rows, long columns = -1L, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            columns = (columns == -1) ? rows : columns;

            var handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_eye (rows, columns, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor randint(long max, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newBoolScalar(bool scalar, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a scalar tensor from a single value
        /// </summary>
        public static TorchTensor from(bool scalar, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);
            var handle = THSTensor_newBoolScalar(scalar, (int) device.Type, device.Index, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        /// <summary>
        /// Create a tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(bool[] rawArray, long[] dimensions, bool requiresGrad = false)
        {
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
                if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(handle);
            }
        }
        
        /// <summary>
        /// Create a 1-D tensor from an array of values, shaping it based on the shape passed in.
        /// </summary>
        /// <remarks>The Torch runtime does not take ownership of the data, so there is no device argument.</remarks>
        public static TorchTensor from(bool[] rawArray, bool requiresGrad = false)
        {
            return from(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        /// Create a sparse tensor by indexing into an existing dense tensor.
        /// </summary>
        public static TorchTensor sparse(TorchTensor indices, TorchTensor values, long[] size, Device device = null, bool requiresGrad = false)
        {
            device = Torch.InitializeDevice(device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) device.Type, device.Index, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }
    }
}
