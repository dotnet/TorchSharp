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
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(TorchScalar start, TorchScalar stop, TorchScalar step, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor RandomIntegers(long max, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newByteScalar(byte scalar, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor From(byte scalar, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);
            var handle = THSTensor_newByteScalar(scalar, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        public static TorchTensor From(byte[] rawArray, long[] dimensions, bool requiresGrad = false)
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
        
        public static TorchTensor From(byte[] rawArray, bool requiresGrad = false)
        {
            return From(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad);
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
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(TorchScalar start, TorchScalar stop, TorchScalar step, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor RandomIntegers(long max, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt8Scalar(sbyte scalar, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor From(sbyte scalar, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);
            var handle = THSTensor_newInt8Scalar(scalar, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        public static TorchTensor From(sbyte[] rawArray, long[] dimensions, bool requiresGrad = false)
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
        
        public static TorchTensor From(sbyte[] rawArray, bool requiresGrad = false)
        {
            return From(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int8, (int) deviceType, deviceIndex, requiresGrad);
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
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(TorchScalar start, TorchScalar stop, TorchScalar step, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor RandomIntegers(long max, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt16Scalar(short scalar, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor From(short scalar, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);
            var handle = THSTensor_newInt16Scalar(scalar, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        public static TorchTensor From(short[] rawArray, long[] dimensions, bool requiresGrad = false)
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
        
        public static TorchTensor From(short[] rawArray, bool requiresGrad = false)
        {
            return From(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int16, (int) deviceType, deviceIndex, requiresGrad);
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
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(TorchScalar start, TorchScalar stop, TorchScalar step, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor RandomIntegers(long max, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt32Scalar(int scalar, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor From(int scalar, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);
            var handle = THSTensor_newInt32Scalar(scalar, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        public static TorchTensor From(int[] rawArray, long[] dimensions, bool requiresGrad = false)
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
        
        public static TorchTensor From(int[] rawArray, bool requiresGrad = false)
        {
            return From(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int32, (int) deviceType, deviceIndex, requiresGrad);
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
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(TorchScalar start, TorchScalar stop, TorchScalar step, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor RandomIntegers(long max, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt64Scalar(long scalar, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor From(long scalar, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);
            var handle = THSTensor_newInt64Scalar(scalar, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newInt64(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

        public static TorchTensor From(long[] rawArray, long[] dimensions, bool requiresGrad = false)
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
        
        public static TorchTensor From(long[] rawArray, bool requiresGrad = false)
        {
            return From(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int64, (int) deviceType, deviceIndex, requiresGrad);
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
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(TorchScalar start, TorchScalar stop, TorchScalar step, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor RandomIntegers(long max, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor Random(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor RandomN(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat16Scalar(float scalar, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor From(float scalar, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);
            var handle = THSTensor_newFloat16Scalar(scalar, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat16(IntPtr rawArray, IntPtr dataArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

        public static TorchTensor From(float[] rawArray, long[] dimensions, bool requiresGrad = false)
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
        
        public static TorchTensor From(float[] rawArray, bool requiresGrad = false)
        {
            return From(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float16, (int) deviceType, deviceIndex, requiresGrad);
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
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(TorchScalar start, TorchScalar stop, TorchScalar step, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor RandomIntegers(long max, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor Random(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor RandomN(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newBFloat16Scalar(float scalar, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor From(float scalar, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);
            var handle = THSTensor_newBFloat16Scalar(scalar, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newBFloat16(IntPtr rawArray, IntPtr dataArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

        public static TorchTensor From(float[] rawArray, long[] dimensions, bool requiresGrad = false)
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
        
        public static TorchTensor From(float[] rawArray, bool requiresGrad = false)
        {
            return From(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.BFloat16, (int) deviceType, deviceIndex, requiresGrad);
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
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(TorchScalar start, TorchScalar stop, TorchScalar step, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor RandomIntegers(long max, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor Random(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor RandomN(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat32Scalar(float scalar, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor From(float scalar, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);
            var handle = THSTensor_newFloat32Scalar(scalar, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        public static TorchTensor From(float[] rawArray, long[] dimensions, bool requiresGrad = false)
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
        
        public static TorchTensor From(float[] rawArray, bool requiresGrad = false)
        {
            return From(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float32, (int) deviceType, deviceIndex, requiresGrad);
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
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(TorchScalar start, TorchScalar stop, TorchScalar step, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor RandomIntegers(long max, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor Random(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor RandomN(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloat64Scalar(double scalar, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor From(double scalar, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);
            var handle = THSTensor_newFloat64Scalar(scalar, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        public static TorchTensor From(double[] rawArray, long[] dimensions, bool requiresGrad = false)
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
        
        public static TorchTensor From(double[] rawArray, bool requiresGrad = false)
        {
            return From(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float64, (int) deviceType, deviceIndex, requiresGrad);
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
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(TorchScalar start, TorchScalar stop, TorchScalar step, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            var handle = THSTensor_randperm (n, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                handle = THSTensor_randperm (n, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
            }
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (handle);
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint(long max, IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with random integer values taken from a uniform distribution in [0, max).
        /// </summary>
        static public TorchTensor RandomIntegers(long max, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newBoolScalar(bool scalar, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor From(bool scalar, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);
            var handle = THSTensor_newBoolScalar(scalar, (int) deviceType, deviceIndex, requiresGrad);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, sbyte type, bool requiresGrad);

        public static TorchTensor From(bool[] rawArray, long[] dimensions, bool requiresGrad = false)
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
        
        public static TorchTensor From(bool[] rawArray, bool requiresGrad = false)
        {
            return From(rawArray, new long[] { (long)rawArray.Length }, requiresGrad);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDeviceType (deviceType);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    var handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
                    if (handle == IntPtr.Zero) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        handle = THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad);
                    }
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor (handle);
                }
            }
        }
    }
}
