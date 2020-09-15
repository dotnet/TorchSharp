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
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad));
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_randperm (n, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newByteScalar(byte scalar, bool requiresGrad);

        public static TorchTensor From(byte scalar, bool requiresGrad = false)
        {
            return new TorchTensor(THSTensor_newByteScalar(scalar, requiresGrad));
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
                return new TorchTensor(THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Byte, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Byte, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type SByte.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class SByteTensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static SByteTensor()
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.SByte, (int) deviceType, deviceIndex, requiresGrad));
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_randperm (n, (sbyte)ScalarType.SByte, (int) deviceType, deviceIndex, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.SByte, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.SByte, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.SByte, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.SByte, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newSByteScalar(sbyte scalar, bool requiresGrad);

        public static TorchTensor From(sbyte scalar, bool requiresGrad = false)
        {
            return new TorchTensor(THSTensor_newSByteScalar(scalar, requiresGrad));
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
                return new TorchTensor(THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.SByte, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.SByte, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Short.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class ShortTensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static ShortTensor()
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Short, (int) deviceType, deviceIndex, requiresGrad));
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_randperm (n, (sbyte)ScalarType.Short, (int) deviceType, deviceIndex, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Short, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Short, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Short, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Short, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newShortScalar(short scalar, bool requiresGrad);

        public static TorchTensor From(short scalar, bool requiresGrad = false)
        {
            return new TorchTensor(THSTensor_newShortScalar(scalar, requiresGrad));
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
                return new TorchTensor(THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Short, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Short, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Int.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class IntTensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static IntTensor()
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Int, (int) deviceType, deviceIndex, requiresGrad));
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_randperm (n, (sbyte)ScalarType.Int, (int) deviceType, deviceIndex, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Int, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newIntScalar(int scalar, bool requiresGrad);

        public static TorchTensor From(int scalar, bool requiresGrad = false)
        {
            return new TorchTensor(THSTensor_newIntScalar(scalar, requiresGrad));
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
                return new TorchTensor(THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Int, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Int, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Long.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class LongTensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static LongTensor()
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Long, (int) deviceType, deviceIndex, requiresGrad));
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_randperm (n, (sbyte)ScalarType.Long, (int) deviceType, deviceIndex, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Long, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Long, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Long, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Long, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newLongScalar(long scalar, bool requiresGrad);

        public static TorchTensor From(long scalar, bool requiresGrad = false)
        {
            return new TorchTensor(THSTensor_newLongScalar(scalar, requiresGrad));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newLong(IntPtr rawArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

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
                return new TorchTensor(THSTensor_newLong(dataArrayAddr, deleter, dimensions, dimensions.Length, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Long, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Half.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class HalfTensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static HalfTensor()
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Half, (int) deviceType, deviceIndex, requiresGrad));
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_randperm (n, (sbyte)ScalarType.Half, (int) deviceType, deviceIndex, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Half, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Half, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Half, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Half, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Half, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Half, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newHalfScalar(float scalar, bool requiresGrad);

        public static TorchTensor From(float scalar, bool requiresGrad = false)
        {
            return new TorchTensor(THSTensor_newHalfScalar(scalar, requiresGrad));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newHalf(IntPtr rawArray, IntPtr dataArray, GCHandleDeleter deleter, long[] dimensions, int numDimensions, bool requiresGrad);

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
                    return new TorchTensor(THSTensor_newHalf((IntPtr)pRawArray, dataArrayAddr, deleter, dimensions, dimensions.Length, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Half, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Float.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class FloatTensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static FloatTensor()
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Float, (int) deviceType, deviceIndex, requiresGrad));
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_randperm (n, (sbyte)ScalarType.Float, (int) deviceType, deviceIndex, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requiresGrad);

        /// <summary>
        ///  Create a new tensor filled with arbitrary values
        /// </summary>
        static public TorchTensor Empty(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Float, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloatScalar(float scalar, bool requiresGrad);

        public static TorchTensor From(float scalar, bool requiresGrad = false)
        {
            return new TorchTensor(THSTensor_newFloatScalar(scalar, requiresGrad));
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
                return new TorchTensor(THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Float, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Float, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Double.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class DoubleTensor
    {
        static private ConcurrentDictionary<GCHandleDeleter, GCHandleDeleter> deleters;
        static DoubleTensor()
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Double, (int) deviceType, deviceIndex, requiresGrad));
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_randperm (n, (sbyte)ScalarType.Double, (int) deviceType, deviceIndex, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Double, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Double, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Double, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Double, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Double, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Double, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newDoubleScalar(double scalar, bool requiresGrad);

        public static TorchTensor From(double scalar, bool requiresGrad = false)
        {
            return new TorchTensor(THSTensor_newDoubleScalar(scalar, requiresGrad));
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
                return new TorchTensor(THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Double, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Double, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_arange (start.Handle, stop.Handle, step.Handle, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad));
        }
		
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm(long n, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        static public TorchTensor RandomPermutation(long n, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            return new TorchTensor (THSTensor_randperm (n, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int length, int scalarType, int deviceType, int deviceIndex, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false)
        {
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randint (max, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newBoolScalar(bool scalar, bool requiresGrad);

        public static TorchTensor From(bool scalar, bool requiresGrad = false)
        {
            return new TorchTensor(THSTensor_newBoolScalar(scalar, requiresGrad));
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
                return new TorchTensor(THSTensor_new(dataArrayAddr, deleter, dimensions, dimensions.Length, (sbyte)ScalarType.Bool, requiresGrad));
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
            Torch.InitializeDevice (deviceType, deviceIndex);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ScalarType.Bool, (int) deviceType, deviceIndex, requiresGrad));
                }
            }
        }
    }
}
