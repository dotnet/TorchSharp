using System;
using System.Runtime.InteropServices;

namespace TorchSharp.Tensor {

    /// <summary>
    ///   Tensor of type Byte.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do not mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class ByteTensor
    {
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, string device, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(byte start, byte stop, byte step, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            return new TorchTensor (THSTensor_arange (start.ToScalar().Handle, stop.ToScalar().Handle, step.ToScalar().Handle, (sbyte)ATenScalarMapping.Byte, device, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Byte, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Byte, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Byte, device, requiresGrad));
                }
            }
        }

                [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor Random(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Byte, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Byte, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newByteScalar(byte scalar);

        public static TorchTensor From(byte scalar)
        {
            return new TorchTensor(THSTensor_newByteScalar(scalar));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static TorchTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new TorchTensor(THSTensor_new(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Byte));
        }

        public static TorchTensor From(byte[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (byte* parray = rawArray)
                {
                    return ByteTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        public static TorchTensor From(byte[] rawArray)
        {
            return From(rawArray, new long[] { (long)rawArray.Length });
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Byte, device, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Short.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do not mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class ShortTensor
    {
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, string device, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(short start, short stop, short step, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            return new TorchTensor (THSTensor_arange (start.ToScalar().Handle, stop.ToScalar().Handle, step.ToScalar().Handle, (sbyte)ATenScalarMapping.Short, device, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Short, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Short, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Short, device, requiresGrad));
                }
            }
        }

                [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor Random(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Short, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Short, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newShortScalar(short scalar);

        public static TorchTensor From(short scalar)
        {
            return new TorchTensor(THSTensor_newShortScalar(scalar));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static TorchTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new TorchTensor(THSTensor_new(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Short));
        }

        public static TorchTensor From(short[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (short* parray = rawArray)
                {
                    return ShortTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        public static TorchTensor From(short[] rawArray)
        {
            return From(rawArray, new long[] { (long)rawArray.Length });
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Short, device, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Int.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do not mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class IntTensor
    {
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, string device, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(int start, int stop, int step, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            return new TorchTensor (THSTensor_arange (start.ToScalar().Handle, stop.ToScalar().Handle, step.ToScalar().Handle, (sbyte)ATenScalarMapping.Int, device, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Int, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Int, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Int, device, requiresGrad));
                }
            }
        }

                [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor Random(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Int, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Int, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newIntScalar(int scalar);

        public static TorchTensor From(int scalar)
        {
            return new TorchTensor(THSTensor_newIntScalar(scalar));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static TorchTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new TorchTensor(THSTensor_new(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Int));
        }

        public static TorchTensor From(int[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (int* parray = rawArray)
                {
                    return IntTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        public static TorchTensor From(int[] rawArray)
        {
            return From(rawArray, new long[] { (long)rawArray.Length });
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Int, device, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Long.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do not mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class LongTensor
    {
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, string device, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(long start, long stop, long step, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            return new TorchTensor (THSTensor_arange (start.ToScalar().Handle, stop.ToScalar().Handle, step.ToScalar().Handle, (sbyte)ATenScalarMapping.Long, device, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Long, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Long, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Long, device, requiresGrad));
                }
            }
        }

                [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor Random(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Long, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Long, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newLongScalar(long scalar);

        public static TorchTensor From(long scalar)
        {
            return new TorchTensor(THSTensor_newLongScalar(scalar));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newLong(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static TorchTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new TorchTensor(THSTensor_newLong(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Long));
        }

        public static TorchTensor From(long[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (long* parray = rawArray)
                {
                    return LongTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        public static TorchTensor From(long[] rawArray)
        {
            return From(rawArray, new long[] { (long)rawArray.Length });
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Long, device, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Double.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do not mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class DoubleTensor
    {
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, string device, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(double start, double stop, double step, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            return new TorchTensor (THSTensor_arange (start.ToScalar().Handle, stop.ToScalar().Handle, step.ToScalar().Handle, (sbyte)ATenScalarMapping.Double, device, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Double, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Double, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Double, device, requiresGrad));
                }
            }
        }

                [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor Random(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Double, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Double, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newDoubleScalar(double scalar);

        public static TorchTensor From(double scalar)
        {
            return new TorchTensor(THSTensor_newDoubleScalar(scalar));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static TorchTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new TorchTensor(THSTensor_new(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Double));
        }

        public static TorchTensor From(double[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (double* parray = rawArray)
                {
                    return DoubleTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        public static TorchTensor From(double[] rawArray)
        {
            return From(rawArray, new long[] { (long)rawArray.Length });
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Double, device, requiresGrad));
                }
            }
        }
    }
    /// <summary>
    ///   Tensor of type Float.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    ///   Please do not mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class FloatTensor
    {
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, int scalarType, string device, bool requireGrad);

        /// <summary>
        /// Creates 1-D tensor of size [(end - start) / step] with values from interval [start, end) and
		/// common difference step, starting from start
        /// </summary>
        static public TorchTensor Arange(float start, float stop, float step, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            return new TorchTensor (THSTensor_arange (start.ToScalar().Handle, stop.ToScalar().Handle, step.ToScalar().Handle, (sbyte)ATenScalarMapping.Float, device, requiresGrad));
        }
		
		[DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public TorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Float, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Float, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public TorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_empty ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Float, device, requiresGrad));
                }
            }
        }

                [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        static public TorchTensor Random(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_rand ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Float, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public TorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Float, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_newFloatScalar(float scalar);

        public static TorchTensor From(float scalar)
        {
            return new TorchTensor(THSTensor_newFloatScalar(scalar));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static TorchTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new TorchTensor(THSTensor_new(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Float));
        }

        public static TorchTensor From(float[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (float* parray = rawArray)
                {
                    return FloatTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        public static TorchTensor From(float[] rawArray)
        {
            return From(rawArray, new long[] { (long)rawArray.Length });
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(TorchTensor indices, TorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
        {
            TorchTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new TorchTensor (THSTensor_sparse (indices.Handle, values.Handle, (IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Float, device, requiresGrad));
                }
            }
        }
    }
}
