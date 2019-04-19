using System;
using System.Runtime.InteropServices;

namespace TorchSharp.Tensor {

    /// <summary>
    ///   Tensor of type Byte.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class ByteTensor
    {
        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public ITorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_newByteScalar(byte scalar);

        public static TorchTensor From(byte scalar)
        {
            return new TorchTensor(THSTensor_newByteScalar(scalar));
        }

        [DllImport("libTorchSharp")]
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(ITorchTensor indices, ITorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
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
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class ShortTensor
    {
        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public ITorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_newShortScalar(short scalar);

        public static TorchTensor From(short scalar)
        {
            return new TorchTensor(THSTensor_newShortScalar(scalar));
        }

        [DllImport("libTorchSharp")]
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(ITorchTensor indices, ITorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
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
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class IntTensor
    {
        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public ITorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_newIntScalar(int scalar);

        public static TorchTensor From(int scalar)
        {
            return new TorchTensor(THSTensor_newIntScalar(scalar));
        }

        [DllImport("libTorchSharp")]
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(ITorchTensor indices, ITorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
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
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class LongTensor
    {
        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public ITorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_newLongScalar(long scalar);

        public static TorchTensor From(long scalar)
        {
            return new TorchTensor(THSTensor_newLongScalar(scalar));
        }

        [DllImport("libTorchSharp")]
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(ITorchTensor indices, ITorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
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
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class DoubleTensor
    {
        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public ITorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_newDoubleScalar(double scalar);

        public static TorchTensor From(double scalar)
        {
            return new TorchTensor(THSTensor_newDoubleScalar(scalar));
        }

        [DllImport("libTorchSharp")]
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(ITorchTensor indices, ITorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
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
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class FloatTensor
    {
        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Ones(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_empty(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor Empty(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        static public ITorchTensor RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_newFloatScalar(float scalar);

        public static TorchTensor From(float scalar)
        {
            return new TorchTensor(THSTensor_newFloatScalar(scalar));
        }

        [DllImport("libTorchSharp")]
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

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, string device, bool requiresGrad);

        public static TorchTensor Sparse(ITorchTensor indices, ITorchTensor values, long[] size, string device = "cpu", bool requiresGrad = false)
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
