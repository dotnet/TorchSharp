using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp.Tensor
{
    public struct TorchTensor : ITorchTensor
    {
        internal IntPtr handle;

        internal TorchTensor(IntPtr handle)
        {
            this.handle = handle;
        }

        /// <summary>
        ///   Releases the tensor and its associated data.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_dispose(IntPtr handle);

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        public void Dispose(bool disposing)
        {
            if (disposing)
            {
                THSTensor_dispose(handle);
                handle = IntPtr.Zero;
            }
        }

        public IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        [DllImport("libTorchSharp")]
        extern static long THSTensor_ndimension(IntPtr handle);

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public long Dimensions
        {
            get
            {
                return THSTensor_ndimension(handle);
            }
        }

        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public long NumberOfElements
        {
            get
            {
                switch (Dimensions)
                {
                    case 0:
                        return 1;
                    case 1:
                        return (int)Shape[0];
                    default:
                        return (int)Shape.Aggregate((x, y) => x * y);
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_data(IntPtr handle);

        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public Span<T> Data<T>()
        {
            if (NumberOfElements > int.MaxValue)
            {
                throw new ArgumentException("Span only supports up to int.MaxValue elements.");
            }
            unsafe
            {
                return new System.Span<T>((void*)THSTensor_data(handle), (int)NumberOfElements);
            }
        }

        public T DataItem<T>()
        {
            if (NumberOfElements != 1)
            {
                throw new ArgumentException($"Number of elements in the tensor must be 1");
            }
            return Data<T>()[0];
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_get1(IntPtr handle, long i1);

        public ITorchTensor this[long i1]
        {
            get
            {
                return new TorchTensor(THSTensor_get1(handle, i1));
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_get2(IntPtr handle, long i1, long i2);

        public ITorchTensor this[long i1, long i2]
        {
            get
            {
                return new TorchTensor(THSTensor_get2(handle, i1, i2));
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_get3(IntPtr handle, long i1, long i2, long i3);

        public ITorchTensor this[long i1, long i2, long i3]
        {
            get
            {
                return new TorchTensor(THSTensor_get3(handle, i1, i2, i3));
            }
        }

        [DllImport("libTorchSharp")]
        extern static sbyte THSTensor_type(IntPtr handle);

        public ATenScalarMapping Type
        {
            get
            {
                return (ATenScalarMapping)THSTensor_type(handle);
            }
        }

        [DllImport("libTorchSharp")]
        extern static string THSTensor_deviceType(IntPtr handle);

        public string Device
        {
            get
            {
                return THSTensor_deviceType(handle);
            }
        }

        [DllImport("libTorchSharp")]
        extern static bool THSTensor_isSparse(IntPtr handle);

        public bool IsSparse
        {
            get
            {
                return THSTensor_isSparse(handle);
            }
        }

        [DllImport("libTorchSharp")]
        extern static bool THSTensor_isVariable(IntPtr handle);

        public bool IsVariable
        {
            get
            {
                return THSTensor_isVariable(handle);
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_cpu(IntPtr handle);

        public ITorchTensor Cpu()
        {
            return new TorchTensor(THSTensor_cpu(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_cuda(IntPtr handle);

        public ITorchTensor Cuda()
        {
            if (!Torch.IsCudaAvailable())
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }

            return new TorchTensor(THSTensor_cuda(handle));
        }

        [DllImport("libTorchSharp")]
        extern static long THSTensor_size(IntPtr handle, long dimension);

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            return THSTensor_size(handle, dim);
        }

        /// <summary>
        /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
        /// </summary>
        /// <remarks>
        ///     An array of size 0 is used for constants, an array of size 1 is used
        ///     for single-dimension arrays, where the dimension is the value of the
        ///     first element.   And so on.
        /// </remarks>
        public long[] Shape
        {
            get
            {
                var dims = new long[Dimensions];
                for (int i = 0; i < dims.Length; i++)
                    dims[i] = GetTensorDimension(i);

                return dims;
            }
        }

        [DllImport("libTorchSharp")]
        extern static long THSTensor_stride(IntPtr handle, long dimension);

        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride(int dim)
        {
            return THSTensor_stride(handle, dim);
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_backward(IntPtr handle);

        public void Backward()
        {
            THSTensor_backward(handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_grad(IntPtr handle);

        public ITorchTensor Grad()
        {
            return new TorchTensor(THSTensor_grad(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_reshape(IntPtr src, IntPtr shape, int length);

        public ITorchTensor Reshape(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new TorchTensor(THSTensor_reshape(handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_t(IntPtr src);

        public ITorchTensor T()
        {
            return new TorchTensor(THSTensor_t(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_transpose(IntPtr src, long dim1, long dim2);

        public ITorchTensor Transpose(long dimension1, long dimension2)
        {
            return new TorchTensor(THSTensor_transpose(handle, dimension1, dimension2));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_transpose_(IntPtr src, long dim1, long dim2);

        public void TransposeInPlace(long dimension1, long dimension2)
        {
            THSTensor_transpose_(handle, dimension1, dimension2);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_view(IntPtr src, IntPtr shape, int length);

        public ITorchTensor View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new TorchTensor(THSTensor_view(handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_add(IntPtr src, int scalar, IntPtr trg);

        public ITorchTensor Add(ITorchTensor target, int scalar = 1)
        {
            return new TorchTensor(THSTensor_add(handle, scalar, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_add_(IntPtr src, int scalar, IntPtr trg);

        public void AddInPlace(ITorchTensor target, int scalar = 1)
        {
            THSTensor_add_(handle, scalar, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public ITorchTensor Addbmm(ITorchTensor batch1, ITorchTensor batch2, float beta, float alpha)
        {
            return new TorchTensor(THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_addmm(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        public ITorchTensor Addmm(ITorchTensor mat1, ITorchTensor mat2, float beta, float alpha)
        {
            return new TorchTensor(THSTensor_addmm(handle, mat1.Handle, mat2.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_argmax(IntPtr src, long dimension, bool keep_dim);

        public ITorchTensor Argmax(long dimension, bool keepDim = false)
        {
            return new TorchTensor(THSTensor_argmax(handle, dimension, keepDim));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta, float alpha);

        public ITorchTensor Baddbmm(ITorchTensor batch2, ITorchTensor mat, float beta, float alpha)
        {
            return new TorchTensor(THSTensor_addbmm(handle, batch2.Handle, mat.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_bmm(IntPtr batch1, IntPtr batch2);

        public ITorchTensor Bmm(ITorchTensor batch2)
        {
            return new TorchTensor(THSTensor_bmm(handle, batch2.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_eq(IntPtr src, IntPtr trg);

        public ITorchTensor Eq(ITorchTensor target)
        {
            return new TorchTensor(THSTensor_eq(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static bool THSTensor_equal(IntPtr src, IntPtr trg);

        public bool Equal(ITorchTensor target)
        {
            return THSTensor_equal(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_exp(IntPtr src);

        public ITorchTensor Exp()
        {
            return new TorchTensor(THSTensor_exp(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_matmul(IntPtr src, IntPtr target);

        public ITorchTensor MatMul(ITorchTensor target)
        {
            return new TorchTensor(THSTensor_matmul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mean(IntPtr src);

        public ITorchTensor Mean()
        {
            return new TorchTensor(THSTensor_mean(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mm(IntPtr src, IntPtr target);

        public ITorchTensor Mm(ITorchTensor target)
        {
            return new TorchTensor(THSTensor_mm(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mul(IntPtr src, IntPtr target);

        public ITorchTensor Mul(ITorchTensor target)
        {
            return new TorchTensor(THSTensor_mul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_mul_(IntPtr src, IntPtr target);

        public void MulInPlace(ITorchTensor target)
        {
            THSTensor_mul_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mulS(IntPtr src, float scalar);

        public ITorchTensor Mul(float scalar)
        {
            return new TorchTensor(THSTensor_mulS(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_pow(IntPtr src, float scalar);

        public ITorchTensor Pow(float scalar)
        {
            return new TorchTensor(THSTensor_pow(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sigmoid(IntPtr src);

        public ITorchTensor Sigmoid()
        {
            return new TorchTensor(THSTensor_sigmoid(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sub(IntPtr src, IntPtr trg);

        public ITorchTensor Sub(ITorchTensor target)
        {
            return new TorchTensor(THSTensor_sub(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_sub_(IntPtr src, IntPtr trg);

        public void SubInPlace(ITorchTensor target)
        {
            THSTensor_sub_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sum(IntPtr src);

        public ITorchTensor Sum()
        {
            return new TorchTensor(THSTensor_sum(handle));
        }

        // Operators overloading
        public static ITorchTensor operator +(TorchTensor left, ITorchTensor right)
        {
            return left.Add(right);
        }

        public static ITorchTensor operator *(TorchTensor left, ITorchTensor right)
        {
            return left.Mul(right);
        }

        public static ITorchTensor operator -(TorchTensor left, ITorchTensor right)
        {
            return left.Sub(right);
        }

        /// <summary>
        ///   Get a string representation of the tensor.
        /// </summary>
        public override string ToString()
        {
            var n = Dimensions;
            if (n == 0)
                return "[]";

            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < n; i++)
            {
                sb.Append(GetTensorDimension(i));
                if (i + 1 < n)
                    sb.Append("x");
            }
            sb.Append("]");
            sb.Append($", device = {Device}");
            return sb.ToString();
        }

        internal static void CheckForCUDA(string device)
        {
            if (!Torch.IsCudaAvailable() && device.ToLower().Contains("cuda"))
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }
        }
    }

    public enum ATenScalarMapping : sbyte
    {
        Byte = 0,
        Short = 2,
        Int = 3,
        Long = 4,
        Float = 6,
        Double = 7
    }

    public static class TensorExtensionMethods
    {
        public static ITorchTensor ToTorchTensor<T>(this T[] rawArray, long[] dimensions)
        {
            switch (true)
            {
                case bool _ when typeof(T) == typeof(byte):
                    {
                        return ByteTensor.From(rawArray as byte[], dimensions);
                    }
                case bool _ when typeof(T) == typeof(short):
                    {
                        return ShortTensor.From(rawArray as short[], dimensions);
                    }
                case bool _ when typeof(T) == typeof(int):
                    {
                        return IntTensor.From(rawArray as int[], dimensions);
                    }
                case bool _ when typeof(T) == typeof(long):
                    {
                        return LongTensor.From(rawArray as long[], dimensions);
                    }
                case bool _ when typeof(T) == typeof(double):
                    {
                        return DoubleTensor.From(rawArray as double[], dimensions);
                    }
                case bool _ when typeof(T) == typeof(float):
                    {
                        return FloatTensor.From(rawArray as float[], dimensions);
                    }
                default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        public static ITorchTensor ToTorchTensor<T>(this T scalar)
        {
            switch (true)
            {
                case bool _ when typeof(T) == typeof(byte):
                    {
                        return ByteTensor.From((byte)(object)scalar);
                    }
                case bool _ when typeof(T) == typeof(short):
                    {
                        return ShortTensor.From((short)(object)scalar);
                    }
                case bool _ when typeof(T) == typeof(int):
                    {
                        return IntTensor.From((int)(object)scalar);
                    }
                case bool _ when typeof(T) == typeof(long):
                    {
                        return LongTensor.From((long)(object)scalar);
                    }
                case bool _ when typeof(T) == typeof(double):
                    {
                        return DoubleTensor.From((double)(object)scalar);
                    }
                case bool _ when typeof(T) == typeof(float):
                    {
                        return FloatTensor.From((float)(object)scalar);
                    }
                default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_cat(IntPtr src, int len, long dim);

        public static ITorchTensor Cat<T>(this ITorchTensor[] tensors, long dimension)
        {
            var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            return new TorchTensor(THSTensor_cat(tensorsRef, parray.Array.Length, dimension));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_stack(IntPtr src, int len, long dim);

        public static TorchTensor Stack(this TorchTensor[] tensors, long dimension)
        {
            var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            return new TorchTensor(THSTensor_stack(tensorsRef, parray.Array.Length, dimension));
        }
    }
}
