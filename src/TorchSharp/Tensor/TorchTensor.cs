using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp.Tensor
{
    public struct TorchTensor : IDisposable
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

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_dispose(IntPtr handle);

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

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_ndimension(IntPtr handle);

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

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_data(IntPtr handle);

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
                return new Span<T>((void*)THSTensor_data(handle), (int)NumberOfElements);
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

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_item(IntPtr handle);

        public Scalar Item()
        {
            var sptr = THSTensor_item(Handle);
            Torch.AssertNoErrors();
            return new Scalar(sptr);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_get1(IntPtr handle, long i1);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_set1(IntPtr handle, long i1, IntPtr value);

        [System.Runtime.CompilerServices.IndexerName("TensorItems")]
        public TorchTensor this[long i1]
        {
            get { return new TorchTensor(THSTensor_get1(handle, i1)); }
            set
            {
                THSTensor_set1(handle, i1, value.Item().Handle);
                Torch.AssertNoErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_get2(IntPtr handle, long i1, long i2);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_set2(IntPtr handle, long i1, long i2, IntPtr value);

        [System.Runtime.CompilerServices.IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2]
        {
            get
            {
                return new TorchTensor(THSTensor_get2(handle, i1, i2));
            }
            set
            {
                THSTensor_set2(handle, i1, i2, value.Item().Handle);
                Torch.AssertNoErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_get3(IntPtr handle, long i1, long i2, long i3);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_set3(IntPtr handle, long i1, long i2, long i3, IntPtr value);

        [System.Runtime.CompilerServices.IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3]
        {
            get
            {
                return new TorchTensor(THSTensor_get3(handle, i1, i2, i3));
            }
            set
            {
                THSTensor_set3(handle, i1, i2, i3, value.Item().Handle);
                Torch.AssertNoErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern sbyte THSTensor_type(IntPtr handle);

        public ATenScalarMapping Type
        {
            get
            {
                return (ATenScalarMapping)THSTensor_type(handle);
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern string THSTensor_deviceType(IntPtr handle);

        public string Device
        {
            get
            {
                return THSTensor_deviceType(handle);
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_isSparse(IntPtr handle);

        public bool IsSparse
        {
            get
            {
                return THSTensor_isSparse(handle);
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_isVariable(IntPtr handle);

        public bool IsVariable
        {
            get
            {
                return THSTensor_isVariable(handle);
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_to_type(IntPtr handle, sbyte scalar_type);

        public TorchTensor ToType(ATenScalarMapping type)
        {
            return new TorchTensor(THSTensor_to_type(handle, (sbyte)type));
        }

        [DllImport("LibTorchSharp")]
        extern static bool THSTensor_requires_grad(IntPtr handle);

        public bool IsGradRequired
        {
            get
            {
                return THSTensor_requires_grad(handle);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_set_requires_grad(IntPtr handle, bool requires_grad);

        public TorchTensor RequiresGrad(bool requiresGrad)
        {
            return new TorchTensor(THSTensor_set_requires_grad(handle, requiresGrad));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cpu(IntPtr handle);

        public TorchTensor Cpu()
        {
            return new TorchTensor(THSTensor_cpu(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cuda(IntPtr handle);

        public TorchTensor Cuda()
        {
            if (!Torch.IsCudaAvailable())
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }

            return new TorchTensor(THSTensor_cuda(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_size(IntPtr handle, long dimension);

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

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_indices(IntPtr handle);

        public TorchTensor Indices
        {
            get
            {
                return new TorchTensor(THSTensor_indices(handle));
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_values(IntPtr handle);

        public TorchTensor Values
        {
            get
            {
                return new TorchTensor(THSTensor_values(handle));
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_stride(IntPtr handle, long dimension);

        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride(int dim)
        {
            return THSTensor_stride(handle, dim);
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_backward(IntPtr handle);

        public void Backward()
        {
            THSTensor_backward(handle);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_to_dense(IntPtr handle);

        public TorchTensor ToDense()
        {
            return new TorchTensor(THSTensor_to_dense(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_clone(IntPtr handle);

        public TorchTensor Clone()
        {
            return new TorchTensor(THSTensor_clone(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_contiguous(IntPtr handle);

        public TorchTensor Contiguous()
        {
            return new TorchTensor(THSTensor_contiguous(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_grad(IntPtr handle);

        public TorchTensor Grad()
        {
            return new TorchTensor(THSTensor_grad(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_index_select(IntPtr src, long dimension, IntPtr index);

        public TorchTensor IndexSelect(long dimension, TorchTensor index)
        {
            return new TorchTensor(THSTensor_index_select(handle, dimension, index.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_reshape(IntPtr src, IntPtr shape, int length);

        public TorchTensor Reshape(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new TorchTensor(THSTensor_reshape(handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_squeeze(IntPtr src, long dimension);

        public TorchTensor Squeeze(long dimension)
        {
            return new TorchTensor(THSTensor_squeeze(handle, dimension));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_t(IntPtr src);

        public TorchTensor T()
        {
            return new TorchTensor(THSTensor_t(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_transpose(IntPtr src, long dim1, long dim2);

        public TorchTensor Transpose(long dimension1, long dimension2)
        {
            return new TorchTensor(THSTensor_transpose(handle, dimension1, dimension2));
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_transpose_(IntPtr src, long dim1, long dim2);

        public void TransposeInPlace(long dimension1, long dimension2)
        {
            THSTensor_transpose_(handle, dimension1, dimension2);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_view(IntPtr src, IntPtr shape, int length);

        public TorchTensor View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new TorchTensor(THSTensor_view(handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_add(IntPtr src, int scalar, IntPtr trg);

        public TorchTensor Add(TorchTensor target, int scalar = 1)
        {
            return new TorchTensor(THSTensor_add(handle, scalar, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_addS(IntPtr src, IntPtr trg);

        public TorchTensor Add(Scalar scalar)
        {
            return new TorchTensor(THSTensor_addS(handle, scalar.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_add_(IntPtr src, int scalar, IntPtr trg);

        public TorchTensor AddInPlace(TorchTensor target, int scalar = 1)
        {
            return new TorchTensor(THSTensor_add_(handle, scalar, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public TorchTensor Addbmm(TorchTensor batch1, TorchTensor batch2, float beta = 1, float alpha = 1)
        {
            return new TorchTensor(THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addmm(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        public TorchTensor Addmm(TorchTensor mat1, TorchTensor mat2, float beta, float alpha)
        {
            return new TorchTensor(THSTensor_addmm(handle, mat1.Handle, mat2.Handle, beta, alpha));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argmax(IntPtr src, long dimension, bool keep_dim);

        public TorchTensor Argmax(long dimension, bool keepDim = false)
        {
            return new TorchTensor(THSTensor_argmax(handle, dimension, keepDim));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta, float alpha);

        public TorchTensor Baddbmm(TorchTensor batch2, TorchTensor mat, float beta = 1, float alpha = 1)
        {
            return new TorchTensor(THSTensor_baddbmm(handle, batch2.Handle, mat.Handle, beta, alpha));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bmm(IntPtr batch1, IntPtr batch2);

        public TorchTensor Bmm(TorchTensor batch2)
        {
            return new TorchTensor(THSTensor_bmm(handle, batch2.Handle));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_clamp(IntPtr input, IntPtr min, IntPtr max);

        public TorchTensor Clamp(Scalar min, Scalar max)
        {
            return new TorchTensor(THSTensor_clamp(handle, min.Handle, max.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_div(IntPtr src, IntPtr trg);

        public TorchTensor Div(TorchTensor target)
        {
            return new TorchTensor(THSTensor_div(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_div_(IntPtr src, IntPtr trg);

        public void DivInPlace(TorchTensor target)
        {
            THSTensor_div_(handle, target.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_divS(IntPtr src, IntPtr trg);

        public TorchTensor Div(Scalar target)
        {
            return new TorchTensor(THSTensor_divS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_gtS(IntPtr src, IntPtr trg);

        public TorchTensor Gt(Scalar target)
        {
            return new TorchTensor(THSTensor_gtS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eq(IntPtr src, IntPtr trg);

        public TorchTensor Eq(TorchTensor target)
        {
            return new TorchTensor(THSTensor_eq(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_eqS(IntPtr src, IntPtr trg);

        public TorchTensor Eq(Scalar target)
        {
            return new TorchTensor(THSTensor_eqS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_equal(IntPtr src, IntPtr trg);

        public bool Equal(TorchTensor target)
        {
            return THSTensor_equal(handle, target.Handle);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_exp(IntPtr src);

        public TorchTensor Exp()
        {
            return new TorchTensor(THSTensor_exp(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_geS(IntPtr src, IntPtr trg);

        public TorchTensor Ge(Scalar target)
        {
            return new TorchTensor(THSTensor_geS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_matmul(IntPtr src, IntPtr target);

        public TorchTensor MatMul(TorchTensor target)
        {
            return new TorchTensor(THSTensor_matmul(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_max(IntPtr src, AllocatePinnedArray allocator, long dimension, bool keep_dim);

        public (TorchTensor values, TorchTensor indexes) Max(long dimension, bool keepDim = false)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSTensor_max(handle, pa.CreateArray, dimension, keepDim);
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mean(IntPtr src);

        public TorchTensor Mean()
        {
            return new TorchTensor(THSTensor_mean(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mm(IntPtr src, IntPtr target);

        public TorchTensor Mm(TorchTensor target)
        {
            return new TorchTensor(THSTensor_mm(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mul(IntPtr src, IntPtr target);

        public TorchTensor Mul(TorchTensor target)
        {
            return new TorchTensor(THSTensor_mul(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_mul_(IntPtr src, IntPtr target);

        public void MulInPlace(TorchTensor target)
        {
            THSTensor_mul_(handle, target.Handle);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_mulS(IntPtr src, IntPtr scalar);

        public TorchTensor Mul(Scalar scalar)
        {
            return new TorchTensor(THSTensor_mulS(handle, scalar.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_norm(IntPtr src, int dimension, bool keep_dimension);

        public TorchTensor Norm(int dimension, bool KeepDimension = false)
        {
            return new TorchTensor(THSTensor_norm(handle, dimension, KeepDimension));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_pow(IntPtr src, IntPtr scalar);

        public TorchTensor Pow(Scalar scalar)
        {
            return new TorchTensor(THSTensor_pow(handle, scalar.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sigmoid(IntPtr src);

        public TorchTensor Sigmoid()
        {
            return new TorchTensor(THSTensor_sigmoid(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sub(IntPtr src, IntPtr trg);

        public TorchTensor Sub(TorchTensor target)
        {
            return new TorchTensor(THSTensor_sub(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_sub_(IntPtr src, IntPtr trg);

        public void SubInPlace(TorchTensor target)
        {
            THSTensor_sub_(handle, target.Handle);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sum(IntPtr src);

        public TorchTensor Sum()
        {
            return new TorchTensor(THSTensor_sum(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sum1(IntPtr src, IntPtr dimensions, int length, bool keep_dimension);

        public TorchTensor Sum(long[] dimensions, bool keepDimension = false)
        {
            unsafe
            {
                fixed (long* pdims = dimensions)
                {
                    return new TorchTensor(THSTensor_sum1(handle, (IntPtr)pdims, dimensions.Length, keepDimension));
                }
            }
        }

        // Operators overloading

        public static TorchTensor operator +(TorchTensor left, TorchTensor right)
        {
            return left.Add(right);
        }

        public static TorchTensor operator +(TorchTensor left, Scalar right)
        {
            return left.Add(right);
        }

        public static TorchTensor operator *(TorchTensor left, TorchTensor right)
        {
            return left.Mul(right);
        }

        public static TorchTensor operator *(TorchTensor left, Scalar right)
        {
            return left.Mul(right);
        }

        public static TorchTensor operator -(TorchTensor left, TorchTensor right)
        {
            return left.Sub(right);
        }

        public static TorchTensor operator /(TorchTensor left, TorchTensor right)
        {
            return left.Div(right);
        }

        public static TorchTensor operator /(TorchTensor left, Scalar right)
        {
            return left.Div(right);
        }

        /// <summary>
        ///   Get a string representation of the tensor.
        /// </summary>
        public override string ToString()
        {
            if (Handle == IntPtr.Zero)
            {
                return "";
            }
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
        public static TorchTensor ToTorchTensor<T>(this T[] rawArray, long[] dimensions, bool doCopy = false, bool requiresGrad = false)
        {
            switch (true)
            {
                case bool _ when typeof(T) == typeof(byte):
                    {
                        var result = ByteTensor.From(rawArray as byte[], dimensions, requiresGrad);

                        if (doCopy)
                        {
                            return result.Clone();
                        }
                        return result;
                    }
                case bool _ when typeof(T) == typeof(short):
                    {
                        var result = ShortTensor.From(rawArray as short[], dimensions, requiresGrad);

                        if (doCopy)
                        {
                            return result.Clone();
                        }
                        return result;
                    }
                case bool _ when typeof(T) == typeof(int):
                    {
                        var result = IntTensor.From(rawArray as int[], dimensions, requiresGrad);

                        if (doCopy)
                        {
                            return result.Clone();
                        }
                        return result;
                    }
                case bool _ when typeof(T) == typeof(long):
                    {
                        var result = LongTensor.From(rawArray as long[], dimensions, requiresGrad);

                        if (doCopy)
                        {
                            return result.Clone();
                        }
                        return result;
                    }
                case bool _ when typeof(T) == typeof(double):
                    {
                        var result = DoubleTensor.From(rawArray as double[], dimensions, requiresGrad);

                        if (doCopy)
                        {
                            return result.Clone();
                        }
                        return result;
                    }
                case bool _ when typeof(T) == typeof(float):
                    {
                        var result =  FloatTensor.From(rawArray as float[], dimensions, requiresGrad);

                        if (doCopy)
                        {
                            return result.Clone();
                        }
                        return result;
                    }
                default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        public static TorchTensor ToTorchTensor<T>(this T scalar, bool requiresGrad = false)
        {
            if (requiresGrad && typeof(T) != typeof(float) && typeof(T) != typeof(double))
            {
                throw new ArgumentException(nameof(requiresGrad), "Only floating point types support gradients.");
            }

            switch (true)
            {
                case bool _ when typeof(T) == typeof(byte):
                    {
                        return ByteTensor.From((byte)(object)scalar, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(short):
                    {
                        return ShortTensor.From((short)(object)scalar, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(int):
                    {
                        return IntTensor.From((int)(object)scalar, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(long):
                    {
                        return LongTensor.From((long)(object)scalar, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(double):
                    {
                        return DoubleTensor.From((double)(object)scalar, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(float):
                    {
                        return FloatTensor.From((float)(object)scalar, requiresGrad);
                    }
                default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_cat(IntPtr src, int len, long dim);

        public static TorchTensor Cat(this TorchTensor[] tensors, long dimension)
        {
            if (tensors.Length == 0)
            {
                throw new ArgumentException(nameof(tensors));
            }
            if (tensors.Length == 1)
            {
                return tensors[0];
            }

            var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            return new TorchTensor(THSTensor_cat(tensorsRef, parray.Array.Length, dimension));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_stack(IntPtr src, int len, long dim);

        public static TorchTensor Stack(this TorchTensor[] tensors, long dimension)
        {
            var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            return new TorchTensor(THSTensor_stack(tensorsRef, parray.Array.Length, dimension));
        }
    }
}
