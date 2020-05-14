using System;
using System.Linq;
using System.Runtime.CompilerServices;
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

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
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

        public IntPtr Handle => handle;

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_ndimension(IntPtr handle);

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public long Dimensions => THSTensor_ndimension(handle);

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
            if (NumberOfElements != 1) throw new ArgumentException("Number of elements in the tensor must be 1");

            return Data<T>()[0];
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_item(IntPtr handle);

        public Scalar Item()
        {
            var sptr = THSTensor_item(Handle);
            Torch.AssertNoErrors();
            return new Scalar(sptr);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_fill_(IntPtr handle, IntPtr value);

        public TorchTensor FillInPlace(Scalar value)
        {
            return new TorchTensor(THSTensor_fill_(handle, value.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get1(IntPtr handle, long i1);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set1(IntPtr handle, long i1, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1]
        {
            get => new TorchTensor(THSTensor_get1(handle, i1));
            set
            {
                THSTensor_set1(handle, i1, value.Item().Handle);
                Torch.AssertNoErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get2(IntPtr handle, long i1, long i2);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set2(IntPtr handle, long i1, long i2, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2]
        {
            get => new TorchTensor(THSTensor_get2(handle, i1, i2));
            set
            {
                THSTensor_set2(handle, i1, i2, value.Item().Handle);
                Torch.AssertNoErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get3(IntPtr handle, long i1, long i2, long i3);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set3(IntPtr handle, long i1, long i2, long i3, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3]
        {
            get => new TorchTensor(THSTensor_get3(handle, i1, i2, i3));
            set
            {
                THSTensor_set3(handle, i1, i2, i3, value.Item().Handle);
                Torch.AssertNoErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get4(IntPtr handle, long i1, long i2, long i3, long i4);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set4(IntPtr handle, long i1, long i2, long i3, long i4, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3, long i4]
        {
            get => new TorchTensor(THSTensor_get4(handle, i1, i2, i3, i4));
            set
            {
                THSTensor_set4(handle, i1, i2, i3, i4, value.Item().Handle);
                Torch.AssertNoErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern sbyte THSTensor_type(IntPtr handle);

        public ATenScalarMapping Type => (ATenScalarMapping)THSTensor_type(handle);

        [DllImport("LibTorchSharp")]
        private static extern string THSTensor_deviceType(IntPtr handle);

        public string Device => THSTensor_deviceType(handle);

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_isSparse(IntPtr handle);

        public bool IsSparse => THSTensor_isSparse(handle);

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_isVariable(IntPtr handle);

        public bool IsVariable => THSTensor_isVariable(handle);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_to_type(IntPtr handle, sbyte scalar_type);

        public TorchTensor ToType(ATenScalarMapping type)
        {
            return new TorchTensor(THSTensor_to_type(handle, (sbyte)type));
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_requires_grad(IntPtr handle);

        public bool IsGradRequired => THSTensor_requires_grad(handle);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set_requires_grad(IntPtr handle, bool requires_grad);

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
                for (var i = 0; i < dims.Length; i++)
                    dims[i] = GetTensorDimension(i);

                return dims;
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_strides(IntPtr handle);

        /// <summary>
        ///  Retrieves the strides for the tensor.
        /// </summary>
        public Span<long> Strides
        {
            get
            {
                unsafe
                {
                    return new Span<long>((void*)THSTensor_strides(handle), (int)Dimensions);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_indices(IntPtr handle);

        public TorchTensor Indices => new TorchTensor(THSTensor_indices(handle));

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_values(IntPtr handle);

        public TorchTensor Values => new TorchTensor(THSTensor_values(handle));

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
        private static extern IntPtr THSTensor_transpose_(IntPtr src, long dim1, long dim2);

        public TorchTensor TransposeInPlace(long dimension1, long dimension2)
        {
            return new TorchTensor(THSTensor_transpose_(handle, dimension1, dimension2));
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
        private static extern IntPtr THSTensor_add(IntPtr src, IntPtr trg, IntPtr alpha);

        public TorchTensor Add(TorchTensor target)
        {
            return Add(target, 1);
        }

        public TorchTensor Add(TorchTensor target, Scalar alpha)
        {
            return new TorchTensor(THSTensor_add(handle, target.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addS(IntPtr src, IntPtr trg, IntPtr alpha);

        public TorchTensor Add(Scalar scalar)
        {
            return Add(scalar, 1);
        }

        public TorchTensor Add(Scalar scalar, Scalar alpha)
        {
            return new TorchTensor(THSTensor_addS(handle, scalar.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_add_(IntPtr src, IntPtr trg, IntPtr alpha);

        public TorchTensor AddInPlace(TorchTensor target)
        {
            return AddInPlace(target, 1);
        }

        public TorchTensor AddInPlace(TorchTensor target, Scalar alpha)
        {
            return new TorchTensor(THSTensor_add_(handle, target.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addS_(IntPtr src, IntPtr trg, IntPtr alpha);

        public TorchTensor AddInPlace(Scalar scalar)
        {
            return AddInPlace(scalar, 1);
        }

        public TorchTensor AddInPlace(Scalar scalar, Scalar alpha)
        {
            return new TorchTensor(THSTensor_addS_(handle, scalar.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr
            THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

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
        private static extern IntPtr THSTensor_argmax(IntPtr src);

        public TorchTensor Argmax()
        {
            return new TorchTensor(THSTensor_argmax(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argmaxT(IntPtr src, long dimension, bool keep_dim);

        public TorchTensor Argmax(long dimension, bool keepDim = false)
        {
            return new TorchTensor(THSTensor_argmaxT(handle, dimension, keepDim));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argmin(IntPtr src);

        public TorchTensor Argmin()
        {
            return new TorchTensor(THSTensor_argmin(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argminT(IntPtr src, long dimension, bool keep_dim);

        public TorchTensor Argmin(long dimension, bool keepDim = false)
        {
            return new TorchTensor(THSTensor_argminT(handle, dimension, keepDim));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cos(IntPtr src);

        public TorchTensor Cos()
        {
            return new TorchTensor(THSTensor_cos(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cos_(IntPtr src);

        public TorchTensor CosInPlace()
        {
            return new TorchTensor(THSTensor_cos_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sin(IntPtr src);

        public TorchTensor Sin()
        {
            return new TorchTensor(THSTensor_sin(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sin_(IntPtr src);

        public TorchTensor SinInPlace()
        {
            return new TorchTensor(THSTensor_sin_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tan(IntPtr src);

        public TorchTensor Tan()
        {
            return new TorchTensor(THSTensor_tan(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tan_(IntPtr src);

        public TorchTensor TanInPlace()
        {
            return new TorchTensor(THSTensor_tan_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_asin(IntPtr src);

        public TorchTensor Asin()
        {
            return new TorchTensor(THSTensor_asin(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_asin_(IntPtr src);

        public TorchTensor AsinInPlace()
        {
            return new TorchTensor(THSTensor_asin_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_acos(IntPtr src);

        public TorchTensor Acos()
        {
            return new TorchTensor(THSTensor_acos(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_acos_(IntPtr src);

        public TorchTensor AcosInPlace()
        {
            return new TorchTensor(THSTensor_acos_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_atan(IntPtr src);

        public TorchTensor Atan()
        {
            return new TorchTensor(THSTensor_atan(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_atan_(IntPtr src);

        public TorchTensor AtanInPlace()
        {
            return new TorchTensor(THSTensor_atan_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sinh(IntPtr src);

        public TorchTensor Sinh()
        {
            return new TorchTensor(THSTensor_sinh(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sinh_(IntPtr src);

        public TorchTensor SinhInPlace()
        {
            return new TorchTensor(THSTensor_sinh_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cosh(IntPtr src);

        public TorchTensor Cosh()
        {
            return new TorchTensor(THSTensor_cosh(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cosh_(IntPtr src);

        public TorchTensor CosInPlaceh()
        {
            return new TorchTensor(THSTensor_cosh_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tanh(IntPtr src);

        public TorchTensor Tanh()
        {
            return new TorchTensor(THSTensor_tanh(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tanh_(IntPtr src);

        public TorchTensor TanhInPlace()
        {
            return new TorchTensor(THSTensor_tanh_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_floor(IntPtr src);

        public TorchTensor Floor()
        {
            return new TorchTensor(THSTensor_floor(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_floor_(IntPtr src);

        public TorchTensor FloorInPlace()
        {
            return new TorchTensor(THSTensor_floor_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ceil(IntPtr src);

        public TorchTensor Ceil()
        {
            return new TorchTensor(THSTensor_ceil(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ceil_(IntPtr src);

        public TorchTensor CeilInPlace()
        {
            return new TorchTensor(THSTensor_ceil_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sign(IntPtr src);

        public TorchTensor Sign()
        {
            return new TorchTensor(THSTensor_sign(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sign_(IntPtr src);

        public TorchTensor SignInPlace()
        {
            return new TorchTensor(THSTensor_sign_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_softplus(IntPtr src);

        public TorchTensor Softplus()
        {
            return new TorchTensor(THSTensor_softplus(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_relu(IntPtr src);

        public TorchTensor Relu()
        {
            return new TorchTensor(THSTensor_relu(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_relu_(IntPtr src);

        public TorchTensor ReluInPlace()
        {
            return new TorchTensor(THSTensor_relu_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_round(IntPtr src);

        public TorchTensor Round()
        {
            return new TorchTensor(THSTensor_round(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_round_(IntPtr src);

        public TorchTensor RoundInPlace()
        {
            return new TorchTensor(THSTensor_round_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_abs(IntPtr src);

        public TorchTensor Abs()
        {
            return new TorchTensor(THSTensor_abs(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_abs_(IntPtr src);

        public TorchTensor AbsInPlace()
        {
            return new TorchTensor(THSTensor_abs_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log10(IntPtr src);

        public TorchTensor Log10()
        {
            return new TorchTensor(THSTensor_log10(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log10_(IntPtr src);

        public TorchTensor Log10InPlace()
        {
            return new TorchTensor(THSTensor_log10_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sqrt(IntPtr src);

        public TorchTensor Sqrt()
        {
            return new TorchTensor(THSTensor_sqrt(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sqrt_(IntPtr src);

        public TorchTensor SqrtInPlace()
        {
            return new TorchTensor(THSTensor_sqrt_(handle));
        }

        public static TorchTensor operator -(TorchTensor tensor)
        {
            return tensor.Neg();
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_neg(IntPtr src);

        public TorchTensor Neg()
        {
            return new TorchTensor(THSTensor_neg(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_neg_(IntPtr src);

        public TorchTensor NegInPlace()
        {
            return new TorchTensor(THSTensor_neg_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta,
            float alpha);

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
        private static extern IntPtr THSTensor_clamp(IntPtr input, IntPtr min, IntPtr max);

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
        private static extern IntPtr THSTensor_divS(IntPtr src, IntPtr trg);

        public TorchTensor Div(Scalar target)
        {
            return new TorchTensor(THSTensor_divS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_div_(IntPtr src, IntPtr trg);

        public TorchTensor DivInPlace(TorchTensor target)
        {
            return new TorchTensor(THSTensor_div_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_divS_(IntPtr src, IntPtr trg);

        public TorchTensor DivInPlace(Scalar target)
        {
            return new TorchTensor(THSTensor_divS_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_erf(IntPtr src);

        public TorchTensor Erf()
        {
            return new TorchTensor(THSTensor_erf(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_erf_(IntPtr src);

        public TorchTensor ErfInPlace()
        {
            return new TorchTensor(THSTensor_erf_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eq(IntPtr src, IntPtr trg);

        public TorchTensor Eq(TorchTensor target)
        {
            return new TorchTensor(THSTensor_eq(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eq_(IntPtr src, IntPtr trg);

        public TorchTensor EqInPlace(TorchTensor target)
        {
            return new TorchTensor(THSTensor_eq_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eqS(IntPtr src, IntPtr trg);

        public TorchTensor Eq(Scalar target)
        {
            return new TorchTensor(THSTensor_eqS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eqS_(IntPtr src, IntPtr trg);

        public TorchTensor EqInPlace(Scalar target)
        {
            return new TorchTensor(THSTensor_eqS_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_equal(IntPtr src, IntPtr trg);

        public bool Equal(TorchTensor target)
        {
            return THSTensor_equal(handle, target.Handle);
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_allclose(IntPtr src, IntPtr trg, double rtol, double atol, bool equal_nan);

        public bool AllClose(TorchTensor target, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false)
        {
            return THSTensor_allclose(handle, target.Handle, rtol, atol, equal_nan);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_exp(IntPtr src);

        public TorchTensor Exp()
        {
            return new TorchTensor(THSTensor_exp(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ge(IntPtr src, IntPtr trg);

        public TorchTensor Ge(TorchTensor target)
        {
            return new TorchTensor(THSTensor_ge(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ge_(IntPtr src, IntPtr trg);

        public TorchTensor GeInPlace(TorchTensor target)
        {
            return new TorchTensor(THSTensor_ge_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_geS(IntPtr src, IntPtr trg);

        public TorchTensor Ge(Scalar target)
        {
            return new TorchTensor(THSTensor_geS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_geS_(IntPtr src, IntPtr trg);

        public TorchTensor GeInPlace(Scalar target)
        {
            return new TorchTensor(THSTensor_geS_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gt(IntPtr src, IntPtr trg);

        public TorchTensor Gt(TorchTensor target)
        {
            return new TorchTensor(THSTensor_gt(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gt_(IntPtr src, IntPtr trg);

        public TorchTensor GtInPlace(TorchTensor target)
        {
            return new TorchTensor(THSTensor_gt_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gtS(IntPtr src, IntPtr trg);

        public TorchTensor Gt(Scalar target)
        {
            return new TorchTensor(THSTensor_gtS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gtS_(IntPtr src, IntPtr trg);

        public TorchTensor GtInPlace(Scalar target)
        {
            return new TorchTensor(THSTensor_gtS_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_le(IntPtr src, IntPtr trg);

        public TorchTensor Le(TorchTensor target)
        {
            return new TorchTensor(THSTensor_le(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_le_(IntPtr src, IntPtr trg);

        public TorchTensor LeInPlace(TorchTensor target)
        {
            return new TorchTensor(THSTensor_le_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_leS(IntPtr src, IntPtr trg);

        public TorchTensor Le(Scalar target)
        {
            return new TorchTensor(THSTensor_leS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_leS_(IntPtr src, IntPtr trg);

        public TorchTensor LeInPlace(Scalar target)
        {
            return new TorchTensor(THSTensor_leS_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log(IntPtr src);

        public TorchTensor Log()
        {
            return new TorchTensor(THSTensor_log(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log_(IntPtr src);

        public TorchTensor LogInPlace()
        {
            return new TorchTensor(THSTensor_log_(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lt(IntPtr src, IntPtr trg);

        public TorchTensor Lt(TorchTensor target)
        {
            return new TorchTensor(THSTensor_lt(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lt_(IntPtr src, IntPtr trg);

        public TorchTensor LtInPlace(TorchTensor target)
        {
            return new TorchTensor(THSTensor_lt_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ltS(IntPtr src, IntPtr trg);

        public TorchTensor Lt(Scalar target)
        {
            return new TorchTensor(THSTensor_ltS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ltS_(IntPtr src, IntPtr trg);

        public TorchTensor LtInPlace(Scalar target)
        {
            return new TorchTensor(THSTensor_ltS_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_matmul(IntPtr src, IntPtr target);

        public TorchTensor MatMul(TorchTensor target)
        {
            return new TorchTensor(THSTensor_matmul(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_topk(IntPtr src, AllocatePinnedArray allocator, int k,
            long dimension, bool largest, bool sorted);

        public (TorchTensor values, TorchTensor indexes) TopK(int k, int dimension = -1, bool largest = true, bool sorted = true)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSTensor_topk(handle, pa.CreateArray, k, dimension, largest, sorted);
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_unbind(IntPtr src, AllocatePinnedArray allocator, long dimension);

        public TorchTensor[] Unbind(int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSTensor_unbind(handle, pa.CreateArray, dimension);
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }


        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_split_with_sizes(IntPtr src, AllocatePinnedArray allocator, IntPtr psizes, int length, long dimension);

        public TorchTensor[] SplitWithSizes(long[] sizes, int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                unsafe
                {
                    fixed (long* psizes = sizes)
                    {
                        THSTensor_split_with_sizes(handle, pa.CreateArray, (IntPtr)psizes, sizes.Length, dimension);
                    }
                }
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_max(IntPtr src, AllocatePinnedArray allocator, long dimension,
            bool keep_dim);

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
        private static extern IntPtr THSTensor_mulS(IntPtr src, IntPtr scalar);

        public TorchTensor Mul(Scalar scalar)
        {
            return new TorchTensor(THSTensor_mulS(handle, scalar.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mul_(IntPtr src, IntPtr target);

        public TorchTensor MulInPlace(TorchTensor target)
        {
            return new TorchTensor(THSTensor_mul_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mulS_(IntPtr src, IntPtr target);

        public TorchTensor MulInPlace(Scalar target)
        {
            return new TorchTensor(THSTensor_mulS_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ne(IntPtr src, IntPtr trg);

        public TorchTensor Ne(TorchTensor target)
        {
            return new TorchTensor(THSTensor_ne(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ne_(IntPtr src, IntPtr trg);

        public TorchTensor NeInPlace(TorchTensor target)
        {
            return new TorchTensor(THSTensor_ne_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_neS(IntPtr src, IntPtr trg);

        public TorchTensor Ne(Scalar target)
        {
            return new TorchTensor(THSTensor_neS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_neS_(IntPtr src, IntPtr trg);

        public TorchTensor NeInPlace(Scalar target)
        {
            return new TorchTensor(THSTensor_neS_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_norm(IntPtr src, int dimension, bool keep_dimension);

        public TorchTensor Norm(int dimension, bool KeepDimension = false)
        {
            return new TorchTensor(THSTensor_norm(handle, dimension, KeepDimension));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_powS(IntPtr src, IntPtr scalar);

        public TorchTensor Pow(Scalar scalar)
        {
            return new TorchTensor(THSTensor_powS(handle, scalar.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_pow(IntPtr src, IntPtr exponent);

        public TorchTensor Pow(TorchTensor exponent)
        {
            return new TorchTensor(THSTensor_pow(handle, exponent.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_powS_(IntPtr src, IntPtr scalar);

        public TorchTensor PowInPlace(Scalar scalar)
        {
            return new TorchTensor(THSTensor_powS_(handle, scalar.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_pow_(IntPtr src, IntPtr exponent);

        public TorchTensor PowInPlace(TorchTensor exponent)
        {
            return new TorchTensor(THSTensor_pow_(handle, exponent.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainder(IntPtr src, IntPtr trg);

        public TorchTensor Remainder(TorchTensor target)
        {
            return new TorchTensor(THSTensor_remainder(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainder_(IntPtr src, IntPtr trg);

        public TorchTensor RemainderInPlace(TorchTensor target)
        {
            return new TorchTensor(THSTensor_remainder_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainderS(IntPtr src, IntPtr scalar);

        public TorchTensor Remainder(Scalar scalar)
        {
            return new TorchTensor(THSTensor_remainderS(handle, scalar.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainderS_(IntPtr src, IntPtr scalar);

        public TorchTensor RemainderInPlace(Scalar scalar)
        {
            return new TorchTensor(THSTensor_remainderS_(handle, scalar.Handle));
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
        private static extern IntPtr THSTensor_subS(IntPtr src, IntPtr trg);

        public TorchTensor Sub(Scalar target)
        {
            return new TorchTensor(THSTensor_subS(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sub_(IntPtr src, IntPtr trg);

        public TorchTensor SubInPlace(TorchTensor target)
        {
            return new TorchTensor(THSTensor_sub_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_subS_(IntPtr src, IntPtr trg);

        public TorchTensor SubInPlace(Scalar target)
        {
            return new TorchTensor(THSTensor_subS_(handle, target.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sum(IntPtr src);

        /// <summary>
        /// Returns the sum of all elements in the :attr:`input` tensor.
        /// </summary>
        public TorchTensor Sum()
        {
            return new TorchTensor(THSTensor_sum(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sum1(IntPtr src, IntPtr dimensions, int length, bool keep_dimension);

        /// <summary>
        ///  Returns the sum of each row of the input tensor in the given dimensions.
        /// </summary>
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


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_expand(IntPtr src, IntPtr psizes, int length, bool isImplicit);

        /// <summary>
        ///  Returns a new view of the tensor with singleton dimensions expanded to a larger size.
        /// </summary>
        public TorchTensor Expand(long[] sizes, bool isImplicit = false)
        {
            unsafe
            {
                fixed (long* psizes = sizes)
                {
                    return new TorchTensor(THSTensor_expand(handle, (IntPtr)psizes, sizes.Length, isImplicit));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_flip(IntPtr src, IntPtr psizes, int length);

        /// <summary>
        ///  Reverse the order of a n-D tensor along given axis in dims.
        /// </summary>
        public TorchTensor Flip(long[] sizes)
        {
            unsafe
            {
                fixed (long* psizes = sizes)
                {
                    return new TorchTensor(THSTensor_flip(handle, (IntPtr)psizes, sizes.Length));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_narrow(IntPtr src, long dimension, long start, long length);

        /// <summary>
        ///  Returns a new tensor that is a narrowed version of the input along one dimension. The
        /// dimension is input from start to start + length. The
        /// returned tensor and the input tensor share the same underlying storage.
        /// </summary>
        public TorchTensor Narrow(long dimension, long start, long length)
        {
            return new TorchTensor(THSTensor_narrow(handle, dimension, start, length));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_slice(IntPtr src, long dimension, long start, long length, long step);

        /// <summary>
        ///  Returns a new tensor that is a sliced version of the input along one dimension. The
        /// dimension is input from start to finish-1. The
        /// returned tensor and the input tensor share the same underlying storage.
        /// </summary>
        public TorchTensor Slice(long dimension, long start, long finish, long step)
        {
            return new TorchTensor(THSTensor_slice(handle, dimension, start, finish, step));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_unsqueeze(IntPtr src, long dimension);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor Conv1D(TorchTensor weight, TorchTensor? bias, 
            long? stride = null,
            long? padding = null,
            long? dilation = null,
            long groups = 1)
        {
            var strides = new long[] { stride ?? 1 };
            var padding = new long[] { padding ?? 0 }; 
            var dilation = new long[] { dilation ?? 1 }; 
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var biasHandle = (bias.HasValue ? bias.Value.Handle : IntPtr.Zero);
                    var res =
                        THSTensor_conv1d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    return new TorchTensor(res);
                }
            }
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor Conv2D(TorchTensor weight, TorchTensor? bias,
            long[] strides = null,
            long[] padding = null,
            long[] dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            padding = (padding == null) ? new long[] { 0 } : padding;
            dilation = (dilation == null) ? new long[] { 1 } : dilation;
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var biasHandle = (bias.HasValue ? bias.Value.Handle : IntPtr.Zero);
                    var res =
                        THSTensor_conv2d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor Conv3D(TorchTensor weight, TorchTensor? bias,
            long[] strides = null,
            long[] padding = null,
            long[] dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            padding = (padding == null) ? new long[] { 0 } : padding;
            dilation = (dilation == null) ? new long[] { 1 } : dilation;
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var biasHandle = (bias.HasValue ? bias.Value.Handle : IntPtr.Zero);
                    var res =
                        THSTensor_conv3d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length, 
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv_transpose1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor ConvTranspose1D(TorchTensor weight, TorchTensor? bias,
            long? stride = null,
            long? padding = null,
            long? outputPadding = null,
            long? dilation = null,
            long groups = 1)
        {
            var strides = new long[] { stride ?? 1 };
            var padding = new long[] { padding ?? 0 };
            var outputPadding = new long[] { outputPadding ?? 0 };
            var dilation = new long[] { dilation ?? 1 };
            //var biasHandle = (bias.HasValue ? bias.Value.Handle : IntPtr.Zero);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, poutputPadding = outputPadding, pdilation = dilation)
                {
                    var biasHandle = (bias.HasValue ? bias.Value.Handle : IntPtr.Zero);
                    var res =
                        THSTensor_conv_transpose1d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)poutputPadding, outputPadding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv_transpose2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor ConvTranspose2D(TorchTensor weight, TorchTensor? bias,
            long[] strides = null,
            long[] padding = null,
            long[] outputPadding = null,
            long[] dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1, 1 } : strides;
            padding = (padding == null) ? new long[] { 0, 0 } : padding;
            outputPadding = (outputPadding == null) ? new long[] { 0, 0 } : outputPadding;
            dilation = (dilation == null) ? new long[] { 1, 1 } : dilation;
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, poutputPadding = outputPadding, pdilation = dilation)
                {
                    var biasHandle = (bias.HasValue ? bias.Value.Handle : IntPtr.Zero);
                    var res =
                        THSTensor_conv_transpose2d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)poutputPadding, outputPadding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv_transpose3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor ConvTranspose3D(TorchTensor weight, TorchTensor? bias,
            long[] strides = null,
            long[] padding = null,
            long[] outputPadding = null,
            long[] dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1, 1, 1 } : strides;
            padding = (padding == null) ? new long[] { 0, 0, 0 } : padding;
            outputPadding = (outputPadding == null) ? new long[] { 0, 0, 0 } : outputPadding;
            dilation = (dilation == null) ? new long[] { 1, 1, 1 } : dilation;
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, poutputPadding = outputPadding, pdilation = dilation)
                {
                    var biasHandle = (bias.HasValue ? bias.Value.Handle : IntPtr.Zero);
                    var res =
                        THSTensor_conv_transpose3d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)poutputPadding, outputPadding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    return new TorchTensor(res);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_maxpool1d(IntPtr input,
                IntPtr kernelSizes, int kernelSizesLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor MaxPool1D(long kernelSize, long? stride = null,
            long? padding = null, long? dilation = null, bool ceil_mode = false)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var padding = new long[] { padding ?? 0 };
            var dilation = new long[] { dilation ?? 1 };
            unsafe
            {
                fixed (long* pkernelSizes = kernelSizes, pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var res =
                        THSTensor_maxpool1d(handle, 
                            (IntPtr)pkernelSizes, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_maxpool2d(IntPtr input,
                IntPtr kernelSizes, int kernelSizesLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor MaxPool2D(long[] kernelSizes, long[] strides = null,
            long[] padding = null, long[] dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSizes.Select(x => 1L).ToArray();
            padding = padding ?? kernelSizes.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSizes.Select(x => 1L).ToArray();
            unsafe
            {
                fixed (long* pkernelSizes = kernelSizes, pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var res =
                        THSTensor_maxpool2d(handle,
                            (IntPtr)pkernelSizes, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_maxpool3d(IntPtr input,
                IntPtr kernelSizes, int kernelSizesLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor MaxPool3D(long[] kernelSizes, long[] strides = null,
            long[] padding = null, long[] dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSizes.Select(x => 1L).ToArray();
            padding = padding ?? kernelSizes.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSizes.Select(x => 1L).ToArray();
            unsafe
            {
                fixed (long* pkernelSizes = kernelSizes, pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var res =
                        THSTensor_maxpool3d(handle,
                            (IntPtr)pkernelSizes, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                    return new TorchTensor(res);
                }
            }
        }

        /// <summary>
        ///  Returns a new tensor with a dimension of size one inserted at the specified position.
        ///  The returned tensor shares the same underlying data with this tensor.
        /// </summary>

        public TorchTensor Unsqueeze(long dimension)
        {
            return new TorchTensor(THSTensor_unsqueeze(handle, dimension));
        }

        // Operators overloading

        public static TorchTensor operator ==(TorchTensor left, TorchTensor right)
        {
            return left.Eq(right);
        }

        public static TorchTensor operator ==(TorchTensor left, Scalar right)
        {
            return left.Eq(right);
        }

        public static TorchTensor operator ==(Scalar left, TorchTensor right)
        {
            return right.Eq(left);
        }

        public static TorchTensor operator !=(TorchTensor left, TorchTensor right)
        {
            return left.Ne(right);
        }

        public static TorchTensor operator !=(TorchTensor left, Scalar right)
        {
            return left.Ne(right);
        }

        public static TorchTensor operator !=(Scalar left, TorchTensor right)
        {
            return right.Ne(left);
        }

        public static TorchTensor operator +(TorchTensor left, TorchTensor right)
        {
            return left.Add(right);
        }

        public static TorchTensor operator +(TorchTensor left, Scalar right)
        {
            return left.Add(right);
        }

        public static TorchTensor operator +(Scalar left, TorchTensor right)
        {
            return right.Add(left);
        }

        public static TorchTensor operator *(TorchTensor left, TorchTensor right)
        {
            return left.Mul(right);
        }

        public static TorchTensor operator *(TorchTensor left, Scalar right)
        {
            return left.Mul(right);
        }

        public static TorchTensor operator *(Scalar left, TorchTensor right)
        {
            return right.Mul(left);
        }

        public static TorchTensor operator -(TorchTensor left, TorchTensor right)
        {
            return left.Sub(right);
        }

        public static TorchTensor operator -(TorchTensor left, Scalar right)
        {
            return new TorchTensor(THSTensor_subS(left.Handle, right.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_subS2(IntPtr src, IntPtr trg);

        public static TorchTensor operator -(Scalar left, TorchTensor right)
        {
            return new TorchTensor(THSTensor_subS2(left.Handle, right.Handle));
        }

        public static TorchTensor operator /(TorchTensor left, TorchTensor right)
        {
            return left.Div(right);
        }

        public static TorchTensor operator /(TorchTensor left, Scalar right)
        {
            return left.Div(right);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_divS2(IntPtr left, IntPtr right);

        public static TorchTensor operator /(Scalar left, TorchTensor right)
        {
            return new TorchTensor(THSTensor_divS2(left.Handle, right.Handle));
        }

        public static TorchTensor operator %(TorchTensor left, TorchTensor right)
        {
            return left.Remainder(right);
        }

        public static TorchTensor operator %(TorchTensor left, Scalar right)
        {
            return left.Remainder(right);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainderS2(IntPtr left, IntPtr right);

        public static TorchTensor operator %(Scalar left, TorchTensor right)
        {
            return new TorchTensor(THSTensor_remainderS2(left.Handle, right.Handle));
        }

        public static TorchTensor operator <(TorchTensor left, TorchTensor right)
        {
            return left.Lt(right);
        }

        public static TorchTensor operator <(TorchTensor left, Scalar right)
        {
            return left.Lt(right);
        }

        public static TorchTensor operator <(Scalar left, TorchTensor right)
        {
            return right.Gt(left);
        }

        public static TorchTensor operator <=(TorchTensor left, TorchTensor right)
        {
            return left.Le(right);
        }

        public static TorchTensor operator <=(TorchTensor left, Scalar right)
        {
            return left.Le(right);
        }

        public static TorchTensor operator <=(Scalar left, TorchTensor right)
        {
            return right.Ge(left);
        }

        public static TorchTensor operator >(TorchTensor left, TorchTensor right)
        {
            return left.Gt(right);
        }

        public static TorchTensor operator >(TorchTensor left, Scalar right)
        {
            return left.Gt(right);
        }

        public static TorchTensor operator >(Scalar left, TorchTensor right)
        {
            return right.Lt(left);
        }

        public static TorchTensor operator >=(TorchTensor left, TorchTensor right)
        {
            return left.Ge(right);
        }

        public static TorchTensor operator >=(TorchTensor left, Scalar right)
        {
            return left.Ge(right);
        }

        public static TorchTensor operator >=(Scalar left, TorchTensor right)
        {
            return right.Le(left);
        }

        /// <summary>
        ///   Get a string representation of the tensor.
        /// </summary>
        public override string ToString()
        {
            if (Handle == IntPtr.Zero) return "";

            var n = Dimensions;
            if (n == 0)
                return "[]";

            var sb = new StringBuilder("[");
            for (var i = 0; i < n; i++)
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
                throw new InvalidOperationException("CUDA non available in the current machine.");
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
            var array = doCopy ? (T[])rawArray.Clone() : rawArray;

            switch (true)
            {
                case bool _ when typeof(T) == typeof(byte):
                    {
                        return ByteTensor.From(array as byte[], dimensions, requiresGrad); ;
                    }
                case bool _ when typeof(T) == typeof(short):
                    {
                        return ShortTensor.From(array as short[], dimensions, requiresGrad); ;
                    }
                case bool _ when typeof(T) == typeof(int):
                    {
                        return IntTensor.From(array as int[], dimensions, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(long):
                    {
                        return LongTensor.From(array as long[], dimensions, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(double):
                    {
                        return DoubleTensor.From(array as double[], dimensions, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(float):
                    {
                        return FloatTensor.From(array as float[], dimensions, requiresGrad);
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