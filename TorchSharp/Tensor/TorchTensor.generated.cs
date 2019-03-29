using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp.Tensor {

    /// <summary>
    ///   Tensor of type Byte.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public struct ByteTensor : ITorchTensor<byte>
    {
        internal IntPtr handle;

        internal ByteTensor(IntPtr handle)
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
        extern static IntPtr THSTensor_newByteScalar(byte scalar);

        public static ByteTensor From(byte scalar)
        {
            return new ByteTensor(THSTensor_newByteScalar(scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static ByteTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new ByteTensor(THSTensor_new(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Byte));
        }

        public static ByteTensor From(byte[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (byte* parray = rawArray)
                {
                    return ByteTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static AtenSharp.ByteTensor.HType THSTensor_unsafeGetTensorImpl(IntPtr handle);

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.ByteTensor (THSTensor_unsafeGetTensorImpl (handle));
                return atenTensor.Dimensions;
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
        public Span<byte> Data
        {
            get
            {               
                if (NumberOfElements > int.MaxValue)
                {
                    throw new ArgumentException("Span only supports up to int.MaxValue elements.");
                }
                unsafe
                {
                    return new System.Span<byte>((void*)THSTensor_data(handle), (int)NumberOfElements);
                }
            }
        }

        public byte Item
        {
            get
            {
                if (NumberOfElements != 1)
                {
                    throw new ArgumentException($"Number of elements in the tensor must be 1");
                }
                return Data[0];
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
        extern static IntPtr THSTensor_cpu(IntPtr handle);

        public ITorchTensor<byte> Cpu()
        {
            return new ByteTensor(THSTensor_cpu(handle));
        }

         [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_cuda(IntPtr handle);

        public ITorchTensor<byte> Cuda()
        {
            if (!Torch.IsCudaAvailable())
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }

            return new ByteTensor(THSTensor_cuda(handle));
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.ByteTensor (THSTensor_unsafeGetTensorImpl (handle));
            return atenTensor.GetTensorDimension (dim);
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

        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            var atenTensor = new AtenSharp.ByteTensor(THSTensor_unsafeGetTensorImpl(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor<byte> Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            ByteTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new ByteTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Byte, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<byte> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            ByteTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new ByteTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Byte, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<byte> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            ByteTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new ByteTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Byte, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_backward(IntPtr handle);

        public void Backward()
        {
            THSTensor_backward(handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_grad(IntPtr handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THSTensor_grad(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_reshape(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<byte> Reshape(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new ByteTensor (THSTensor_reshape (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_view(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<byte> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new ByteTensor (THSTensor_view (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_add(IntPtr src, int scalar, IntPtr trg);

        public ITorchTensor<byte>  Add(ITorchTensor<byte> target, int scalar)
        {
            return new ByteTensor(THSTensor_add(handle, scalar, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_add_(IntPtr src, int scalar, IntPtr trg);

        public void AddInPlace(ITorchTensor<byte> target, int scalar)
        {
            THSTensor_add_(handle, scalar, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public ITorchTensor<byte> Addbmm(ITorchTensor<byte> batch1, ITorchTensor<byte> batch2, float beta, float alpha)
        {
            return new ByteTensor(THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_argmax(IntPtr src, long dimension, bool keep_dim);

        public ITorchTensor<byte> Argmax(long dimension, bool keepDim = false)
        {
            return new ByteTensor(THSTensor_argmax(handle, dimension, keepDim));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta, float alpha);

        public ITorchTensor<byte> Baddbmm(ITorchTensor<byte> batch2, ITorchTensor<byte> mat, float beta, float alpha)
        {
            return new ByteTensor(THSTensor_addbmm(handle, batch2.Handle, mat.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_eq(IntPtr src, IntPtr trg);

        public ITorchTensor<U> Eq<U>(ITorchTensor<U> target)
        {
            return THSTensor_eq(handle, target.Handle).ToTorchTensor<U>();
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_exp(IntPtr src);

        public ITorchTensor<byte> Exp()
        {
            return new ByteTensor(THSTensor_exp(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_matMul(IntPtr src, IntPtr target);

        public ITorchTensor<byte> MatMul(ITorchTensor<byte> target)
        {
            return new ByteTensor(THSTensor_matMul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mul(IntPtr src, IntPtr target);

        public ITorchTensor<byte> Mul(ITorchTensor<byte> target)
        {
            return new ByteTensor(THSTensor_mul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_mul_(IntPtr src, IntPtr target);

        public void MulInPlace(ITorchTensor<byte> target)
        {
            THSTensor_mul_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mulS(IntPtr src, byte scalar);

        public ITorchTensor<byte> Mul(byte scalar)
        {
            return new ByteTensor(THSTensor_mulS(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_pow(IntPtr src, float scalar);

        public ITorchTensor<byte> Pow(float scalar)
        {
            return new ByteTensor(THSTensor_pow(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sigmoid(IntPtr src);

        public ITorchTensor<byte> Sigmoid()
        {
            return new ByteTensor(THSTensor_sigmoid(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sub(IntPtr src, IntPtr trg);

        public ITorchTensor<byte> Sub(ITorchTensor<byte> target)
        {
            return new ByteTensor(THSTensor_sub(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_sub_(IntPtr src, IntPtr trg);

        public void SubInPlace(ITorchTensor<byte> target)
        {
            THSTensor_sub_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sum(IntPtr src);

        public ITorchTensor<byte> Sum()
        {
            return new ByteTensor(THSTensor_sum(handle));
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

        private static void CheckForCUDA(string device)
        {
            if (!Torch.IsCudaAvailable() && device.ToLower().Contains("cuda"))
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }
        }
    }
    /// <summary>
    ///   Tensor of type Short.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public struct ShortTensor : ITorchTensor<short>
    {
        internal IntPtr handle;

        internal ShortTensor(IntPtr handle)
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
        extern static IntPtr THSTensor_newShortScalar(short scalar);

        public static ShortTensor From(short scalar)
        {
            return new ShortTensor(THSTensor_newShortScalar(scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static ShortTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new ShortTensor(THSTensor_new(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Short));
        }

        public static ShortTensor From(short[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (short* parray = rawArray)
                {
                    return ShortTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static AtenSharp.ShortTensor.HType THSTensor_unsafeGetTensorImpl(IntPtr handle);

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.ShortTensor (THSTensor_unsafeGetTensorImpl (handle));
                return atenTensor.Dimensions;
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
        public Span<short> Data
        {
            get
            {               
                if (NumberOfElements > int.MaxValue)
                {
                    throw new ArgumentException("Span only supports up to int.MaxValue elements.");
                }
                unsafe
                {
                    return new System.Span<short>((void*)THSTensor_data(handle), (int)NumberOfElements);
                }
            }
        }

        public short Item
        {
            get
            {
                if (NumberOfElements != 1)
                {
                    throw new ArgumentException($"Number of elements in the tensor must be 1");
                }
                return Data[0];
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
        extern static IntPtr THSTensor_cpu(IntPtr handle);

        public ITorchTensor<short> Cpu()
        {
            return new ShortTensor(THSTensor_cpu(handle));
        }

         [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_cuda(IntPtr handle);

        public ITorchTensor<short> Cuda()
        {
            if (!Torch.IsCudaAvailable())
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }

            return new ShortTensor(THSTensor_cuda(handle));
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.ShortTensor (THSTensor_unsafeGetTensorImpl (handle));
            return atenTensor.GetTensorDimension (dim);
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

        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            var atenTensor = new AtenSharp.ShortTensor(THSTensor_unsafeGetTensorImpl(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor<short> Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            ShortTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new ShortTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Short, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<short> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            ShortTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new ShortTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Short, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<short> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            ShortTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new ShortTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Short, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_backward(IntPtr handle);

        public void Backward()
        {
            THSTensor_backward(handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_grad(IntPtr handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THSTensor_grad(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_reshape(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<short> Reshape(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new ShortTensor (THSTensor_reshape (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_view(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<short> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new ShortTensor (THSTensor_view (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_add(IntPtr src, int scalar, IntPtr trg);

        public ITorchTensor<short>  Add(ITorchTensor<short> target, int scalar)
        {
            return new ShortTensor(THSTensor_add(handle, scalar, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_add_(IntPtr src, int scalar, IntPtr trg);

        public void AddInPlace(ITorchTensor<short> target, int scalar)
        {
            THSTensor_add_(handle, scalar, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public ITorchTensor<short> Addbmm(ITorchTensor<short> batch1, ITorchTensor<short> batch2, float beta, float alpha)
        {
            return new ShortTensor(THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_argmax(IntPtr src, long dimension, bool keep_dim);

        public ITorchTensor<short> Argmax(long dimension, bool keepDim = false)
        {
            return new ShortTensor(THSTensor_argmax(handle, dimension, keepDim));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta, float alpha);

        public ITorchTensor<short> Baddbmm(ITorchTensor<short> batch2, ITorchTensor<short> mat, float beta, float alpha)
        {
            return new ShortTensor(THSTensor_addbmm(handle, batch2.Handle, mat.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_eq(IntPtr src, IntPtr trg);

        public ITorchTensor<U> Eq<U>(ITorchTensor<U> target)
        {
            return THSTensor_eq(handle, target.Handle).ToTorchTensor<U>();
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_exp(IntPtr src);

        public ITorchTensor<short> Exp()
        {
            return new ShortTensor(THSTensor_exp(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_matMul(IntPtr src, IntPtr target);

        public ITorchTensor<short> MatMul(ITorchTensor<short> target)
        {
            return new ShortTensor(THSTensor_matMul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mul(IntPtr src, IntPtr target);

        public ITorchTensor<short> Mul(ITorchTensor<short> target)
        {
            return new ShortTensor(THSTensor_mul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_mul_(IntPtr src, IntPtr target);

        public void MulInPlace(ITorchTensor<short> target)
        {
            THSTensor_mul_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mulS(IntPtr src, short scalar);

        public ITorchTensor<short> Mul(short scalar)
        {
            return new ShortTensor(THSTensor_mulS(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_pow(IntPtr src, float scalar);

        public ITorchTensor<short> Pow(float scalar)
        {
            return new ShortTensor(THSTensor_pow(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sigmoid(IntPtr src);

        public ITorchTensor<short> Sigmoid()
        {
            return new ShortTensor(THSTensor_sigmoid(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sub(IntPtr src, IntPtr trg);

        public ITorchTensor<short> Sub(ITorchTensor<short> target)
        {
            return new ShortTensor(THSTensor_sub(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_sub_(IntPtr src, IntPtr trg);

        public void SubInPlace(ITorchTensor<short> target)
        {
            THSTensor_sub_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sum(IntPtr src);

        public ITorchTensor<short> Sum()
        {
            return new ShortTensor(THSTensor_sum(handle));
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

        private static void CheckForCUDA(string device)
        {
            if (!Torch.IsCudaAvailable() && device.ToLower().Contains("cuda"))
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }
        }
    }
    /// <summary>
    ///   Tensor of type Int.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public struct IntTensor : ITorchTensor<int>
    {
        internal IntPtr handle;

        internal IntTensor(IntPtr handle)
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
        extern static IntPtr THSTensor_newIntScalar(int scalar);

        public static IntTensor From(int scalar)
        {
            return new IntTensor(THSTensor_newIntScalar(scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static IntTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new IntTensor(THSTensor_new(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Int));
        }

        public static IntTensor From(int[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (int* parray = rawArray)
                {
                    return IntTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static AtenSharp.IntTensor.HType THSTensor_unsafeGetTensorImpl(IntPtr handle);

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.IntTensor (THSTensor_unsafeGetTensorImpl (handle));
                return atenTensor.Dimensions;
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
        public Span<int> Data
        {
            get
            {               
                if (NumberOfElements > int.MaxValue)
                {
                    throw new ArgumentException("Span only supports up to int.MaxValue elements.");
                }
                unsafe
                {
                    return new System.Span<int>((void*)THSTensor_data(handle), (int)NumberOfElements);
                }
            }
        }

        public int Item
        {
            get
            {
                if (NumberOfElements != 1)
                {
                    throw new ArgumentException($"Number of elements in the tensor must be 1");
                }
                return Data[0];
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
        extern static IntPtr THSTensor_cpu(IntPtr handle);

        public ITorchTensor<int> Cpu()
        {
            return new IntTensor(THSTensor_cpu(handle));
        }

         [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_cuda(IntPtr handle);

        public ITorchTensor<int> Cuda()
        {
            if (!Torch.IsCudaAvailable())
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }

            return new IntTensor(THSTensor_cuda(handle));
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.IntTensor (THSTensor_unsafeGetTensorImpl (handle));
            return atenTensor.GetTensorDimension (dim);
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

        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            var atenTensor = new AtenSharp.IntTensor(THSTensor_unsafeGetTensorImpl(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor<int> Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            IntTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new IntTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Int, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<int> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            IntTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new IntTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Int, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<int> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            IntTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new IntTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Int, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_backward(IntPtr handle);

        public void Backward()
        {
            THSTensor_backward(handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_grad(IntPtr handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THSTensor_grad(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_reshape(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<int> Reshape(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new IntTensor (THSTensor_reshape (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_view(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<int> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new IntTensor (THSTensor_view (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_add(IntPtr src, int scalar, IntPtr trg);

        public ITorchTensor<int>  Add(ITorchTensor<int> target, int scalar)
        {
            return new IntTensor(THSTensor_add(handle, scalar, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_add_(IntPtr src, int scalar, IntPtr trg);

        public void AddInPlace(ITorchTensor<int> target, int scalar)
        {
            THSTensor_add_(handle, scalar, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public ITorchTensor<int> Addbmm(ITorchTensor<int> batch1, ITorchTensor<int> batch2, float beta, float alpha)
        {
            return new IntTensor(THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_argmax(IntPtr src, long dimension, bool keep_dim);

        public ITorchTensor<int> Argmax(long dimension, bool keepDim = false)
        {
            return new IntTensor(THSTensor_argmax(handle, dimension, keepDim));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta, float alpha);

        public ITorchTensor<int> Baddbmm(ITorchTensor<int> batch2, ITorchTensor<int> mat, float beta, float alpha)
        {
            return new IntTensor(THSTensor_addbmm(handle, batch2.Handle, mat.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_eq(IntPtr src, IntPtr trg);

        public ITorchTensor<U> Eq<U>(ITorchTensor<U> target)
        {
            return THSTensor_eq(handle, target.Handle).ToTorchTensor<U>();
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_exp(IntPtr src);

        public ITorchTensor<int> Exp()
        {
            return new IntTensor(THSTensor_exp(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_matMul(IntPtr src, IntPtr target);

        public ITorchTensor<int> MatMul(ITorchTensor<int> target)
        {
            return new IntTensor(THSTensor_matMul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mul(IntPtr src, IntPtr target);

        public ITorchTensor<int> Mul(ITorchTensor<int> target)
        {
            return new IntTensor(THSTensor_mul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_mul_(IntPtr src, IntPtr target);

        public void MulInPlace(ITorchTensor<int> target)
        {
            THSTensor_mul_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mulS(IntPtr src, int scalar);

        public ITorchTensor<int> Mul(int scalar)
        {
            return new IntTensor(THSTensor_mulS(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_pow(IntPtr src, float scalar);

        public ITorchTensor<int> Pow(float scalar)
        {
            return new IntTensor(THSTensor_pow(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sigmoid(IntPtr src);

        public ITorchTensor<int> Sigmoid()
        {
            return new IntTensor(THSTensor_sigmoid(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sub(IntPtr src, IntPtr trg);

        public ITorchTensor<int> Sub(ITorchTensor<int> target)
        {
            return new IntTensor(THSTensor_sub(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_sub_(IntPtr src, IntPtr trg);

        public void SubInPlace(ITorchTensor<int> target)
        {
            THSTensor_sub_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sum(IntPtr src);

        public ITorchTensor<int> Sum()
        {
            return new IntTensor(THSTensor_sum(handle));
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

        private static void CheckForCUDA(string device)
        {
            if (!Torch.IsCudaAvailable() && device.ToLower().Contains("cuda"))
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }
        }
    }
    /// <summary>
    ///   Tensor of type Long.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public struct LongTensor : ITorchTensor<long>
    {
        internal IntPtr handle;

        internal LongTensor(IntPtr handle)
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
        extern static IntPtr THSTensor_newLongScalar(long scalar);

        public static LongTensor From(long scalar)
        {
            return new LongTensor(THSTensor_newLongScalar(scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static LongTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new LongTensor(THSTensor_new(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Long));
        }

        public static LongTensor From(long[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (long* parray = rawArray)
                {
                    return LongTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static AtenSharp.LongTensor.HType THSTensor_unsafeGetTensorImpl(IntPtr handle);

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.LongTensor (THSTensor_unsafeGetTensorImpl (handle));
                return atenTensor.Dimensions;
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
        public Span<long> Data
        {
            get
            {               
                if (NumberOfElements > int.MaxValue)
                {
                    throw new ArgumentException("Span only supports up to int.MaxValue elements.");
                }
                unsafe
                {
                    return new System.Span<long>((void*)THSTensor_data(handle), (int)NumberOfElements);
                }
            }
        }

        public long Item
        {
            get
            {
                if (NumberOfElements != 1)
                {
                    throw new ArgumentException($"Number of elements in the tensor must be 1");
                }
                return Data[0];
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
        extern static IntPtr THSTensor_cpu(IntPtr handle);

        public ITorchTensor<long> Cpu()
        {
            return new LongTensor(THSTensor_cpu(handle));
        }

         [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_cuda(IntPtr handle);

        public ITorchTensor<long> Cuda()
        {
            if (!Torch.IsCudaAvailable())
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }

            return new LongTensor(THSTensor_cuda(handle));
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.LongTensor (THSTensor_unsafeGetTensorImpl (handle));
            return atenTensor.GetTensorDimension (dim);
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

        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            var atenTensor = new AtenSharp.LongTensor(THSTensor_unsafeGetTensorImpl(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor<long> Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            LongTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new LongTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Long, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<long> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            LongTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new LongTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Long, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<long> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            LongTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new LongTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Long, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_backward(IntPtr handle);

        public void Backward()
        {
            THSTensor_backward(handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_grad(IntPtr handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THSTensor_grad(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_reshape(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<long> Reshape(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new LongTensor (THSTensor_reshape (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_view(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<long> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new LongTensor (THSTensor_view (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_add(IntPtr src, int scalar, IntPtr trg);

        public ITorchTensor<long>  Add(ITorchTensor<long> target, int scalar)
        {
            return new LongTensor(THSTensor_add(handle, scalar, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_add_(IntPtr src, int scalar, IntPtr trg);

        public void AddInPlace(ITorchTensor<long> target, int scalar)
        {
            THSTensor_add_(handle, scalar, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public ITorchTensor<long> Addbmm(ITorchTensor<long> batch1, ITorchTensor<long> batch2, float beta, float alpha)
        {
            return new LongTensor(THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_argmax(IntPtr src, long dimension, bool keep_dim);

        public ITorchTensor<long> Argmax(long dimension, bool keepDim = false)
        {
            return new LongTensor(THSTensor_argmax(handle, dimension, keepDim));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta, float alpha);

        public ITorchTensor<long> Baddbmm(ITorchTensor<long> batch2, ITorchTensor<long> mat, float beta, float alpha)
        {
            return new LongTensor(THSTensor_addbmm(handle, batch2.Handle, mat.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_eq(IntPtr src, IntPtr trg);

        public ITorchTensor<U> Eq<U>(ITorchTensor<U> target)
        {
            return THSTensor_eq(handle, target.Handle).ToTorchTensor<U>();
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_exp(IntPtr src);

        public ITorchTensor<long> Exp()
        {
            return new LongTensor(THSTensor_exp(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_matMul(IntPtr src, IntPtr target);

        public ITorchTensor<long> MatMul(ITorchTensor<long> target)
        {
            return new LongTensor(THSTensor_matMul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mul(IntPtr src, IntPtr target);

        public ITorchTensor<long> Mul(ITorchTensor<long> target)
        {
            return new LongTensor(THSTensor_mul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_mul_(IntPtr src, IntPtr target);

        public void MulInPlace(ITorchTensor<long> target)
        {
            THSTensor_mul_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mulS(IntPtr src, long scalar);

        public ITorchTensor<long> Mul(long scalar)
        {
            return new LongTensor(THSTensor_mulS(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_pow(IntPtr src, float scalar);

        public ITorchTensor<long> Pow(float scalar)
        {
            return new LongTensor(THSTensor_pow(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sigmoid(IntPtr src);

        public ITorchTensor<long> Sigmoid()
        {
            return new LongTensor(THSTensor_sigmoid(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sub(IntPtr src, IntPtr trg);

        public ITorchTensor<long> Sub(ITorchTensor<long> target)
        {
            return new LongTensor(THSTensor_sub(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_sub_(IntPtr src, IntPtr trg);

        public void SubInPlace(ITorchTensor<long> target)
        {
            THSTensor_sub_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sum(IntPtr src);

        public ITorchTensor<long> Sum()
        {
            return new LongTensor(THSTensor_sum(handle));
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

        private static void CheckForCUDA(string device)
        {
            if (!Torch.IsCudaAvailable() && device.ToLower().Contains("cuda"))
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }
        }
    }
    /// <summary>
    ///   Tensor of type Double.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public struct DoubleTensor : ITorchTensor<double>
    {
        internal IntPtr handle;

        internal DoubleTensor(IntPtr handle)
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
        extern static IntPtr THSTensor_newDoubleScalar(double scalar);

        public static DoubleTensor From(double scalar)
        {
            return new DoubleTensor(THSTensor_newDoubleScalar(scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static DoubleTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new DoubleTensor(THSTensor_new(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Double));
        }

        public static DoubleTensor From(double[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (double* parray = rawArray)
                {
                    return DoubleTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static AtenSharp.DoubleTensor.HType THSTensor_unsafeGetTensorImpl(IntPtr handle);

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.DoubleTensor (THSTensor_unsafeGetTensorImpl (handle));
                return atenTensor.Dimensions;
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
        public Span<double> Data
        {
            get
            {               
                if (NumberOfElements > int.MaxValue)
                {
                    throw new ArgumentException("Span only supports up to int.MaxValue elements.");
                }
                unsafe
                {
                    return new System.Span<double>((void*)THSTensor_data(handle), (int)NumberOfElements);
                }
            }
        }

        public double Item
        {
            get
            {
                if (NumberOfElements != 1)
                {
                    throw new ArgumentException($"Number of elements in the tensor must be 1");
                }
                return Data[0];
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
        extern static IntPtr THSTensor_cpu(IntPtr handle);

        public ITorchTensor<double> Cpu()
        {
            return new DoubleTensor(THSTensor_cpu(handle));
        }

         [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_cuda(IntPtr handle);

        public ITorchTensor<double> Cuda()
        {
            if (!Torch.IsCudaAvailable())
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }

            return new DoubleTensor(THSTensor_cuda(handle));
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.DoubleTensor (THSTensor_unsafeGetTensorImpl (handle));
            return atenTensor.GetTensorDimension (dim);
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

        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            var atenTensor = new AtenSharp.DoubleTensor(THSTensor_unsafeGetTensorImpl(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor<double> Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            DoubleTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new DoubleTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Double, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<double> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            DoubleTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new DoubleTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Double, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<double> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            DoubleTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new DoubleTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Double, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_backward(IntPtr handle);

        public void Backward()
        {
            THSTensor_backward(handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_grad(IntPtr handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THSTensor_grad(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_reshape(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<double> Reshape(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new DoubleTensor (THSTensor_reshape (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_view(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<double> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new DoubleTensor (THSTensor_view (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_add(IntPtr src, int scalar, IntPtr trg);

        public ITorchTensor<double>  Add(ITorchTensor<double> target, int scalar)
        {
            return new DoubleTensor(THSTensor_add(handle, scalar, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_add_(IntPtr src, int scalar, IntPtr trg);

        public void AddInPlace(ITorchTensor<double> target, int scalar)
        {
            THSTensor_add_(handle, scalar, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public ITorchTensor<double> Addbmm(ITorchTensor<double> batch1, ITorchTensor<double> batch2, float beta, float alpha)
        {
            return new DoubleTensor(THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_argmax(IntPtr src, long dimension, bool keep_dim);

        public ITorchTensor<double> Argmax(long dimension, bool keepDim = false)
        {
            return new DoubleTensor(THSTensor_argmax(handle, dimension, keepDim));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta, float alpha);

        public ITorchTensor<double> Baddbmm(ITorchTensor<double> batch2, ITorchTensor<double> mat, float beta, float alpha)
        {
            return new DoubleTensor(THSTensor_addbmm(handle, batch2.Handle, mat.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_eq(IntPtr src, IntPtr trg);

        public ITorchTensor<U> Eq<U>(ITorchTensor<U> target)
        {
            return THSTensor_eq(handle, target.Handle).ToTorchTensor<U>();
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_exp(IntPtr src);

        public ITorchTensor<double> Exp()
        {
            return new DoubleTensor(THSTensor_exp(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_matMul(IntPtr src, IntPtr target);

        public ITorchTensor<double> MatMul(ITorchTensor<double> target)
        {
            return new DoubleTensor(THSTensor_matMul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mul(IntPtr src, IntPtr target);

        public ITorchTensor<double> Mul(ITorchTensor<double> target)
        {
            return new DoubleTensor(THSTensor_mul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_mul_(IntPtr src, IntPtr target);

        public void MulInPlace(ITorchTensor<double> target)
        {
            THSTensor_mul_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mulS(IntPtr src, double scalar);

        public ITorchTensor<double> Mul(double scalar)
        {
            return new DoubleTensor(THSTensor_mulS(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_pow(IntPtr src, float scalar);

        public ITorchTensor<double> Pow(float scalar)
        {
            return new DoubleTensor(THSTensor_pow(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sigmoid(IntPtr src);

        public ITorchTensor<double> Sigmoid()
        {
            return new DoubleTensor(THSTensor_sigmoid(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sub(IntPtr src, IntPtr trg);

        public ITorchTensor<double> Sub(ITorchTensor<double> target)
        {
            return new DoubleTensor(THSTensor_sub(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_sub_(IntPtr src, IntPtr trg);

        public void SubInPlace(ITorchTensor<double> target)
        {
            THSTensor_sub_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sum(IntPtr src);

        public ITorchTensor<double> Sum()
        {
            return new DoubleTensor(THSTensor_sum(handle));
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

        private static void CheckForCUDA(string device)
        {
            if (!Torch.IsCudaAvailable() && device.ToLower().Contains("cuda"))
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }
        }
    }
    /// <summary>
    ///   Tensor of type Float.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public struct FloatTensor : ITorchTensor<float>
    {
        internal IntPtr handle;

        internal FloatTensor(IntPtr handle)
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
        extern static IntPtr THSTensor_newFloatScalar(float scalar);

        public static FloatTensor From(float scalar)
        {
            return new FloatTensor(THSTensor_newFloatScalar(scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_new(IntPtr rawArray, long[] dimensions, int numDimensions, long[] strides, int numStrides, sbyte type);

        public static FloatTensor From(IntPtr rawArray, long[] dimensions)
        {
            var length = dimensions.Length;
            var strides = new long[length];

            strides[0] = 1;
            for (int i = 1; i < length; i++)
            {
                strides[i] = dimensions[i - 1];
            }

            return new FloatTensor(THSTensor_new(rawArray, dimensions, dimensions.Length, strides, strides.Length, (sbyte)ATenScalarMapping.Float));
        }

        public static FloatTensor From(float[] rawArray, long[] dimensions)
        {
            unsafe
            {
                fixed (float* parray = rawArray)
                {
                    return FloatTensor.From((IntPtr)parray, dimensions);
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static AtenSharp.FloatTensor.HType THSTensor_unsafeGetTensorImpl(IntPtr handle);

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.FloatTensor (THSTensor_unsafeGetTensorImpl (handle));
                return atenTensor.Dimensions;
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
        public Span<float> Data
        {
            get
            {               
                if (NumberOfElements > int.MaxValue)
                {
                    throw new ArgumentException("Span only supports up to int.MaxValue elements.");
                }
                unsafe
                {
                    return new System.Span<float>((void*)THSTensor_data(handle), (int)NumberOfElements);
                }
            }
        }

        public float Item
        {
            get
            {
                if (NumberOfElements != 1)
                {
                    throw new ArgumentException($"Number of elements in the tensor must be 1");
                }
                return Data[0];
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
        extern static IntPtr THSTensor_cpu(IntPtr handle);

        public ITorchTensor<float> Cpu()
        {
            return new FloatTensor(THSTensor_cpu(handle));
        }

         [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_cuda(IntPtr handle);

        public ITorchTensor<float> Cuda()
        {
            if (!Torch.IsCudaAvailable())
            {
                throw new InvalidOperationException("CUDA non available in the current machine.");
            }

            return new FloatTensor(THSTensor_cuda(handle));
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.FloatTensor (THSTensor_unsafeGetTensorImpl (handle));
            return atenTensor.GetTensorDimension (dim);
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

        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride (int dim)
        {
            var atenTensor = new AtenSharp.FloatTensor(THSTensor_unsafeGetTensorImpl(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_zeros(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with zeros
        /// </summary>
        static public ITorchTensor<float> Zeros(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            FloatTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new FloatTensor (THSTensor_zeros ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Float, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<float> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            FloatTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new FloatTensor (THSTensor_ones ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Float, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<float> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            FloatTensor.CheckForCUDA (device);

            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new FloatTensor (THSTensor_randn ((IntPtr)psizes, size.Length, (sbyte)ATenScalarMapping.Float, device, requiresGrad));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_backward(IntPtr handle);

        public void Backward()
        {
            THSTensor_backward(handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_grad(IntPtr handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THSTensor_grad(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_reshape(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<float> Reshape(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new FloatTensor (THSTensor_reshape (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_view(IntPtr src, IntPtr shape, int length);

        public ITorchTensor<float> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new FloatTensor (THSTensor_view (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_add(IntPtr src, int scalar, IntPtr trg);

        public ITorchTensor<float>  Add(ITorchTensor<float> target, int scalar)
        {
            return new FloatTensor(THSTensor_add(handle, scalar, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_add_(IntPtr src, int scalar, IntPtr trg);

        public void AddInPlace(ITorchTensor<float> target, int scalar)
        {
            THSTensor_add_(handle, scalar, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public ITorchTensor<float> Addbmm(ITorchTensor<float> batch1, ITorchTensor<float> batch2, float beta, float alpha)
        {
            return new FloatTensor(THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_argmax(IntPtr src, long dimension, bool keep_dim);

        public ITorchTensor<float> Argmax(long dimension, bool keepDim = false)
        {
            return new FloatTensor(THSTensor_argmax(handle, dimension, keepDim));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta, float alpha);

        public ITorchTensor<float> Baddbmm(ITorchTensor<float> batch2, ITorchTensor<float> mat, float beta, float alpha)
        {
            return new FloatTensor(THSTensor_addbmm(handle, batch2.Handle, mat.Handle, beta, alpha));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_eq(IntPtr src, IntPtr trg);

        public ITorchTensor<U> Eq<U>(ITorchTensor<U> target)
        {
            return THSTensor_eq(handle, target.Handle).ToTorchTensor<U>();
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_exp(IntPtr src);

        public ITorchTensor<float> Exp()
        {
            return new FloatTensor(THSTensor_exp(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_matMul(IntPtr src, IntPtr target);

        public ITorchTensor<float> MatMul(ITorchTensor<float> target)
        {
            return new FloatTensor(THSTensor_matMul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mul(IntPtr src, IntPtr target);

        public ITorchTensor<float> Mul(ITorchTensor<float> target)
        {
            return new FloatTensor(THSTensor_mul(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_mul_(IntPtr src, IntPtr target);

        public void MulInPlace(ITorchTensor<float> target)
        {
            THSTensor_mul_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_mulS(IntPtr src, float scalar);

        public ITorchTensor<float> Mul(float scalar)
        {
            return new FloatTensor(THSTensor_mulS(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_pow(IntPtr src, float scalar);

        public ITorchTensor<float> Pow(float scalar)
        {
            return new FloatTensor(THSTensor_pow(handle, scalar));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sigmoid(IntPtr src);

        public ITorchTensor<float> Sigmoid()
        {
            return new FloatTensor(THSTensor_sigmoid(handle));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sub(IntPtr src, IntPtr trg);

        public ITorchTensor<float> Sub(ITorchTensor<float> target)
        {
            return new FloatTensor(THSTensor_sub(handle, target.Handle));
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_sub_(IntPtr src, IntPtr trg);

        public void SubInPlace(ITorchTensor<float> target)
        {
            THSTensor_sub_(handle, target.Handle);
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSTensor_sum(IntPtr src);

        public ITorchTensor<float> Sum()
        {
            return new FloatTensor(THSTensor_sum(handle));
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

        private static void CheckForCUDA(string device)
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
        internal static ITorchTensor<T> ToTorchTensor<T>(this IntPtr rawTensor)
        {
            switch (true)
            {
                case bool _ when typeof(T) == typeof(byte):
                {
                    return new ByteTensor(rawTensor) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(short):
                {
                    return new ShortTensor(rawTensor) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(int):
                {
                    return new IntTensor(rawTensor) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(long):
                {
                    return new LongTensor(rawTensor) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(double):
                {
                    return new DoubleTensor(rawTensor) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(float):
                {
                    return new FloatTensor(rawTensor) as ITorchTensor<T>;
                }
                default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        public static ITorchTensor<T> ToTorchTensor<T>(this T[] rawArray, long[] dimensions)
        {
            switch (true)
            {
                case bool _ when typeof(T) == typeof(byte):
                {
                    return ByteTensor.From(rawArray as byte[], dimensions) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(short):
                {
                    return ShortTensor.From(rawArray as short[], dimensions) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(int):
                {
                    return IntTensor.From(rawArray as int[], dimensions) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(long):
                {
                    return LongTensor.From(rawArray as long[], dimensions) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(double):
                {
                    return DoubleTensor.From(rawArray as double[], dimensions) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(float):
                {
                    return FloatTensor.From(rawArray as float[], dimensions) as ITorchTensor<T>;
                }
                default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        public static ITorchTensor<T> ToTorchTensor<T>(this T scalar)
        {
            switch (true)
            {
                case bool _ when typeof(T) == typeof(byte):
                {
                    return ByteTensor.From((byte)(object)scalar) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(short):
                {
                    return ShortTensor.From((short)(object)scalar) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(int):
                {
                    return IntTensor.From((int)(object)scalar) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(long):
                {
                    return LongTensor.From((long)(object)scalar) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(double):
                {
                    return DoubleTensor.From((double)(object)scalar) as ITorchTensor<T>;
                }
                case bool _ when typeof(T) == typeof(float):
                {
                    return FloatTensor.From((float)(object)scalar) as ITorchTensor<T>;
                }
                default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        [DllImport("libTorchSharp")]
        extern static void THSTensor_initUniform(IntPtr src, double low, double high);

        internal static void InitUniform<T>(this ITorchTensor<T> tensor, double low = 0, double high = 1)
        {
            THSTensor_initUniform(tensor.Handle, low, high);
        }
    }
}
