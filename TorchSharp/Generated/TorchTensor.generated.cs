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
    public class ByteTensor : ITorchTensor<byte>
    {
        [DllImport("LibTorchSharp")]
        extern static AtenSharp.ByteTensor.HType THS_getTHTensorUnsafe(HType handle);

        internal sealed class HType : SafeHandle
        {
            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
            {
                SetHandle(preexistingHandle);
            }

            // This is just for marshalling
            internal HType() : base(IntPtr.Zero, true)
            {
            }

            public override bool IsInvalid => handle == (IntPtr)0;

            protected override bool ReleaseHandle()
            {
                var atenTensor = new AtenSharp.ByteTensor(THS_getTHTensorUnsafe(this));
                atenTensor.Dispose();
                return true;
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    ReleaseHandle();
                }
            }
        }

        internal HType handle;

        internal ByteTensor(HType handle)
        {
            this.handle = handle;
        }

        /// <summary>
        ///  Finalizer for ~ByteTensor
        /// </summary>
        ~ByteTensor()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases the tensor and its associated data.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose(bool disposing)
        {
            if (disposing)
            {
                handle.Dispose();
                handle.SetHandleAsInvalid();
            }
        }

        public IntPtr Handle
        {
            get
            {
                return handle.DangerousGetHandle();
            }
        }

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.ByteTensor (THS_getTHTensorUnsafe (handle));
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

        [DllImport("LibTorchSharp")]
        extern static IntPtr THS_data(HType handle);

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
                    return new System.Span<byte>((void*)THS_data(handle), (int)NumberOfElements);
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

        [DllImport("LibTorchSharp")]
        extern static string THS_deviceType(HType handle);

        public string Device
        {
            get
            {
                return THS_deviceType(handle);
            }
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.ByteTensor (THS_getTHTensorUnsafe (handle));
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
            var atenTensor = new AtenSharp.ByteTensor(THS_getTHTensorUnsafe(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<byte> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new ByteTensor (THS_ones ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Byte, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<byte> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new ByteTensor (THS_randn ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Byte, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static void THS_Backward(HType handle);

        public void Backward()
        {
            THS_Backward(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType THS_Grad(HType handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_View(HType src, IntPtr shape, int length);

        public ITorchTensor<byte> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new ByteTensor (THS_View (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, IntPtr trg, bool is_grad);

        public ITorchTensor<byte> SubInPlace(ITorchTensor<byte> target, bool no_grad = true)
        {
            return new ByteTensor(THS_Sub_(handle, target.Handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, byte scalar, bool is_grad);

        public ITorchTensor<byte> Mul(byte scalar, bool no_grad = true)
        {
            return new ByteTensor(THS_Mul(handle, scalar, !no_grad));
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
    }
    /// <summary>
    ///   Tensor of type Short.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class ShortTensor : ITorchTensor<short>
    {
        [DllImport("LibTorchSharp")]
        extern static AtenSharp.ShortTensor.HType THS_getTHTensorUnsafe(HType handle);

        internal sealed class HType : SafeHandle
        {
            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
            {
                SetHandle(preexistingHandle);
            }

            // This is just for marshalling
            internal HType() : base(IntPtr.Zero, true)
            {
            }

            public override bool IsInvalid => handle == (IntPtr)0;

            protected override bool ReleaseHandle()
            {
                var atenTensor = new AtenSharp.ShortTensor(THS_getTHTensorUnsafe(this));
                atenTensor.Dispose();
                return true;
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    ReleaseHandle();
                }
            }
        }

        internal HType handle;

        internal ShortTensor(HType handle)
        {
            this.handle = handle;
        }

        /// <summary>
        ///  Finalizer for ~ShortTensor
        /// </summary>
        ~ShortTensor()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases the tensor and its associated data.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose(bool disposing)
        {
            if (disposing)
            {
                handle.Dispose();
                handle.SetHandleAsInvalid();
            }
        }

        public IntPtr Handle
        {
            get
            {
                return handle.DangerousGetHandle();
            }
        }

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.ShortTensor (THS_getTHTensorUnsafe (handle));
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

        [DllImport("LibTorchSharp")]
        extern static IntPtr THS_data(HType handle);

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
                    return new System.Span<short>((void*)THS_data(handle), (int)NumberOfElements);
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

        [DllImport("LibTorchSharp")]
        extern static string THS_deviceType(HType handle);

        public string Device
        {
            get
            {
                return THS_deviceType(handle);
            }
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.ShortTensor (THS_getTHTensorUnsafe (handle));
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
            var atenTensor = new AtenSharp.ShortTensor(THS_getTHTensorUnsafe(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<short> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new ShortTensor (THS_ones ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Short, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<short> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new ShortTensor (THS_randn ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Short, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static void THS_Backward(HType handle);

        public void Backward()
        {
            THS_Backward(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType THS_Grad(HType handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_View(HType src, IntPtr shape, int length);

        public ITorchTensor<short> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new ShortTensor (THS_View (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, IntPtr trg, bool is_grad);

        public ITorchTensor<short> SubInPlace(ITorchTensor<short> target, bool no_grad = true)
        {
            return new ShortTensor(THS_Sub_(handle, target.Handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, short scalar, bool is_grad);

        public ITorchTensor<short> Mul(short scalar, bool no_grad = true)
        {
            return new ShortTensor(THS_Mul(handle, scalar, !no_grad));
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
    }
    /// <summary>
    ///   Tensor of type Int.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class IntTensor : ITorchTensor<int>
    {
        [DllImport("LibTorchSharp")]
        extern static AtenSharp.IntTensor.HType THS_getTHTensorUnsafe(HType handle);

        internal sealed class HType : SafeHandle
        {
            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
            {
                SetHandle(preexistingHandle);
            }

            // This is just for marshalling
            internal HType() : base(IntPtr.Zero, true)
            {
            }

            public override bool IsInvalid => handle == (IntPtr)0;

            protected override bool ReleaseHandle()
            {
                var atenTensor = new AtenSharp.IntTensor(THS_getTHTensorUnsafe(this));
                atenTensor.Dispose();
                return true;
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    ReleaseHandle();
                }
            }
        }

        internal HType handle;

        internal IntTensor(HType handle)
        {
            this.handle = handle;
        }

        /// <summary>
        ///  Finalizer for ~IntTensor
        /// </summary>
        ~IntTensor()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases the tensor and its associated data.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose(bool disposing)
        {
            if (disposing)
            {
                handle.Dispose();
                handle.SetHandleAsInvalid();
            }
        }

        public IntPtr Handle
        {
            get
            {
                return handle.DangerousGetHandle();
            }
        }

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.IntTensor (THS_getTHTensorUnsafe (handle));
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

        [DllImport("LibTorchSharp")]
        extern static IntPtr THS_data(HType handle);

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
                    return new System.Span<int>((void*)THS_data(handle), (int)NumberOfElements);
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

        [DllImport("LibTorchSharp")]
        extern static string THS_deviceType(HType handle);

        public string Device
        {
            get
            {
                return THS_deviceType(handle);
            }
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.IntTensor (THS_getTHTensorUnsafe (handle));
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
            var atenTensor = new AtenSharp.IntTensor(THS_getTHTensorUnsafe(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<int> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new IntTensor (THS_ones ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Int, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<int> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new IntTensor (THS_randn ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Int, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static void THS_Backward(HType handle);

        public void Backward()
        {
            THS_Backward(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType THS_Grad(HType handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_View(HType src, IntPtr shape, int length);

        public ITorchTensor<int> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new IntTensor (THS_View (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, IntPtr trg, bool is_grad);

        public ITorchTensor<int> SubInPlace(ITorchTensor<int> target, bool no_grad = true)
        {
            return new IntTensor(THS_Sub_(handle, target.Handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, int scalar, bool is_grad);

        public ITorchTensor<int> Mul(int scalar, bool no_grad = true)
        {
            return new IntTensor(THS_Mul(handle, scalar, !no_grad));
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
    }
    /// <summary>
    ///   Tensor of type Long.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class LongTensor : ITorchTensor<long>
    {
        [DllImport("LibTorchSharp")]
        extern static AtenSharp.LongTensor.HType THS_getTHTensorUnsafe(HType handle);

        internal sealed class HType : SafeHandle
        {
            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
            {
                SetHandle(preexistingHandle);
            }

            // This is just for marshalling
            internal HType() : base(IntPtr.Zero, true)
            {
            }

            public override bool IsInvalid => handle == (IntPtr)0;

            protected override bool ReleaseHandle()
            {
                var atenTensor = new AtenSharp.LongTensor(THS_getTHTensorUnsafe(this));
                atenTensor.Dispose();
                return true;
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    ReleaseHandle();
                }
            }
        }

        internal HType handle;

        internal LongTensor(HType handle)
        {
            this.handle = handle;
        }

        /// <summary>
        ///  Finalizer for ~LongTensor
        /// </summary>
        ~LongTensor()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases the tensor and its associated data.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose(bool disposing)
        {
            if (disposing)
            {
                handle.Dispose();
                handle.SetHandleAsInvalid();
            }
        }

        public IntPtr Handle
        {
            get
            {
                return handle.DangerousGetHandle();
            }
        }

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.LongTensor (THS_getTHTensorUnsafe (handle));
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

        [DllImport("LibTorchSharp")]
        extern static IntPtr THS_data(HType handle);

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
                    return new System.Span<long>((void*)THS_data(handle), (int)NumberOfElements);
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

        [DllImport("LibTorchSharp")]
        extern static string THS_deviceType(HType handle);

        public string Device
        {
            get
            {
                return THS_deviceType(handle);
            }
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.LongTensor (THS_getTHTensorUnsafe (handle));
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
            var atenTensor = new AtenSharp.LongTensor(THS_getTHTensorUnsafe(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<long> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new LongTensor (THS_ones ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Long, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<long> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new LongTensor (THS_randn ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Long, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static void THS_Backward(HType handle);

        public void Backward()
        {
            THS_Backward(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType THS_Grad(HType handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_View(HType src, IntPtr shape, int length);

        public ITorchTensor<long> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new LongTensor (THS_View (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, IntPtr trg, bool is_grad);

        public ITorchTensor<long> SubInPlace(ITorchTensor<long> target, bool no_grad = true)
        {
            return new LongTensor(THS_Sub_(handle, target.Handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, long scalar, bool is_grad);

        public ITorchTensor<long> Mul(long scalar, bool no_grad = true)
        {
            return new LongTensor(THS_Mul(handle, scalar, !no_grad));
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
    }
    /// <summary>
    ///   Tensor of type Double.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class DoubleTensor : ITorchTensor<double>
    {
        [DllImport("LibTorchSharp")]
        extern static AtenSharp.DoubleTensor.HType THS_getTHTensorUnsafe(HType handle);

        internal sealed class HType : SafeHandle
        {
            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
            {
                SetHandle(preexistingHandle);
            }

            // This is just for marshalling
            internal HType() : base(IntPtr.Zero, true)
            {
            }

            public override bool IsInvalid => handle == (IntPtr)0;

            protected override bool ReleaseHandle()
            {
                var atenTensor = new AtenSharp.DoubleTensor(THS_getTHTensorUnsafe(this));
                atenTensor.Dispose();
                return true;
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    ReleaseHandle();
                }
            }
        }

        internal HType handle;

        internal DoubleTensor(HType handle)
        {
            this.handle = handle;
        }

        /// <summary>
        ///  Finalizer for ~DoubleTensor
        /// </summary>
        ~DoubleTensor()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases the tensor and its associated data.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose(bool disposing)
        {
            if (disposing)
            {
                handle.Dispose();
                handle.SetHandleAsInvalid();
            }
        }

        public IntPtr Handle
        {
            get
            {
                return handle.DangerousGetHandle();
            }
        }

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.DoubleTensor (THS_getTHTensorUnsafe (handle));
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

        [DllImport("LibTorchSharp")]
        extern static IntPtr THS_data(HType handle);

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
                    return new System.Span<double>((void*)THS_data(handle), (int)NumberOfElements);
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

        [DllImport("LibTorchSharp")]
        extern static string THS_deviceType(HType handle);

        public string Device
        {
            get
            {
                return THS_deviceType(handle);
            }
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.DoubleTensor (THS_getTHTensorUnsafe (handle));
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
            var atenTensor = new AtenSharp.DoubleTensor(THS_getTHTensorUnsafe(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<double> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new DoubleTensor (THS_ones ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Double, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<double> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new DoubleTensor (THS_randn ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Double, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static void THS_Backward(HType handle);

        public void Backward()
        {
            THS_Backward(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType THS_Grad(HType handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_View(HType src, IntPtr shape, int length);

        public ITorchTensor<double> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new DoubleTensor (THS_View (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, IntPtr trg, bool is_grad);

        public ITorchTensor<double> SubInPlace(ITorchTensor<double> target, bool no_grad = true)
        {
            return new DoubleTensor(THS_Sub_(handle, target.Handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, double scalar, bool is_grad);

        public ITorchTensor<double> Mul(double scalar, bool no_grad = true)
        {
            return new DoubleTensor(THS_Mul(handle, scalar, !no_grad));
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
    }
    /// <summary>
    ///   Tensor of type Float.
    ///   This tensor maps to a Torch variable (see torch/csrc/autograd/variable.h).
    //    Please do no mix Aten Tensors and Torch Tensors.
    /// </summary>
    public class FloatTensor : ITorchTensor<float>
    {
        [DllImport("LibTorchSharp")]
        extern static AtenSharp.FloatTensor.HType THS_getTHTensorUnsafe(HType handle);

        [DllImport("LibTorchSharp")]
        extern static void THS_Delete(HType handle);

        internal sealed class HType : SafeHandle
        {
            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
            {
                SetHandle(preexistingHandle);
            }

            // This is just for marshalling
            internal HType() : base(IntPtr.Zero, true)
            {
            }

            public override bool IsInvalid => handle == (IntPtr)0;

            protected override bool ReleaseHandle()
            {
                //var atenTensor = new AtenSharp.FloatTensor(THS_getTHTensorUnsafe(this));
                //atenTensor.Dispose();
                THS_Delete(this);
                return true;
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    ReleaseHandle();
                }
            }
        }

        internal HType handle;

        internal FloatTensor(HType handle)
        {
            this.handle = handle;
        }

        /// <summary>
        ///  Finalizer for ~FloatTensor
        /// </summary>
        ~FloatTensor()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases the tensor and its associated data.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        protected void Dispose(bool disposing)
        {
            if (disposing)
            {
                handle.Dispose();
                handle.SetHandleAsInvalid();
            }
        }

        public IntPtr Handle
        {
            get
            {
                return handle.DangerousGetHandle();
            }
        }

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public int Dimensions
        {
            get
            {
                var atenTensor = new AtenSharp.FloatTensor (THS_getTHTensorUnsafe (handle));
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

        [DllImport("LibTorchSharp")]
        extern static IntPtr THS_data(HType handle);

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
                    return new System.Span<float>((void*)THS_data(handle), (int)NumberOfElements);
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

        [DllImport("LibTorchSharp")]
        extern static string THS_deviceType(HType handle);

        public string Device
        {
            get
            {
                return THS_deviceType(handle);
            }
        }

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var atenTensor = new AtenSharp.FloatTensor (THS_getTHTensorUnsafe (handle));
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
            var atenTensor = new AtenSharp.FloatTensor(THS_getTHTensorUnsafe(handle));
            return atenTensor.GetTensorStride(dim);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<float> Ones(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new FloatTensor (THS_ones ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Float, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_randn(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ITorchTensor<float> RandomN(long[] size, string device = "cpu", bool requiresGrad = false)
        {
            unsafe
            {
                fixed (long* psizes = size)
                {
                    return new FloatTensor (THS_randn ((IntPtr)psizes, size.Length, (short)ATenScalarMapping.Float, device, requiresGrad));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static void THS_Backward(HType handle);

        public void Backward()
        {
            THS_Backward(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType THS_Grad(HType handle);

        public ITorchTensor<float> Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_View(HType src, IntPtr shape, int length);

        public ITorchTensor<float> View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    return new FloatTensor (THS_View (handle, (IntPtr)pshape, shape.Length));
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, IntPtr trg, bool is_grad);

        public ITorchTensor<float> SubInPlace(ITorchTensor<float> target, bool no_grad = true)
        {
            return new FloatTensor(THS_Sub_(handle, target.Handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, float scalar, bool is_grad);

        public ITorchTensor<float> Mul(float scalar, bool no_grad = true)
        {
            return new FloatTensor(THS_Mul(handle, scalar, !no_grad));
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
    }
    
    internal enum ATenScalarMapping : short
    {
        Byte = 0,
        Short = 2,
        Int = 3,
        Long = 4,
        Float = 6,
        Double = 7
    }
}
