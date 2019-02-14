using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp {

    /// <summary>
    ///   Tensor of type Byte.
    ///   This tensor maps to a Torch variable. 
    /// </summary>
    public class ByteTensor : IDisposable
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
        extern static IntPtr THS_data(HType handle);

        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public Span<byte> Data
        {
            get
            {
                
                int length;
                switch (Dimensions)
                {
                    case 0: 
                        length = 1;
                        break;
                    case 1:
                        length = (int)Shape[0];
                        break;
                    default:
                        length = (int)Shape.Aggregate((x, y) => x * y);
                        break;
                }
                
                unsafe
                {
                    return new System.Span<byte>((void*)THS_data(handle), length);
                }
            }
        }

        public byte Item()
        {
            unsafe
            {
                return Data[0];
            }
        }

        [DllImport("LibTorchSharp")]
        extern static string THS_device(HType handle);

        public string getDevice()
        {
            return THS_device(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ByteTensor Ones(long[] size, string device = "cpu:0", bool requiresGrad = false)
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
        static public ByteTensor RandomN(long[] size, string device = "cpu:0", bool requiresGrad = false)
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

        public FloatTensor Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, HType trg, bool is_grad);

        public ByteTensor SubInPlace(ByteTensor target, bool no_grad = true)
        {
            return new ByteTensor(THS_Sub_(handle, target.handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, byte scalar, bool is_grad);

        public ByteTensor Mul(byte scalar, bool no_grad = true)
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
            sb.Append($", device = {getDevice()}");
            return sb.ToString();
        }
    }
    /// <summary>
    ///   Tensor of type Short.
    ///   This tensor maps to a Torch variable. 
    /// </summary>
    public class ShortTensor : IDisposable
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
        extern static IntPtr THS_data(HType handle);

        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public Span<short> Data
        {
            get
            {
                
                int length;
                switch (Dimensions)
                {
                    case 0: 
                        length = 1;
                        break;
                    case 1:
                        length = (int)Shape[0];
                        break;
                    default:
                        length = (int)Shape.Aggregate((x, y) => x * y);
                        break;
                }
                
                unsafe
                {
                    return new System.Span<short>((void*)THS_data(handle), length);
                }
            }
        }

        public short Item()
        {
            unsafe
            {
                return Data[0];
            }
        }

        [DllImport("LibTorchSharp")]
        extern static string THS_device(HType handle);

        public string getDevice()
        {
            return THS_device(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public ShortTensor Ones(long[] size, string device = "cpu:0", bool requiresGrad = false)
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
        static public ShortTensor RandomN(long[] size, string device = "cpu:0", bool requiresGrad = false)
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

        public FloatTensor Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, HType trg, bool is_grad);

        public ShortTensor SubInPlace(ShortTensor target, bool no_grad = true)
        {
            return new ShortTensor(THS_Sub_(handle, target.handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, short scalar, bool is_grad);

        public ShortTensor Mul(short scalar, bool no_grad = true)
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
            sb.Append($", device = {getDevice()}");
            return sb.ToString();
        }
    }
    /// <summary>
    ///   Tensor of type Int.
    ///   This tensor maps to a Torch variable. 
    /// </summary>
    public class IntTensor : IDisposable
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
        extern static IntPtr THS_data(HType handle);

        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public Span<int> Data
        {
            get
            {
                
                int length;
                switch (Dimensions)
                {
                    case 0: 
                        length = 1;
                        break;
                    case 1:
                        length = (int)Shape[0];
                        break;
                    default:
                        length = (int)Shape.Aggregate((x, y) => x * y);
                        break;
                }
                
                unsafe
                {
                    return new System.Span<int>((void*)THS_data(handle), length);
                }
            }
        }

        public int Item()
        {
            unsafe
            {
                return Data[0];
            }
        }

        [DllImport("LibTorchSharp")]
        extern static string THS_device(HType handle);

        public string getDevice()
        {
            return THS_device(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public IntTensor Ones(long[] size, string device = "cpu:0", bool requiresGrad = false)
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
        static public IntTensor RandomN(long[] size, string device = "cpu:0", bool requiresGrad = false)
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

        public FloatTensor Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, HType trg, bool is_grad);

        public IntTensor SubInPlace(IntTensor target, bool no_grad = true)
        {
            return new IntTensor(THS_Sub_(handle, target.handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, int scalar, bool is_grad);

        public IntTensor Mul(int scalar, bool no_grad = true)
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
            sb.Append($", device = {getDevice()}");
            return sb.ToString();
        }
    }
    /// <summary>
    ///   Tensor of type Long.
    ///   This tensor maps to a Torch variable. 
    /// </summary>
    public class LongTensor : IDisposable
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
        extern static IntPtr THS_data(HType handle);

        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public Span<long> Data
        {
            get
            {
                
                int length;
                switch (Dimensions)
                {
                    case 0: 
                        length = 1;
                        break;
                    case 1:
                        length = (int)Shape[0];
                        break;
                    default:
                        length = (int)Shape.Aggregate((x, y) => x * y);
                        break;
                }
                
                unsafe
                {
                    return new System.Span<long>((void*)THS_data(handle), length);
                }
            }
        }

        public long Item()
        {
            unsafe
            {
                return Data[0];
            }
        }

        [DllImport("LibTorchSharp")]
        extern static string THS_device(HType handle);

        public string getDevice()
        {
            return THS_device(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public LongTensor Ones(long[] size, string device = "cpu:0", bool requiresGrad = false)
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
        static public LongTensor RandomN(long[] size, string device = "cpu:0", bool requiresGrad = false)
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

        public FloatTensor Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, HType trg, bool is_grad);

        public LongTensor SubInPlace(LongTensor target, bool no_grad = true)
        {
            return new LongTensor(THS_Sub_(handle, target.handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, long scalar, bool is_grad);

        public LongTensor Mul(long scalar, bool no_grad = true)
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
            sb.Append($", device = {getDevice()}");
            return sb.ToString();
        }
    }
    /// <summary>
    ///   Tensor of type Double.
    ///   This tensor maps to a Torch variable. 
    /// </summary>
    public class DoubleTensor : IDisposable
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
        extern static IntPtr THS_data(HType handle);

        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public Span<double> Data
        {
            get
            {
                
                int length;
                switch (Dimensions)
                {
                    case 0: 
                        length = 1;
                        break;
                    case 1:
                        length = (int)Shape[0];
                        break;
                    default:
                        length = (int)Shape.Aggregate((x, y) => x * y);
                        break;
                }
                
                unsafe
                {
                    return new System.Span<double>((void*)THS_data(handle), length);
                }
            }
        }

        public double Item()
        {
            unsafe
            {
                return Data[0];
            }
        }

        [DllImport("LibTorchSharp")]
        extern static string THS_device(HType handle);

        public string getDevice()
        {
            return THS_device(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public DoubleTensor Ones(long[] size, string device = "cpu:0", bool requiresGrad = false)
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
        static public DoubleTensor RandomN(long[] size, string device = "cpu:0", bool requiresGrad = false)
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

        public FloatTensor Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, HType trg, bool is_grad);

        public DoubleTensor SubInPlace(DoubleTensor target, bool no_grad = true)
        {
            return new DoubleTensor(THS_Sub_(handle, target.handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, double scalar, bool is_grad);

        public DoubleTensor Mul(double scalar, bool no_grad = true)
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
            sb.Append($", device = {getDevice()}");
            return sb.ToString();
        }
    }
    /// <summary>
    ///   Tensor of type Float.
    ///   This tensor maps to a Torch variable. 
    /// </summary>
    public class FloatTensor : IDisposable
    {
        [DllImport("LibTorchSharp")]
        extern static AtenSharp.FloatTensor.HType THS_getTHTensorUnsafe(HType handle);

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
                var atenTensor = new AtenSharp.FloatTensor(THS_getTHTensorUnsafe(this));
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
        extern static IntPtr THS_data(HType handle);

        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public Span<float> Data
        {
            get
            {
                
                int length;
                switch (Dimensions)
                {
                    case 0: 
                        length = 1;
                        break;
                    case 1:
                        length = (int)Shape[0];
                        break;
                    default:
                        length = (int)Shape.Aggregate((x, y) => x * y);
                        break;
                }
                
                unsafe
                {
                    return new System.Span<float>((void*)THS_data(handle), length);
                }
            }
        }

        public float Item()
        {
            unsafe
            {
                return Data[0];
            }
        }

        [DllImport("LibTorchSharp")]
        extern static string THS_device(HType handle);

        public string getDevice()
        {
            return THS_device(handle);
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_ones(IntPtr psizes, int scalarType, int length, string device, bool requireGrad);

        /// <summary>
        ///  Create a new tensor filled with ones
        /// </summary>
        static public FloatTensor Ones(long[] size, string device = "cpu:0", bool requiresGrad = false)
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
        static public FloatTensor RandomN(long[] size, string device = "cpu:0", bool requiresGrad = false)
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

        public FloatTensor Grad()
        {
            return new FloatTensor(THS_Grad(handle));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Sub_(HType src, HType trg, bool is_grad);

        public FloatTensor SubInPlace(FloatTensor target, bool no_grad = true)
        {
            return new FloatTensor(THS_Sub_(handle, target.handle, !no_grad));
        }

        [DllImport("LibTorchSharp")]
        extern static HType THS_Mul(HType src, float scalar, bool is_grad);

        public FloatTensor Mul(float scalar, bool no_grad = true)
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
            sb.Append($", device = {getDevice()}");
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
