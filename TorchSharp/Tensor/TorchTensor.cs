//using System;
//using System.Linq;
//using System.Runtime.InteropServices;
//using System.Text;

//namespace TorchSharp.Tensor
//{
//    public partial class TorchTensor
//    {
//        [DllImport("LibTorchSharp")]
//        extern static void THS_delete(HType handle);

//        [DllImport("LibTorchSharp")]
//        extern static IntPtr THS_getTHTensorUnsafe(HType handle);

//        internal sealed class HType : SafeHandle
//        {
//            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
//            {
//                SetHandle(preexistingHandle);
//            }

//            // This is just for marshalling
//            internal HType() : base(IntPtr.Zero, true)
//            {
//            }

//            public override bool IsInvalid => handle == (IntPtr)0;

//            protected override bool ReleaseHandle()
//            {
//                THS_delete(this);
//                return true;
//            }

//            protected override void Dispose(bool disposing)
//            {
//                if (disposing)
//                {
//                    ReleaseHandle();
//                }
//            }
//        }

//        internal HType handle;

//        int Dimensions { get; }

//        /// <summary>
//        ///  Returns a pointer to the unmanaged data managed by this tensor.
//        /// </summary>
//        public long NumberOfElements
//        {
//            get
//            {
//                switch (Dimensions)
//                {
//                    case 0:
//                        return 1;
//                    case 1:
//                        return (int)Shape[0];
//                    default:
//                        return (int)Shape.Aggregate((x, y) => x * y);
//                }
//            }
//        }

//        [DllImport("LibTorchSharp")]
//        extern static string THS_deviceType(HType handle);

//        public string Device
//        {
//            get
//            {
//                return THS_deviceType(handle);
//            }
//        }

//        internal TorchTensor(HType handle)
//        {
//            this.handle = handle;
//        }

//        /// <summary>
//        ///  Finalizer for ~TorchTensor
//        /// </summary>
//        ~TorchTensor()
//        {
//            Dispose(false);
//        }

//        /// <summary>
//        ///   Releases the tensor and its associated data.
//        /// </summary>
//        public void Dispose()
//        {
//            Dispose(true);
//            GC.SuppressFinalize(this);
//        }

//        /// <summary>
//        ///   Implements the .NET Dispose pattern.
//        /// </summary>
//        protected void Dispose(bool disposing)
//        {
//            if (disposing)
//            {
//                handle.Dispose();
//                handle.SetHandleAsInvalid();
//            }
//        }

//        internal TorchTensor CreateTensor<T>(HType handle)
//        {
//            switch (true)
//            {
//                case bool _ when typeof(T) == typeof(byte):
//                        return new ByteTensor(handle);
//                case bool _ when typeof(T) == typeof(short):
//                    return new ShortTensor(handle);
//                case bool _ when typeof(T) == typeof(int):
//                    return new IntTensor(handle);
//                case bool _ when typeof(T) == typeof(long):
//                    return new LongTensor(handle);
//                case bool _ when typeof(T) == typeof(double):
//                    return new DoubleTensor(handle);
//                case bool _ when typeof(T) == typeof(float):
//                    return new FloatTensor(handle);
//                default: throw new NotImplementedException($"Tensor type {typeof(T)} is not supported.");
//            }
//        }

//        /// <summary>
//        ///  Retrieves the size of the specified dimension in the tensor.
//        /// </summary>
//        public abstract long GetTensorDimension(int dim);

//        /// <summary>
//        /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
//        /// </summary>
//        /// <remarks>
//        ///     An array of size 0 is used for constants, an array of size 1 is used
//        ///     for single-dimension arrays, where the dimension is the value of the
//        ///     first element.   And so on.
//        /// </remarks>
//        public long[] Shape
//        {
//            get
//            {
//                var dims = new long[Dimensions];
//                for (int i = 0; i < dims.Length; i++)
//                    dims[i] = GetTensorDimension(i);

//                return dims;
//            }
//        }

//        /// <summary>
//        ///  Retrieves the stride of the specified dimension in the tensor.
//        /// </summary>
//        public abstract long GetTensorStride(int dim);

//        /// <summary>
//        ///  Returns a pointer to the unmanaged data managed by this tensor.
//        /// </summary>
//        public Span<T> Data<T>()
//        {
//            if (NumberOfElements > int.MaxValue)
//            {
//                throw new ArgumentException("Span only supports up to int.MaxValue elements.");
//            }
//            unsafe
//            {
//                return new System.Span<T>((void*)THS_data(handle), (int)NumberOfElements);
//            }
//        }

//        public T Item<T>()
//        {
//            if (NumberOfElements != 1)
//            {
//                throw new ArgumentException($"Number of elements in the tensor must be 1");
//            }
//            return Data<T>()[0];
//        }

//        [DllImport("LibTorchSharp")]
//        extern static HType THS_Grad(HType handle);

//        public TorchTensor Grad()
//        {
//            return new FloatTensor(THS_Grad(handle));
//        }

//        [DllImport("LibTorchSharp")]
//        extern static void THS_Backward(HType handle);

//        public void Backward()
//        {
//            THS_Backward(handle);
//        }

//        [DllImport("LibTorchSharp")]
//        extern static HType THS_Sub_(HType src, IntPtr trg, bool is_grad);

//        public TorchTensor SubInPlace<T>(ITorchTensor target, bool no_grad = true)
//        {
//            return new CreateTensor<T>(THS_Sub_(handle, target.Handle, !no_grad));
//        }

//        [DllImport("LibTorchSharp")]
//        extern static HType THS_Mul(HType src, float scalar, bool is_grad);

//        public ITorchTensor Mul<T>(T scalar, bool no_grad = true)
//        {
//            return new CreateTensor<T>(THS_Mul(handle, scalar, !no_grad));
//        }

//        /// <summary>
//        ///   Get a string representation of the tensor.
//        /// </summary>
//        public override string ToString()
//        {
//            var n = Dimensions;
//            if (n == 0)
//                return "[]";

//            StringBuilder sb = new StringBuilder("[");
//            for (int i = 0; i < n; i++)
//            {
//                sb.Append(GetTensorDimension(i));
//                if (i + 1 < n)
//                    sb.Append("x");
//            }
//            sb.Append("]");
//            sb.Append($", device = {Device}");
//            return sb.ToString();
//        }
//    }

//    internal enum ATenScalarMapping : short
//    {
//        Byte = 0,
//        Short = 2,
//        Int = 3,
//        Long = 4,
//        Float = 6,
//        Double = 7
//    }
//}
