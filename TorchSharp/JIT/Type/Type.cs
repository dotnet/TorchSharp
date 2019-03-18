using System;
using System.Runtime.InteropServices;

namespace TorchSharp.JIT
{
    public class Type : IDisposable
    {
        /// <summary>
        ///    Class wrapping PyTorch's type object reference.
        /// </summary>
        internal sealed class HType : SafeHandle
        {
            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
            {
                SetHandle(preexistingHandle);
            }

            public override bool IsInvalid => handle == IntPtr.Zero;

            // This is just for marshalling
            internal HType() : base(IntPtr.Zero, true)
            {
            }

            [DllImport("LibTorchSharp")]
            extern static void JIT_Type_Dispose(HType handle);

            protected override bool ReleaseHandle()
            {
                JIT_Type_Dispose(this);
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

        internal Type(IntPtr handle)
        {
            this.handle = new HType(handle, true);
        }

        protected Type()
        {
        }

        ~Type()
        {
            Dispose(false);
        }

        /// <summary>
        ///   Releases the storage.
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

        [DllImport("LibTorchSharp")]
        extern static short JIT_TypeKind(HType handle);

        internal TypeKind Kind
        {
            get { return (TypeKind)JIT_TypeKind(handle); }
        }

        internal TensorType AsTensorType()
        {
            return new TensorType(this);
        }

        internal DynamicType AsDynamicType()
        {
            return new DynamicType(this);
        }

        internal enum TypeKind : sbyte
        {
            DynamicType = 0,
            TensorType = 1
        }
    }
}
