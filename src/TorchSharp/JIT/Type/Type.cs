//// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
//using System;
//using System.Runtime.InteropServices;

//namespace TorchSharp.JIT
//{
//    public class Type : IDisposable
//    {
//        /// <summary>
//        ///    Class wrapping PyTorch's type object reference.
//        /// </summary>
//        internal sealed class HType : SafeHandle
//        {
//            public HType(IntPtr preexistingHandle, bool ownsHandle) : base(IntPtr.Zero, ownsHandle)
//            {
//                SetHandle(preexistingHandle);
//            }

//            public override bool IsInvalid => handle == IntPtr.Zero;

//            // This is just for marshalling
//            internal HType() : base(IntPtr.Zero, true)
//            {
//            }

//            [DllImport("LibTorchSharp")]
//            private static extern void THSJIT_typeDispose(HType handle);

//            protected override bool ReleaseHandle()
//            {
//                THSJIT_typeDispose(this);
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

//        internal Type(IntPtr handle)
//        {
//            this.handle = new HType(handle, true);
//        }

//        protected Type()
//        {
//        }

//        ~Type()
//        {
//            Dispose(false);
//        }

//        /// <summary>
//        ///   Releases the storage.
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

//        [DllImport("LibTorchSharp")]
//        private static extern sbyte THSJIT_typeKind(HType handle);

//        internal TypeKind Kind
//        {
//            get { return (TypeKind)THSJIT_typeKind(handle); }
//        }

//        [DllImport("LibTorchSharp")]
//        private static extern IntPtr THSJIT_typeCast(HType module);

//        internal TensorType AsTensorType()
//        {
//            return new TensorType(THSJIT_typeCast(handle));
//        }

//        internal DynamicType AsDynamicType()
//        {
//            return new DynamicType(THSJIT_typeCast(handle));
//        }

//        internal enum TypeKind : sbyte
//        {
//            TensorType = 0,
//            DimensionedTensorType = 1
//        }
//    }
//}
