// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    public static partial class jit
    {

        public class Type : IDisposable
        {
            /// <summary>
            ///    Class wrapping PyTorch's type object reference.
            /// </summary>
            internal sealed class HType : SafeHandle
            {
                public HType(IntPtr preexistingHandle, bool ownsHandle, TypeKind kind) : base(IntPtr.Zero, ownsHandle)
                {
                    SetHandle(preexistingHandle);
                    this.kind = kind;
                }

                public override bool IsInvalid => handle == IntPtr.Zero;

                // This is just for marshalling
                internal HType() : base(IntPtr.Zero, true)
                {
                }

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_Type_dispose(HType handle);

                [DllImport("LibTorchSharp")]
                private static extern void THSJIT_TensorType_dispose(HType handle);

                protected override bool ReleaseHandle()
                {
                    switch (kind) {
                    case TypeKind.TensorType:
                        THSJIT_TensorType_dispose(this);
                        break;
                    default:
                        THSJIT_Type_dispose(this);
                        break;
                    }
                    return true;
                }

                protected override void Dispose(bool disposing)
                {
                    if (disposing) {
                        ReleaseHandle();
                    }
                }

                private TypeKind kind;
            }

            internal HType handle;

            internal Type(IntPtr handle, TypeKind kind)
            {
                this.handle = new HType(handle, true, kind);
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
                if (disposing) {
                    handle.Dispose();
                    handle.SetHandleAsInvalid();
                }
            }

            [DllImport("LibTorchSharp")]
            private static extern sbyte THSJIT_Type_kind(HType handle);

            internal TypeKind Kind {
                get { return (TypeKind)THSJIT_Type_kind(handle); }
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSJIT_Type_cast(HType module);

            internal TensorType AsTensorType()
            {
                return new TensorType(THSJIT_Type_cast(handle));
            }

            internal DynamicType AsDynamicType()
            {
                return new DynamicType(THSJIT_Type_cast(handle));
            }

            internal enum TypeKind : sbyte
            {
                AnyType = 0,
                TensorType = 3,
            }
        }
    }
}