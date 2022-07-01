// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class jit
        {
            public sealed class TensorType : Type
            {
                internal TensorType(IntPtr handle) : base(handle, TypeKind.TensorType)
                {
                    this.handle = new HType(handle, true, TypeKind.TensorType);
                }

                internal TensorType(Type type) : base()
                {
                    handle = type.handle;
                    type.handle = new HType(IntPtr.Zero, true, TypeKind.TensorType);
                    type.Dispose();
                }

                [DllImport("LibTorchSharp")]
                private static extern sbyte THSJIT_TensorType_dtype(HType handle);

                public torch.ScalarType GetScalarType()
                {
                    return (torch.ScalarType)THSJIT_TensorType_dtype(handle);
                }


                [DllImport("LibTorchSharp")]
                static extern long THSJIT_TensorType_sizes(HType handle, AllocatePinnedArray allocator);

                /// <summary>
                ///  Retrieves the sizes of all dimensions of the tensor.
                /// </summary>
                public long[] size()
                {
                    long[] ptrArray;

                    using (var pa = new PinnedArray<long>()) {
                        THSJIT_TensorType_sizes(handle, pa.CreateArray);
                        torch.CheckForErrors();
                        ptrArray = pa.Array;
                    }

                    return ptrArray;
                }

                [DllImport("LibTorchSharp")]
                private static extern int THSJIT_getDimensionedTensorTypeDimensions(HType handle);

                public int GetDimensions()
                {
                    return THSJIT_getDimensionedTensorTypeDimensions(handle);
                }

                [DllImport("LibTorchSharp")]
                private static extern string THSJIT_getDimensionedTensorDevice(HType handle);

                public string GetDevice()
                {
                    return THSJIT_getDimensionedTensorDevice(handle);
                }
            }
        }
    }
}
