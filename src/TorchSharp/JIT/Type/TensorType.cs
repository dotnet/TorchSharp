// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

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

                public ScalarType GetScalarType()
                {
                    return (ScalarType)THSJIT_TensorType_dtype(handle);
                }

                /// <summary>
                ///  Retrieves the sizes of all dimensions of the tensor.
                /// </summary>
                public long[] size()
                {
                    long[] ptrArray;

                    using (var pa = new PinnedArray<long>()) {
                        THSJIT_TensorType_sizes(handle, pa.CreateArray);
                        CheckForErrors();
                        ptrArray = pa.Array;
                    }

                    return ptrArray;
                }

                public int GetDimensions()
                {
                    return THSJIT_getDimensionedTensorTypeDimensions(handle);
                }

                public string GetDevice()
                {
                    return THSJIT_getDimensionedTensorDevice(handle);
                }
            }
        }
    }
}
