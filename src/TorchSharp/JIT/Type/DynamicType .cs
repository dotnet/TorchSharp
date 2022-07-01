// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

namespace TorchSharp
{
    public static partial class torch
    {

        public static partial class jit
        {
            public sealed class DynamicType : Type
            {
                internal DynamicType(IntPtr handle) : base(handle, Type.TypeKind.AnyType)
                {
                    this.handle = new HType(handle, true, Type.TypeKind.AnyType);
                }
            }
        }
    }
}