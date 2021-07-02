// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using impl;

    namespace impl
    {
        /// <summary>
        /// This class is used to represent an unflattening operation.
        /// </summary>
        public class Unflatten : torch.nn.Module
        {
            internal Unflatten(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Unflatten_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Unflatten_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Unflatten_ctor(long dim, IntPtr shape, long shape_len, out IntPtr pBoxedModule);

            static public Unflatten Unflatten(long dim, long[] unflattenedSize)
            {
                unsafe {
                    fixed (long* pUnflattenedSize = unflattenedSize) {
                        var handle = THSNN_Unflatten_ctor(dim, (IntPtr)pUnflattenedSize, unflattenedSize.Length, out var boxedHandle);
                        if (handle == IntPtr.Zero) { CheckForErrors(); }
                        return new Unflatten(handle, boxedHandle);
                    }
                }
            }
        }
    }
}
