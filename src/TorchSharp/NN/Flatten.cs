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
        /// This class is used to represent a dropout module for 2d/3d convolutational layers.
        /// </summary>
        public class Flatten : torch.nn.Module
        {
            internal Flatten(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Flatten_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Flatten_forward(handle, tensor.Handle);
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
            extern static IntPtr THSNN_Flatten_ctor(long startDim, long endDim, out IntPtr pBoxedModule);

            static public Flatten Flatten(long startDim = 1, long endDim = -1)
            {
                var handle = THSNN_Flatten_ctor(startDim, endDim, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Flatten(handle, boxedHandle);
            }
        }

        public static partial class functional
        {
            static public Tensor Flatten(Tensor x, long startDim = 1, long endDim = -1)
            {
                using (var f = nn.Flatten(startDim, endDim)) {
                    return f.forward(x);
                }
            }
        }
    }
}
