// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a dropout module for 2d/3d convolutational layers.
        /// </summary>
        public sealed class Flatten : torch.nn.Module<Tensor, Tensor>
        {
            internal Flatten(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

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
            /// <summary>
            /// Flattens a contiguous range of dims into a tensor. For use with Sequential.
            /// </summary>
            /// <param name="startDim">First dim to flatten (default = 1).</param>
            /// <param name="endDim">Last dim to flatten (default = -1).</param>
            /// <returns></returns>
            public static Flatten Flatten(long startDim = 1, long endDim = -1)
            {
                var handle = THSNN_Flatten_ctor(startDim, endDim, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Flatten(handle, boxedHandle);
            }
        }
    }
}
