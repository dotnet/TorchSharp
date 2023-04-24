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
        /// This class is used to represent a ZeroPad2d module.
        /// </summary>
        public sealed class ZeroPad2d : torch.nn.Module<Tensor, Tensor>
        {
            internal ZeroPad2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ZeroPad2d_forward(handle, tensor.Handle);
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
            /// Pads the input tensor boundaries with zero.
            /// </summary>
            /// <param name="padding">The size of the padding.</param>
            /// <returns></returns>
            public static ZeroPad2d ZeroPad2d(long padding)
            {
                var handle = THSNN_ZeroPad2d_ctor(padding, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ZeroPad2d(handle, boxedHandle);
            }

            /// <summary>
            /// Pads the input tensor boundaring with zero
            /// </summary>
            /// <param name="padding">The size of the padding: (padding_left, padding_right, padding_top, padding_bottom).</param>
            /// <returns></returns>
            public static ZeroPad2d ZeroPad2d((long, long, long, long) padding)
            {
                var handle = THSNN_ZeroPad2d_ctor_tuple(padding.Item1, padding.Item2, padding.Item3, padding.Item4, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ZeroPad2d(handle, boxedHandle);
            }
        }
    }
}
