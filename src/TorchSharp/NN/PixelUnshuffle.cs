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
        /// This class is used to represent a dropout module.
        /// </summary>
        public sealed class PixelUnshuffle : torch.nn.Module<Tensor, Tensor>
        {
            internal PixelUnshuffle(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_PixelUnshuffle_forward(handle, tensor.Handle);
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
            /// Reverses the PixelShuffle operation by rearranging elements in a tensor of shape (*, C, H * r, W * r) to a tensor of shape (*, C * r^2, H, W), where r is an downscale factor.
            /// </summary>
            /// <param name="downscaleFactor">Factor to increase spatial resolution by</param>
            /// <returns></returns>
            public static PixelUnshuffle PixelUnshuffle(long downscaleFactor)
            {
                var handle = THSNN_PixelUnshuffle_ctor(downscaleFactor, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new PixelUnshuffle(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Reverses the PixelShuffle operation by rearranging elements in a tensor of shape (*, C * r^2, H, W) to a tensor of shape(*, C, H * r, W * r), where r is an downscale factor.
                /// This is useful for implementing efficient sub-pixel convolution with a stride of 1/r.
                /// </summary>
                /// <param name="x">Input tensor</param>
                /// <param name="downscaleFactor">Factor to increase spatial resolution by</param>
                /// <returns></returns>
                /// <returns></returns>
                public static Tensor pixel_unshuffle(Tensor x, long downscaleFactor)
                {
                    using (var d = nn.PixelUnshuffle(downscaleFactor)) {
                        return d.call(x);
                    }
                }
            }
        }
    }
}
