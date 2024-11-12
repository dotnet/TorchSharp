// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a dropout module.
        /// </summary>
        public sealed class PixelUnshuffle : ParameterLessModule<Tensor, Tensor>
        {
            internal PixelUnshuffle(long downscale_factor) : base(nameof(PixelUnshuffle))
            {
                this.downscale_factor = downscale_factor;
            }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="input">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.pixel_unshuffle(input, downscale_factor);
            }

            public long downscale_factor { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {

            /// <summary>
            /// Reverses the PixelShuffle operation by rearranging elements in a tensor of shape (*, C, H * r, W * r) to a tensor of shape (*, C * r^2, H, W), where r is an downscale factor.
            /// </summary>
            /// <param name="downscale_factor">Factor to increase spatial resolution by</param>
            /// <returns></returns>
            public static PixelUnshuffle PixelUnshuffle(long downscale_factor)
            {
                return new PixelUnshuffle(downscale_factor);
            }

            public static partial class functional
            {
                /// <summary>
                /// Reverses the PixelShuffle operation by rearranging elements in a tensor of shape (*, C * r^2, H, W) to a tensor of shape(*, C, H * r, W * r), where r is an downscale factor.
                /// This is useful for implementing efficient sub-pixel convolution with a stride of 1/r.
                /// </summary>
                /// <param name="input">Input tensor</param>
                /// <param name="downscale_factor">Factor to increase spatial resolution by</param>
                /// <returns></returns>
                /// <returns></returns>
                public static Tensor pixel_unshuffle(Tensor input, long downscale_factor)
                {
                    var res = THSNN_pixel_unshuffle(input.Handle, downscale_factor);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
