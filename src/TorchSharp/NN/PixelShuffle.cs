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
        public sealed class PixelShuffle : ParameterLessModule<Tensor, Tensor>
        {
            internal PixelShuffle(long upscale_factor) : base(nameof(PixelShuffle))
            {
                this.upscale_factor = upscale_factor;
            }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="input">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.pixel_shuffle(input, this.upscale_factor);
            }

            public long upscale_factor { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Rearranges elements in a tensor of shape (*, C * r^2, H, W) to a tensor of shape(*, C, H * r, W * r), where r is an upscale factor.
            /// This is useful for implementing efficient sub-pixel convolution with a stride of 1/r.
            /// </summary>
            /// <param name="upscale_factor">Factor to increase spatial resolution by</param>
            /// <returns></returns>
            public static PixelShuffle PixelShuffle(long upscale_factor)
            {
                return new PixelShuffle(upscale_factor);
            }

            public static partial class functional
            {
                /// <summary>
                /// Rearranges elements in a tensor of shape (*, C * r^2, H, W) to a tensor of shape(*, C, H * r, W * r), where r is an upscale factor.
                /// This is useful for implementing efficient sub-pixel convolution with a stride of 1/r.
                /// </summary>
                /// <param name="input">Input tensor</param>
                /// <param name="upscale_factor">Factor to increase spatial resolution by</param>
                /// <returns></returns>
                /// <returns></returns>
                public static Tensor pixel_shuffle(Tensor input, long upscale_factor)
                {
                    var res = THSNN_pixel_shuffle(input.Handle, upscale_factor);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
