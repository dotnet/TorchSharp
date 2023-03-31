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
        /// This class is used to represent a AdaptiveMaxPool3D module.
        /// </summary>
        public sealed class AdaptiveMaxPool3d : torch.nn.Module<Tensor, Tensor>
        {
            internal AdaptiveMaxPool3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AdaptiveMaxPool3d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            /// Applies a 3D adaptive max pooling over an input signal composed of several input planes.
            /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size of the image of the form D x H x W.
            /// Can be a tuple (D, H, W) or a single D for a cube D x D x D. D, H and W can be either a int, or null which means the size will be the same as that of the input.</param>
            /// <returns></returns>
            public static AdaptiveMaxPool3d AdaptiveMaxPool3d(long[] outputSize)
            {
                unsafe {
                    fixed (long* pkernelSize = outputSize) {
                        var handle = THSNN_AdaptiveMaxPool3d_ctor((IntPtr)pkernelSize, outputSize.Length, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new AdaptiveMaxPool3d(handle, boxedHandle);
                    }
                }
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 3D adaptive max pooling over an input signal composed of several input planes.
                /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="outputSize">The target output size of the image of the form D x H x W.
                /// Can be a tuple (D, H, W) or a single D for a cube D x D x D. D, H and W can be either a int, or null which means the size will be the same as that of the input.</param>
                /// <returns></returns>
                public static Tensor adaptive_max_pool3d(Tensor x, long[] outputSize)
                {
                    using (var d = nn.AdaptiveMaxPool3d(outputSize)) {
                        return d.call(x);
                    }
                }
            }
        }
    }
}
