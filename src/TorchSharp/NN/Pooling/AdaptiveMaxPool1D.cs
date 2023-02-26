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
        /// This class is used to represent a AdaptiveMaxPool1D module.
        /// </summary>
        public sealed class AdaptiveMaxPool1d : torch.nn.Module<Tensor, Tensor>
        {
            internal AdaptiveMaxPool1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AdaptiveMaxPool1d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            /// Applies a 1D adaptive max pooling over an input signal composed of several input planes.
            /// The output size is H, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size H.</param>
            /// <returns></returns>
            public static AdaptiveMaxPool1d AdaptiveMaxPool1d(long outputSize)
            {
                unsafe {
                    fixed (long* pkernelSize = new long[] { outputSize }) {
                        var handle = THSNN_AdaptiveMaxPool1d_ctor((IntPtr)pkernelSize, 1, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new AdaptiveMaxPool1d(handle, boxedHandle);
                    }
                }
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 1D adaptive max pooling over an input signal composed of several input planes.
                /// The output size is H, for any input size.The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="x"></param>
                /// <param name="outputSize">The target output size H.</param>
                /// <returns></returns>
                public static Tensor adaptive_max_pool1d(Tensor x, long outputSize)
                {
                    using (var d = nn.AdaptiveMaxPool1d(outputSize)) {
                        return d.call(x);
                    }
                }
            }
        }
    }
}
