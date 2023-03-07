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
        /// This class is used to represent a AdaptiveAvgPool1D module.
        /// </summary>
        public sealed class AdaptiveAvgPool1d : torch.nn.Module<Tensor, Tensor>
        {
            internal AdaptiveAvgPool1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AdaptiveAvgPool1d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            /// Applies a 1D adaptive average pooling over an input signal composed of several input planes.
            /// The output size is H, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">the target output size H</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool1d AdaptiveAvgPool1d(long outputSize)
            {
                long* pkernelSize = stackalloc long[1] { outputSize };
                var handle = THSNN_AdaptiveAvgPool1d_ctor((IntPtr)pkernelSize, 1, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdaptiveAvgPool1d(handle, boxedHandle);
            }

            public static partial class functional
            {

                /// <summary>
                /// Applies a 1D adaptive average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="output_size"></param>
                /// <returns></returns>
                public static Tensor adaptive_avg_pool1d(Tensor input, long output_size)
                {
                    var outputSizes = new long[] { output_size };
                    unsafe {
                        fixed (long* poutputSize = outputSizes) {
                            var res =
                                THSTensor_adaptive_avg_pool1d(input.Handle, (IntPtr)poutputSize, outputSizes.Length);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }
            }
        }
    }
}
