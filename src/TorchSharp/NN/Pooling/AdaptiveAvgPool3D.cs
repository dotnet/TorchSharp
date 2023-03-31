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
        /// This class is used to represent a AdaptiveAvgPool3D module.
        /// </summary>
        public sealed class AdaptiveAvgPool3d : torch.nn.Module<Tensor, Tensor>
        {
            internal AdaptiveAvgPool3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AdaptiveAvgPool3d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size of the image of the form D x H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool3d AdaptiveAvgPool3d(long[] outputSize)
            {
                fixed (long* pkernelSize = outputSize) {
                    var handle = THSNN_AdaptiveAvgPool3d_ctor((IntPtr)pkernelSize, outputSize.Length, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new AdaptiveAvgPool3d(handle, boxedHandle);
                }
            }

            /// <summary>
            /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size (D,H,W) of the image of the form D x H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool3d AdaptiveAvgPool3d((long, long, long) outputSize)
            {
                long* pkernelSize = stackalloc long[3] { outputSize.Item1, outputSize.Item2, outputSize.Item3 };

                var handle = THSNN_AdaptiveAvgPool3d_ctor((IntPtr)pkernelSize, 3, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdaptiveAvgPool3d(handle, boxedHandle);
            }

            /// <summary>
            /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size (D,H,W) of the image of the form H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool3d AdaptiveAvgPool3d(long outputSize)
            {
                long* pkernelSize = stackalloc long[3] { outputSize, outputSize, outputSize };
                var handle = THSNN_AdaptiveAvgPool3d_ctor((IntPtr)pkernelSize, 3, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdaptiveAvgPool3d(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="output_size"></param>
                /// <returns></returns>
                public static unsafe Tensor adaptive_avg_pool3d(Tensor input, long[] output_size)
                {
                    fixed (long* poutputSize = output_size) {
                        var res =
                            THSTensor_adaptive_avg_pool3d(input.Handle, (IntPtr)poutputSize, output_size.Length);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }

                /// <summary>
                /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="output_size"></param>
                /// <returns></returns>
                public static unsafe Tensor adaptive_avg_pool3d(Tensor input, (long, long, long) output_size)
                {
                    long* poutputSize = stackalloc long[3] { output_size.Item1, output_size.Item2, output_size.Item3 };
                    var res = THSTensor_adaptive_avg_pool3d(input.Handle, (IntPtr)poutputSize, 3);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="output_size"></param>
                /// <returns></returns>
                public static unsafe Tensor adaptive_avg_pool3d(Tensor input, long output_size)
                {
                    var os = new long[] { output_size, output_size, output_size };
                    long* poutputSize = stackalloc long[3] { output_size, output_size, output_size };
                    var res = THSTensor_adaptive_avg_pool3d(input.Handle, (IntPtr)poutputSize, 3);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                public static Tensor adaptive_avg_pool3d_backward(Tensor gradInput, Tensor gradOutput, Tensor originalInput)
                {
                    var res = THSTensor_adaptive_avg_pool3d_backward_out(gradInput.Handle, gradOutput.Handle, originalInput.Handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
