// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;
    using static torch;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a AdaptiveAvgPool2D module.
        /// </summary>
        public sealed class AdaptiveAvgPool2d : torch.nn.Module<Tensor, Tensor>
        {
            internal AdaptiveAvgPool2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AdaptiveAvgPool2d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size (H,W) of the image of the form H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool2d AdaptiveAvgPool2d(long[] outputSize)
            {
                fixed (long* poutputSize = outputSize) {
                    var handle = THSNN_AdaptiveAvgPool2d_ctor((IntPtr)poutputSize, outputSize.Length, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new AdaptiveAvgPool2d(handle, boxedHandle);
                }
            }

            /// <summary>
            /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size (H,W) of the image of the form H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool2d AdaptiveAvgPool2d((long,long) outputSize)
            {
                long* poutputSize = stackalloc long[2] { outputSize.Item1, outputSize.Item2 };
                var handle = THSNN_AdaptiveAvgPool2d_ctor((IntPtr)poutputSize, 2, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdaptiveAvgPool2d(handle, boxedHandle);
            }

            /// <summary>
            /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size (H,W) of the image of the form H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool2d AdaptiveAvgPool2d(long outputSize)
            {
                long* poutputSize = stackalloc long[2] { outputSize, outputSize };
                var handle = THSNN_AdaptiveAvgPool2d_ctor((IntPtr)poutputSize, 2, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdaptiveAvgPool2d(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="output_size"></param>
                /// <returns></returns>
                public static Tensor adaptive_avg_pool2d(Tensor input, long[] output_size)
                {
                    unsafe {
                        fixed (long* poutputSize = output_size) {
                            var res = THSTensor_adaptive_avg_pool2d(input.Handle, (IntPtr)poutputSize, output_size.Length);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                /// <summary>
                /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="output_size"></param>
                /// <returns></returns>
                public static unsafe Tensor adaptive_avg_pool2d(Tensor input, (long, long) output_size)
                {
                    long* poutputSize = stackalloc long[2] { output_size.Item1, output_size.Item2 };

                    var res = THSTensor_adaptive_avg_pool2d(input.Handle, (IntPtr)poutputSize, 2);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="output_size"></param>
                /// <returns></returns>
                public static unsafe Tensor adaptive_avg_pool2d(Tensor input, long output_size)
                {
                    long* poutputSize = stackalloc long[2] { output_size, output_size };

                    var res = THSTensor_adaptive_avg_pool2d(input.Handle, (IntPtr)poutputSize, 2);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
