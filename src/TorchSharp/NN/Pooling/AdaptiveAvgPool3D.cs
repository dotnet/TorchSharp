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
        /// This class is used to represent a AdaptiveAvgPool3D module.
        /// </summary>
        public sealed class AdaptiveAvgPool3d : ParameterLessModule<Tensor, Tensor>
        {
            internal AdaptiveAvgPool3d(long[] output_size) : base(nameof(AdaptiveAvgPool3d))
            {
                this.output_size = output_size;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.adaptive_avg_pool3d(input, this.output_size);
            }

            public long[] output_size { get; set; }
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
            /// <param name="output_size">The target output size of the image of the form D x H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool3d AdaptiveAvgPool3d(long[] output_size)
            {
                return new AdaptiveAvgPool3d(output_size);
            }

            /// <summary>
            /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="output_size">The target output size (D,H,W) of the image of the form D x H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool3d AdaptiveAvgPool3d((long, long, long) output_size)
            {
                return new AdaptiveAvgPool3d(new[] { output_size.Item1, output_size.Item2, output_size.Item3 });
            }

            /// <summary>
            /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="output_size">The target output size (D,H,W) of the image of the form H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool3d AdaptiveAvgPool3d(long output_size)
            {
                return new AdaptiveAvgPool3d(new [] { output_size, output_size, output_size });
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
