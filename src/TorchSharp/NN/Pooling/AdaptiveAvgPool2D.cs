// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;
    using static torch;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a AdaptiveAvgPool2D module.
        /// </summary>
        public sealed class AdaptiveAvgPool2d : ParameterLessModule<Tensor, Tensor>
        {
            internal AdaptiveAvgPool2d(long[] output_size) : base(nameof(AdaptiveAvgPool2d))
            {
                this.output_size = output_size;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.adaptive_avg_pool2d(input, this.output_size);
            }

            public long[] output_size { get; set; }
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
            /// <param name="output_size">The target output size (H,W) of the image of the form H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool2d AdaptiveAvgPool2d(long[] output_size)
            {
                return new AdaptiveAvgPool2d(output_size);
            }

            /// <summary>
            /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="output_size">The target output size (H,W) of the image of the form H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool2d AdaptiveAvgPool2d((long,long) output_size)
            {
                return new AdaptiveAvgPool2d(new[] { output_size.Item1, output_size.Item2 });
            }

            /// <summary>
            /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="output_size">The target output size (H,W) of the image of the form H x W.</param>
            /// <returns></returns>
            public static unsafe AdaptiveAvgPool2d AdaptiveAvgPool2d(long output_size)
            {
                return new AdaptiveAvgPool2d(new[] { output_size, output_size });
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
