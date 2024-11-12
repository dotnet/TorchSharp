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
        /// This class is used to represent a AdaptiveMaxPool2D module.
        /// </summary>
        public sealed class AdaptiveMaxPool2d : ParameterLessModule<Tensor, Tensor>
        {
            internal AdaptiveMaxPool2d(long[] output_size) : base(nameof(AdaptiveMaxPool2d))
            {
                this.output_size = output_size;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.adaptive_max_pool2d(input, this.output_size);
            }

            public (Tensor output, Tensor indices) forward_with_indices(Tensor input)
            {
                return torch.nn.functional.adaptive_max_pool2d_with_indices(input, this.output_size);
            }

            public long[] output_size { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 2D adaptive max pooling over an input signal composed of several input planes.
            /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="output_size">Applies a 2D adaptive max pooling over an input signal composed of several input planes.
            /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.</param>
            /// <returns></returns>
            public static AdaptiveMaxPool2d AdaptiveMaxPool2d(long[] output_size)
            {
                return new AdaptiveMaxPool2d(output_size);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 2D adaptive max pooling over an input signal composed of several input planes.
                /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="input"></param>
                /// <param name="output_size">Applies a 2D adaptive max pooling over an input signal composed of several input planes.
                /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.</param>
                /// <returns></returns>
                public static Tensor adaptive_max_pool2d(Tensor input, long[] output_size)
                {
                    var ret = adaptive_max_pool2d_with_indices(input, output_size);
                    ret.Indices.Dispose();
                    return ret.Values;
                }

                /// <summary>
                /// Applies a 2D adaptive max pooling over an input signal composed of several input planes.
                /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="input"></param>
                /// <param name="output_size">Applies a 2D adaptive max pooling over an input signal composed of several input planes.
                /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.</param>
                /// <returns></returns>
                public static (Tensor Values, Tensor Indices) adaptive_max_pool2d_with_indices(Tensor input, long[] output_size)
                {
                    unsafe {
                        fixed (long* poutputSize = output_size) {
                            var resOutput = THSTensor_adaptive_max_pool2d(input.Handle, (IntPtr)poutputSize, output_size.Length, out var resIndices);
                            if (resOutput == IntPtr.Zero || resIndices == IntPtr.Zero) { torch.CheckForErrors(); }
                            return (new Tensor(resOutput), new Tensor(resIndices));
                        }
                    }
                }
            }
        }
    }
}
