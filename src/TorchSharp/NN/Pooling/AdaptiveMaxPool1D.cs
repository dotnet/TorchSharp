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
        /// This class is used to represent a AdaptiveMaxPool1D module.
        /// </summary>
        public sealed class AdaptiveMaxPool1d : ParameterLessModule<Tensor, Tensor>
        {
            internal AdaptiveMaxPool1d(long output_size) : base(nameof(AdaptiveMaxPool1d))
            {
                this.output_size = output_size;
            }

            public (Tensor Values, Tensor Indices) forward_with_indices(Tensor input)
            {
                return torch.nn.functional.adaptive_max_pool1d_with_indices(input, this.output_size);
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.adaptive_max_pool1d(input, this.output_size);
            }

            public long output_size { get; set; }
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
            /// <param name="output_size">The target output size H.</param>
            /// <returns></returns>
            public static AdaptiveMaxPool1d AdaptiveMaxPool1d(long output_size)
            {
                return new AdaptiveMaxPool1d(output_size);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 1D adaptive max pooling over an input signal composed of several input planes.
                /// The output size is H, for any input size.The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="input"></param>
                /// <param name="output_size">The target output size H.</param>
                /// <returns></returns>
                public static Tensor adaptive_max_pool1d(Tensor input, long output_size)
                {
                    var ret = adaptive_max_pool1d_with_indices(input, output_size);
                    ret.Indices.Dispose();
                    return ret.Values;
                }

                /// <summary>
                /// Applies a 1D adaptive max pooling over an input signal composed of several input planes.
                /// The output size is H, for any input size.The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="input"></param>
                /// <param name="output_size">The target output size H.</param>
                /// <returns></returns>
                public static (Tensor Values, Tensor Indices) adaptive_max_pool1d_with_indices(Tensor input, long output_size)
                {
                    var outputSizes = new long[] { output_size };
                    unsafe {
                        fixed (long* poutputSize = outputSizes) {
                            var resOutput = THSTensor_adaptive_max_pool1d(input.Handle, (IntPtr)poutputSize, outputSizes.Length, out var resIndices);
                            if (resOutput == IntPtr.Zero || resIndices == IntPtr.Zero) { torch.CheckForErrors(); }
                            return (new Tensor(resOutput), new Tensor(resIndices));
                        }
                    }
                }
            }
        }
    }
}
