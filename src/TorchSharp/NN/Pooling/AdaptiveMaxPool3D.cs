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
        /// This class is used to represent a AdaptiveMaxPool3D module.
        /// </summary>
        public sealed class AdaptiveMaxPool3d : ParameterLessModule<Tensor, Tensor>
        {
            internal AdaptiveMaxPool3d(long[] output_size) : base(nameof(AdaptiveMaxPool3d))
            {
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.adaptive_max_pool3d(input, output_size);
            }

            public (Tensor output, Tensor indices) forward_with_indices(Tensor input)
            {
                return torch.nn.functional.adaptive_max_pool3d_with_indices(input, output_size);
            }


            public long[] output_size { get; set; }
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
            /// <param name="output_size">The target output size of the image of the form D x H x W.
            /// Can be a tuple (D, H, W) or a single D for a cube D x D x D. D, H and W can be either a int, or null which means the size will be the same as that of the input.</param>
            /// <returns></returns>
            public static AdaptiveMaxPool3d AdaptiveMaxPool3d(long[] output_size)
            {
                return new AdaptiveMaxPool3d(output_size);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 3D adaptive max pooling over an input signal composed of several input planes.
                /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="output_size">The target output size of the image of the form D x H x W.
                /// Can be a tuple (D, H, W) or a single D for a cube D x D x D. D, H and W can be either a int, or null which means the size will be the same as that of the input.</param>
                /// <returns></returns>
                public static Tensor adaptive_max_pool3d(Tensor input, long[] output_size)
                {
                    var ret = adaptive_max_pool3d_with_indices(input, output_size);
                    ret.Indices.Dispose();
                    return ret.Values;
                }

                /// <summary>
                /// Applies a 3D adaptive max pooling over an input signal composed of several input planes.
                /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="output_size">The target output size of the image of the form D x H x W.
                /// Can be a tuple (D, H, W) or a single D for a cube D x D x D. D, H and W can be either a int, or null which means the size will be the same as that of the input.</param>
                /// <returns></returns>
                public static (Tensor Values, Tensor Indices) adaptive_max_pool3d_with_indices(Tensor input, long[] output_size)
                {
                    unsafe {
                        fixed (long* poutputSize = output_size) {
                            var resOutput = THSTensor_adaptive_max_pool1d(input.Handle, (IntPtr)poutputSize, output_size.Length, out var resIndices);
                            if (resOutput == IntPtr.Zero || resIndices == IntPtr.Zero) { torch.CheckForErrors(); }
                            return (new Tensor(resOutput), new Tensor(resIndices));
                        }
                    }
                }
            }
        }
    }
}
