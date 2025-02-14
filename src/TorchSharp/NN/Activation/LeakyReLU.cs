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
        /// This class is used to represent a LeakyReLU module.
        /// </summary>
        public sealed class LeakyReLU : ParameterLessModule<Tensor, Tensor>
        {
            internal LeakyReLU(double negative_slope, bool inplace) : base(nameof(LeakyReLU))
            {
                this.inplace = inplace;
                this.negative_slope = negative_slope;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.leaky_relu(tensor, negative_slope, inplace);
            }

            public bool inplace {get; set; }
            public double negative_slope {get; set;}
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Continuously Differentiable Exponential Linear Unit
            /// </summary>
            /// <param name="negative_slope">The α value for the LeakyReLU formulation.</param>
            /// <param name="inplace">Do the operation in-place.</param>
            /// <returns></returns>
            public static LeakyReLU LeakyReLU(double negative_slope = 0.01, bool inplace = false)
            {
                return new LeakyReLU(negative_slope, inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Continuously Differentiable Exponential Linear Unit
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="negative_slope">The α value for the LeakyReLU formulation. Default: 1.0</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor leaky_relu(Tensor input, double negative_slope = 0.01, bool inplace = false)
                {
                    return inplace ? input.leaky_relu_(negative_slope).alias() : input.leaky_relu(negative_slope);
                }
            }
        }
    }
}
