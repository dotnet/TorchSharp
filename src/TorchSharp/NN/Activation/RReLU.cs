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
        /// This class is used to represent a RReLU module.
        /// </summary>
        public sealed class RReLU : ParameterLessModule<Tensor, Tensor>
        {
            internal RReLU(double lower, double upper, bool inplace) : base(nameof(RReLU))
            {
                this.lower = lower;
                this.upper = upper;
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.rrelu(tensor, lower, upper, inplace);
            }
            public double lower {get; set;}
            public double upper {get; set;}
            public bool inplace {get; set;}
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Randomized Rectified Linear Unit
            /// </summary>
            /// <param name="lower">Lower bound of the uniform distribution. Default: 1/8</param>
            /// <param name="upper">Upper bound of the uniform distribution. Default: 1/3</param>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            public static RReLU RReLU(double lower = one_eighth, double upper = one_third, bool inplace = false)
            {
                return new RReLU(lower, upper, inplace);
            }

            private const double one_eighth = 1.0 / 8.0;
            private const double one_third = 1.0 / 3.0;

            public static partial class functional
            {
                /// <summary>
                /// Randomized Rectified Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="lower">Lower bound of the uniform distribution. Default: 1/8</param>
                /// <param name="upper">Upper bound of the uniform distribution. Default: 1/3</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor rrelu(Tensor x, double lower = one_eighth, double upper = one_third, bool inplace = false)
                {
                    return inplace ? x.rrelu_(lower, upper).alias() : x.rrelu(lower, upper);
                }
            }
        }
    }
}
