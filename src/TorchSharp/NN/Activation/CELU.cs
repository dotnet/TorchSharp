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
        /// This class is used to represent a CELU module.
        /// </summary>
        public sealed class CELU : ParameterLessModule<Tensor, Tensor>
        {
            internal CELU(double alpha, bool inplace) : base(nameof(CELU))
            {
                this.alpha = alpha;
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.celu(tensor, alpha, inplace);
            }

            public double alpha {get; set;}
            public bool inplace {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Continuously Differentiable Exponential Linear Unit
            /// </summary>
            /// <param name="alpha">The α value for the CELU formulation. Default: 1.0</param>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            public static CELU CELU(double alpha = 1.0, bool inplace = false)
            {
                return new CELU(alpha, inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Continuously Differentiable Exponential Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="alpha">The α value for the CELU formulation. Default: 1.0</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor celu(Tensor x, double alpha, bool inplace = false)
                {
                    return inplace ? x.celu_(alpha).alias() : x.celu(alpha);
                }
            }
        }
    }
}
