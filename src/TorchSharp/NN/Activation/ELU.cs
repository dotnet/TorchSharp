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
        /// This class is used to represent a ELU module.
        /// </summary>
        public sealed class ELU : ParameterLessModule<Tensor, Tensor>
        {
            internal ELU(double alpha, bool inplace) : base(nameof(ELU))
            {
                this.alpha = alpha;
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.elu(tensor, alpha, inplace);
            }

            public double alpha {get; set;}

            public bool inplace {get; set;}
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Exponential Linear Unit
            /// </summary>
            /// <param name="alpha">The α value for the ELU formulation. Default: 1.0</param>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            public static ELU ELU(double alpha = 1.0, bool inplace = false)
            {
                return new ELU(alpha, inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Exponential Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="alpha">The α value for the ELU formulation. Default: 1.0</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor elu(Tensor x, double alpha, bool inplace = false)
                {
                    return inplace ? x.elu_(alpha).alias() : x.elu(alpha);
                }
            }
        }
    }
}
