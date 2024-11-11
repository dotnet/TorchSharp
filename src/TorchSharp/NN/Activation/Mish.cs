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
        /// This class is used to represent a Mish module.
        /// </summary>
        public sealed class Mish : ParameterLessModule<Tensor, Tensor>
        {
            internal Mish(bool inplace) : base(nameof(Mish))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.mish(tensor, inplace);
            }

            public bool inplace {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// A Self Regularized Non-Monotonic Neural Activation Function.
            /// </summary>
            public static Mish Mish()
            {
                return new Mish(false);
            }

            /// <summary>
            /// A Self Regularized Non-Monotonic Neural Activation Function.
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            public static Mish Mish(bool inplace)
            {
                return new Mish(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// A Self Regularized Non-Monotonic Neural Activation Function.
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                public static Tensor mish(Tensor x, bool inplace = false)
                {
                    using var t1 = softplus(x);
                    using var t2 = t1.tanh();
                    return inplace ? x.mul_(t2).alias() : x.mul(t2);
                }

                /// <summary>
                /// A Self Regularized Non-Monotonic Neural Activation Function.
                /// </summary>
                /// <param name="x">The input tensor</param>
                [Obsolete("Not using the PyTorch naming convention.",false)]
                public static Tensor Mish(Tensor x) => mish(x, false);
            }
        }
    }
}
