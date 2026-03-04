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
        /// This class is used to represent a GELU module.
        /// </summary>
        public sealed class GELU : ParameterLessModule<Tensor, Tensor>
        {
            /// <summary>
            /// Specifies the approximation method for GELU.
            /// </summary>
            public enum Approximate
            {
                /// <summary>
                /// Exact GELU computation.
                /// </summary>
                none,
                /// <summary>
                /// Tanh-based approximation.
                /// </summary>
                tanh
            }

            internal GELU(bool inplace, Approximate approximate = Approximate.none) : base(nameof(GELU))
            {
                this.inplace = inplace;
                this.approximate = approximate;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.gelu(tensor, approximate, inplace);
            }

            public bool inplace {get; set; }

            public Approximate approximate { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Gaussian Error Linear Units
            /// </summary>
            public static GELU GELU()
            {
                return new GELU(false);
            }

            /// <summary>
            /// Gaussian Error Linear Units
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            public static GELU GELU(bool inplace)
            {
                return new GELU(inplace);
            }

            /// <summary>
            /// Gaussian Error Linear Units
            /// </summary>
            /// <param name="approximate">The approximation method to use. Default: none</param>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            public static GELU GELU(GELU.Approximate approximate, bool inplace = false)
            {
                return new GELU(inplace, approximate);
            }

            public static partial class functional
            {
                /// <summary>
                /// Gaussian Error Linear Units
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                public static Tensor gelu(Tensor x, bool inplace)
                {
                    return inplace ? x.gelu_().alias() : x.gelu();
                }

                /// <summary>
                /// Gaussian Error Linear Units
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="approximate">The approximation method to use.</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                public static Tensor gelu(Tensor x, GELU.Approximate approximate, bool inplace = false)
                {
                    return inplace ? x.gelu_(approximate).alias() : x.gelu(approximate);
                }

                /// <summary>
                /// Gaussian Error Linear Units
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <remarks>The defaulting of 'inplace' to 'false' is implemented as an overload to avoid a breaking change.</remarks>
                public static Tensor gelu(Tensor x)
                {
                    return gelu(x,false);
                }
            }
        }
    }
}
