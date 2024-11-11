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
        /// This class is used to represent a SiLU module.
        /// </summary>
        public sealed class SiLU : ParameterLessModule<Tensor, Tensor>
        {
            internal SiLU(bool inplace) : base(nameof(SiLU))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.silu(tensor, inplace);
            }

            public override string GetName()
            {
                return typeof(SiLU).Name;
            }

            public bool inplace {get; set; }
        }
    }
    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Sigmoid-Weighted Linear Unit
            /// </summary>
            public static SiLU SiLU()
            {
                return new SiLU(false);
            }

            /// <summary>
            /// Sigmoid-Weighted Linear Unit
            /// </summary>
            public static SiLU SiLU(bool inplace)
            {
                return new SiLU(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Sigmoid-Weighted Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: false</param>
                public static Tensor silu(Tensor x, bool inplace = false)
                {
                    return inplace ? x.silu_().alias() : x.silu();
                }

                [Obsolete("Incorrect name capitalization", false)]
                public static Tensor SiLU(Tensor x, bool inplace = false) => silu(x, inplace);
            }
        }
    }
}
