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
            internal GELU(bool inplace) : base(nameof(GELU))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.gelu(tensor, inplace);
            }

            public bool inplace {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Gaussian Error Linear Units
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            public static GELU GELU(bool inplace = false)
            {
                return new GELU(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Gaussian Error Linear Units
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                public static Tensor gelu(Tensor x, bool inplace = false)
                {
                    return inplace ? x.gelu_().alias() : x.gelu();
                }
            }
        }
    }
}
