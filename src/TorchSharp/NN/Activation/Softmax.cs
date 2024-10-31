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
        /// This class is used to represent a Softmax module.
        /// </summary>
        public sealed class Softmax : ParameterLessModule<Tensor, Tensor>
        {
            internal Softmax(long dim) : base(nameof(Softmax))
            {
                this.dim = dim;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.softmax(tensor, dim);
            }

            public long dim {get; set;}
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Softmax
            /// </summary>
            /// <param name="dim">A dimension along which Softmax will be computed (so every slice along dim will sum to 1)</param>
            /// <returns></returns>
            public static Softmax Softmax(long dim)
            {
                return new Softmax(dim);
            }

            public static partial class functional
            {
                /// <summary>
                /// Computes the softmax function for the input tensor.
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="dim">A dimension along which softmax will be computed.</param>
                /// <param name="dtype">The desired data type of returned tensor.</param>
                public static Tensor softmax(Tensor input, long dim, ScalarType? dtype = null) =>
                    torch.special.softmax(input, dim, dtype);
            }
        }
    }
}
