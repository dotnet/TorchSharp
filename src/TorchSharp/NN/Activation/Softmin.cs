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
        /// This class is used to represent a Softmin module.
        /// </summary>
        public sealed class Softmin : ParameterLessModule<Tensor, Tensor>
        {
            internal Softmin(long dim) : base(nameof(Softmin))
            {
                this.dim = dim;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.softmin(tensor, dim);
            }

            public long dim {get; set;}
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Softmin
            /// </summary>
            /// <param name="dim">A dimension along which Softmin will be computed (so every slice along dim will sum to 1)</param>
            /// <returns></returns>
            public static Softmin Softmin(long dim)
            {
                return new Softmin(dim);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softmin
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="dim">A dimension along which Softmin will be computed (so every slice along dim will sum to 1)</param>
                /// <returns></returns>
                public static Tensor softmin(Tensor x, long dim)
                {
                    using var minus_x = -x;
                    return softmax(minus_x, dim);
                }
            }
        }
    }
}
