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
        /// This class is used to represent a Softsign module.
        /// </summary>
        public sealed class Softsign : ParameterLessModule<Tensor, Tensor>
        {
            internal Softsign(bool inplace) : base(nameof(Softsign))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.softsign(tensor, inplace);
            }

            public bool inplace {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Softsign
            /// </summary>
            public static Softsign Softsign()
            {
                return new Softsign(false);
            }

            /// <summary>
            /// Softsign
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            public static Softsign Softsign(bool inplace)
            {
                return new Softsign(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softsign
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                public static Tensor softsign(Tensor x, bool inplace = false)
                {
                    using var abs = x.abs();
                    using var y = 1 + abs;
                    return inplace ? x.div_(y).alias() : x.div(y);
                }

                /// <summary>
                /// Softsign
                /// </summary>
                /// <param name="x">The input tensor</param>
                [Obsolete("Not using the PyTorch naming convention.",false)]
                public static Tensor Softsign(Tensor x) => softsign(x, false);
            }
        }
    }
}
