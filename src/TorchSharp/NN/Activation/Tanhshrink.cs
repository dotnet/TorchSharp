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
        /// This class is used to represent a Tanhshrink module.
        /// </summary>
        public sealed class Tanhshrink : ParameterLessModule<Tensor, Tensor>
        {
            internal Tanhshrink(bool inplace) : base(nameof(Tanhshrink))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.tanhshrink(tensor, inplace);
            }

            public bool inplace {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Tanhshrink
            /// </summary>
            public static Tanhshrink Tanhshrink()
            {
                return new Tanhshrink(false);
            }

            /// <summary>
            /// Tanhshrink
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            public static Tanhshrink Tanhshrink(bool inplace = false)
            {
                return new Tanhshrink(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Tanhshrink
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                public static Tensor tanhshrink(Tensor x, bool inplace = false)
                {
                    using var tanh_x = x.tanh();
                    return inplace ? x.sub_(tanh_x).alias() : x.sub(tanh_x);
                }

                /// <summary>
                /// Tanhshrink
                /// </summary>
                /// <param name="x">The input tensor</param>
                [Obsolete("Not using the PyTorch naming convention.",false)]
                public static Tensor Tanhshrink(Tensor x) => tanhshrink(x, false);
            }
        }
    }
}
