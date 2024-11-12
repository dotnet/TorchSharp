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
        /// This class is used to represent a Sigmoid module.
        /// </summary>
        public sealed class Sigmoid : ParameterLessModule<Tensor, Tensor>
        {
            internal Sigmoid(bool inplace) : base(nameof(Sigmoid))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.sigmoid(tensor, inplace);
            }

            public bool inplace {get; set; }
        }
    }
    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Sigmoid activation
            /// </summary>
            /// <returns></returns>
            public static Sigmoid Sigmoid()
            {
                return new Sigmoid(false);
            }

            /// <summary>
            /// Sigmoid activation
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            public static Sigmoid Sigmoid(bool inplace)
            {
                return new Sigmoid(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Sigmoid activation
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor sigmoid(Tensor x, bool inplace)
                {
                    return inplace ? x.sigmoid_().alias() : x.sigmoid();
                }

                /// <summary>
                /// Gaussian Error Linear Units
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <remarks>The defaulting of 'inplace' to 'false' is implemented as an overload to avoid a breaking change.</remarks>
                public static Tensor sigmoid(Tensor x)
                {
                    return sigmoid(x,false);
                }
            }
        }
    }
}
