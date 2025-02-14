// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Hardsigmoid module.
        /// </summary>
        public sealed class Hardsigmoid : ParameterLessModule<Tensor, Tensor>
        {
            internal Hardsigmoid(bool inplace) : base(nameof(Hardsigmoid))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.hardsigmoid(tensor, inplace);
            }

            public bool inplace {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Hardsigmoid
            /// </summary>
            /// <param name="inplace">Do the operation in-place</param>
            /// <returns></returns>
            public static Hardsigmoid Hardsigmoid(bool inplace = false)
            {
                return new Hardsigmoid(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Hardsigmoid
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="inplace">Do the operation in-place</param>
                /// <returns></returns>
                public static Tensor hardsigmoid(Tensor input, bool inplace = false)
                {
                    return inplace ? input.hardsigmoid_().alias() : input.hardsigmoid();
                }
            }
        }
    }
}
