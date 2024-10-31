// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Hardswish module.
        /// </summary>
        public sealed class Hardswish : ParameterLessModule<Tensor, Tensor>
        {
            public bool inplace { get; set;}

            internal Hardswish(bool inplace = false) : base(nameof(Hardswish))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.hardswish(tensor, this.inplace);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies the Hardswish function, element-wise, as described in the paper:
            /// `Searching for MobileNetV3 https://arxiv.org/abs/1905.02244`.
            /// </summary>
            /// <param name="inplace">Do the operation in-place</param>
            /// <returns></returns>
            public static Hardswish Hardswish(bool inplace = false)
            {
                return new Hardswish(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies the Hardswish function, element-wise, as described in the paper:
                /// `Searching for MobileNetV3 https://arxiv.org/abs/1905.02244`.
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="inplace">Do the operation in-place</param>
                /// <returns></returns>
                public static Tensor hardswish(Tensor input, bool inplace = false)
                {
                    return inplace ? input.hardswish_().alias() : input.hardswish();
                }
            }
        }
    }
}
