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
        public sealed class Hardsigmoid : ParamLessModule<Tensor, Tensor>
        {
            internal Hardsigmoid(bool inplace) : base(nameof(Hardsigmoid))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.hardsigmoid(tensor, inplace);
            }

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;

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
