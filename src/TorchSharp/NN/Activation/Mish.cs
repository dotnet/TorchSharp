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
        /// This class is used to represent a Mish module.
        /// </summary>
        public sealed class Mish : ParamLessModule<Tensor, Tensor>
        {
            internal Mish(bool inplace) : base(nameof(Mish))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.mish(tensor, inplace);
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
            /// A Self Regularized Non-Monotonic Neural Activation Function.
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            public static Mish Mish(bool inplace = false)
            {
                return new Mish(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// A Self Regularized Non-Monotonic Neural Activation Function.
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                public static Tensor mish(Tensor x, bool inplace = false)
                {
                    using var t1 = softplus(x);
                    using var t2 = t1.tanh();
                    return inplace ? x.mul_(t2).alias() : x.mul(t2);
                }
            }
        }
    }
}
