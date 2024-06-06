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
        public sealed class Softsign : ParamLessModule<Tensor, Tensor>
        {
            internal Softsign(bool inplace) : base(nameof(Softsign))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.softsign(tensor, inplace);
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
            /// Softsign
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            public static Softsign Softsign(bool inplace = false)
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
            }
        }
    }
}
