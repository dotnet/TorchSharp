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
        /// This class is used to represent a ReLU module.
        /// </summary>
        public sealed class ReLU : ParamLessModule<Tensor, Tensor>
        {
            internal ReLU(bool inplace) : base(nameof(ReLU))
            {
                this.inplace = inplace;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.relu(tensor, inplace);
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
            /// Rectified Linear Unit
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            public static ReLU ReLU(bool inplace = false)
            {
                return new ReLU(inplace);
            }

            public static partial class functional
            {
                /// <summary>
                /// Rectified Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor relu(Tensor x, bool inplace = false)
                {
                    return inplace ? x.relu_().alias() : x.relu();
                }
            }
        }
    }
}
