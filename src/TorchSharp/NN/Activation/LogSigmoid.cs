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
        /// This class is used to represent a LogSigmoid module.
        /// </summary>
        public sealed class LogSigmoid : torch.nn.Module<Tensor, Tensor>
        {
            internal LogSigmoid() : base(nameof(LogSigmoid)) { }

            public override Tensor forward(Tensor tensor)
            {
                return tensor.log_sigmoid();
            }

            public override string GetName()
            {
                return typeof(LogSigmoid).Name;
            }

           // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;
        }
    }
    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// LogSigmoid activation
            /// </summary>
            /// <returns></returns>
            public static LogSigmoid LogSigmoid()
            {
                return new LogSigmoid();
            }

            public static partial class functional
            {
                /// <summary>
                /// LogSigmoid activation
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                public static Tensor logsigmoid(Tensor x)
                {
                    return x.log_sigmoid();
                }
            }
        }
    }
}
