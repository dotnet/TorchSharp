// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class Identity : ParamLessModule<Tensor, Tensor>
        {
            internal Identity() : base(nameof(Identity)) { }

            public override Tensor forward(Tensor tensor)
            {
                return tensor.alias();
            }

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex = -1) => this;
            protected internal override nn.Module _to(ScalarType dtype) => this;
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// A placeholder identity operator.
            /// </summary>
            /// <returns>The same tensor as is input.</returns>
            public static Identity Identity()
            {
                return new Identity();
            }
        }
    }
}
