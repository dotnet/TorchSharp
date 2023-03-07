// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent the base of all padding-related modules.
        /// </summary>
        public class PadBase : nn.Module<Tensor, Tensor>
        {
            protected PadBase(string name, PaddingModes mode, double value, params long[] padding) : base(name)
            {
                _value = value;
                _padding = padding;
                _paddingMode = mode;
            }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="input">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor input)
            {
                return nn.functional.pad(input, _padding, _paddingMode, _value);
            }

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype) => this;

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex = -1) => this;

            protected internal override nn.Module _to(ScalarType dtype) => this;

            private PaddingModes _paddingMode;
            private long[] _padding;
            private double _value = 0.0;
        }
    }
}
