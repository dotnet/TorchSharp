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
        public class PadBase : ParamLessModule<Tensor, Tensor>
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

            private PaddingModes _paddingMode;
            private long[] _padding;
            private double _value = 0.0;
        }
    }
}
