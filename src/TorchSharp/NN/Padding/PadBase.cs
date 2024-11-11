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
        /// This class is used to represent the base of all padding-related modules.
        /// </summary>
        public abstract class PadBase : ParameterLessModule<Tensor, Tensor>
        {
            protected PadBase(string name, PaddingModes mode, double value, params long[] padding) : base(name)
            {
                this.value = value;
                this.padding = padding;
                padding_mode = mode;
            }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="input">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor input)
            {
                return nn.functional.pad(input, padding, padding_mode, value);
            }

            private PaddingModes padding_mode { get; set; }
            public long[] padding { get; set; }
            public double value { get; set; }
        }
    }
}
