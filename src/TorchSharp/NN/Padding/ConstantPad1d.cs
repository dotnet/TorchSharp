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
        /// This class is used to represent a ConstantPad1d module.
        /// </summary>
        public sealed class ConstantPad1d : PadBase
        {
            internal ConstantPad1d(double value, params long[] padding) : base(nameof(ConstantPad1d), PaddingModes.Constant, value, padding) { }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Pads the input tensor boundaries with a constant value.
            /// </summary>
            /// <param name="padding">The size of the padding.</param>
            /// <param name="value"></param>
            /// <returns></returns>
            public static ConstantPad1d ConstantPad1d(long padding, double value)
            {
                return new ConstantPad1d(value, padding, padding);
            }

            /// <summary>
            /// Pads the input tensor boundaries with a constant value.
            /// </summary>
            /// <param name="padding">The size of the padding: (padding_right, padding_left).</param>
            /// <param name="value"></param>
            /// <returns></returns>
            public static ConstantPad1d ConstantPad1d((long, long) padding, double value)
            {
                return new ConstantPad1d(value, padding.Item1, padding.Item2);
            }
        }
    }
}
