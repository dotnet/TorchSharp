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
        /// This class is used to represent a ConstantPad3d module.
        /// </summary>
        public sealed class ConstantPad3d : PadBase
        {
            internal ConstantPad3d(double value, params long[] padding) : base(nameof(ConstantPad3d), PaddingModes.Constant, value, padding) { }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Pads the input tensor boundaries with a constant value.
            /// </summary>
            /// <param name="value"></param>
            /// <param name="padding">The size of the padding.</param>
            /// <returns></returns>
            public static ConstantPad3d ConstantPad3d(long padding, double value)
            {
                return new ConstantPad3d(value, padding, padding, padding, padding, padding, padding);
            }

            /// <summary>
            /// Pads the input tensor boundaries with a constant value.
            /// </summary>
            /// <param name="value"></param>
            /// <param name="padding">The size of the padding (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)</param>
            /// <returns></returns>
            public static ConstantPad3d ConstantPad3d((long, long, long, long, long, long) padding, double value)
            {
                return new ConstantPad3d(value, padding.Item1, padding.Item2, padding.Item3, padding.Item4, padding.Item5, padding.Item6);
            }
        }
    }
}
