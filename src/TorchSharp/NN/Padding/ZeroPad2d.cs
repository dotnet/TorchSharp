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
        /// This class is used to represent a ZeroPad2d module.
        /// </summary>
        public sealed class ZeroPad2d : PadBase
        {
            internal ZeroPad2d(params long[] padding) : base(nameof(ZeroPad2d), PaddingModes.Zeros, 0, padding) { }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Pads the input tensor boundaries with zero.
            /// </summary>
            /// <param name="padding">The size of the padding.</param>
            /// <returns></returns>
            public static ZeroPad2d ZeroPad2d(long padding)
            {
                return new ZeroPad2d(padding, padding, padding, padding);
            }

            /// <summary>
            /// Pads the input tensor boundaring with zero
            /// </summary>
            /// <param name="padding">The size of the padding: (padding_left, padding_right, padding_top, padding_bottom).</param>
            /// <returns></returns>
            public static ZeroPad2d ZeroPad2d((long, long, long, long) padding)
            {
                return new ZeroPad2d(padding.Item1, padding.Item2, padding.Item3, padding.Item4);
            }
        }
    }
}
