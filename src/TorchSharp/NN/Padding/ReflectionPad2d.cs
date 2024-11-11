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
        /// This class is used to represent a ReflectionPad2d module.
        /// </summary>
        public sealed class ReflectionPad2d : PadBase
        {
            internal ReflectionPad2d(params long[] padding) : base(nameof(ReflectionPad2d), PaddingModes.Reflect, 0, padding) { }        
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Pads the input tensor using the reflection of the input boundary.
            /// </summary>
            /// <param name="padding">The size of the padding.</param>
            /// <returns></returns>
            public static ReflectionPad2d ReflectionPad2d(long padding)
            {
                return new ReflectionPad2d(padding, padding, padding, padding);
            }

            /// <summary>
            /// Pads the input tensor using reflection of the input boundary.
            /// </summary>
            /// <param name="padding">The size of the padding: (padding_left, padding_right, padding_top, padding_bottom).</param>
            /// <returns></returns>
            public static ReflectionPad2d ReflectionPad2d((long, long, long, long) padding)
            {
                return new ReflectionPad2d(padding.Item1, padding.Item2, padding.Item3, padding.Item4);
            }
        }
    }
}
