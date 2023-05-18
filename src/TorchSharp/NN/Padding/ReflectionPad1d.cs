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
        /// This class is used to represent a ReflectionPad1d module.
        /// </summary>
        public sealed class ReflectionPad1d : PadBase
        {
            internal ReflectionPad1d(params long[] padding) : base(nameof(ReflectionPad1d), PaddingModes.Reflect, 0, padding) { }
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
            public static ReflectionPad1d ReflectionPad1d(long padding)
            {
                return new ReflectionPad1d(padding, padding);
            }

            /// <summary>
            /// Pads the input tensor using the reflection of the input boundary.
            /// </summary>
            /// <param name="padding">The size of the padding: (padding_right, padding_left).</param>
            /// <returns></returns>
            public static ReflectionPad1d ReflectionPad1d((long, long) padding)
            {
                return new ReflectionPad1d(padding.Item1, padding.Item2);
            }
        }
    }
}
