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
        /// This class is used to represent a ReplicationPad1d module.
        /// </summary>
        public sealed class ReplicationPad1d : PadBase
        {
            internal ReplicationPad1d(params long[] padding) : base(nameof(ReplicationPad1d), PaddingModes.Replicate, 0, padding) { }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Pads the input tensor using replication of the input boundary.
            /// </summary>
            /// <param name="padding">The size of the padding.</param>
            /// <returns></returns>
            public static ReplicationPad1d ReplicationPad1d(long padding)
            {
                return new ReplicationPad1d(padding, padding);
            }

            /// <summary>
            /// Pads the input tensor using replication of the input boundary.
            /// </summary>
            /// <param name="padding">The size of the padding: (padding_right, padding_left).</param>
            /// <returns></returns>
            public static ReplicationPad1d ReplicationPad1d((long, long) padding)
            {
                return new ReplicationPad1d(padding.Item1, padding.Item2);
            }
        }
    }
}
