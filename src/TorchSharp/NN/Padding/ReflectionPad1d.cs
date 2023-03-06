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
        /// This class is used to represent a ReflectionPad1d module.
        /// </summary>
        public sealed class ReflectionPad1d : torch.nn.Module<Tensor, Tensor>
        {
            internal ReflectionPad1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ReflectionPad1d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
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
                var handle = THSNN_ReflectionPad1d_ctor(padding, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ReflectionPad1d(handle, boxedHandle);
            }

            /// <summary>
            /// Pads the input tensor using the reflection of the input boundary.
            /// </summary>
            /// <param name="padding">The size of the padding: (padding_right, padding_left).</param>
            /// <returns></returns>
            public static ReflectionPad1d ReflectionPad1d((long, long) padding)
            {
                var handle = THSNN_ReflectionPad1d_ctor_tuple(padding.Item1, padding.Item2, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ReflectionPad1d(handle, boxedHandle);
            }
        }
    }
}
