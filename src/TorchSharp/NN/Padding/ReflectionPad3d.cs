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
        /// This class is used to represent a ReflectionPad3d module.
        /// </summary>
        public sealed class ReflectionPad3d : torch.nn.Module<Tensor, Tensor>
        {
            internal ReflectionPad3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ReflectionPad3d_forward(handle, tensor.Handle);
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
            public static ReflectionPad3d ReflectionPad3d(long padding)
            {
                var handle = THSNN_ReflectionPad3d_ctor(padding, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ReflectionPad3d(handle, boxedHandle);
            }

            /// <summary>
            /// Pads the input tensor using reflection of the input boundary.
            /// </summary>
            /// <param name="padding">The size of the padding: (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back).</param>
            /// <returns></returns>
            public static ReflectionPad3d ReflectionPad3d((long, long, long, long, long, long) padding)
            {
                var handle = THSNN_ReflectionPad3d_ctor_tuple(padding.Item1, padding.Item2, padding.Item3, padding.Item4, padding.Item5, padding.Item6, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ReflectionPad3d(handle, boxedHandle);
            }
        }
    }
}
