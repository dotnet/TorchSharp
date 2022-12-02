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
        /// This class is used to represent a ConstantPad3d module.
        /// </summary>
        public sealed class ConstantPad3d : torch.nn.Module<Tensor, Tensor>
        {
            internal ConstantPad3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            protected override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ConstantPad3d_forward(handle, tensor.Handle);
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
            /// Pads the input tensor using replication of the input boundary.
            /// </summary>
            /// <param name="value"></param>
            /// <param name="padding">The size of the padding.</param>
            /// <returns></returns>
            public static ConstantPad3d ConstantPad3d(long padding, double value)
            {
                var handle = THSNN_ConstantPad3d_ctor(value, padding, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ConstantPad3d(handle, boxedHandle);
            }
        }
    }
}
