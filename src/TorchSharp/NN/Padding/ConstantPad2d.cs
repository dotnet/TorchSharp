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
        /// This class is used to represent a ConstantPad2d module.
        /// </summary>
        public sealed class ConstantPad2d : torch.nn.Module<Tensor, Tensor>
        {
            internal ConstantPad2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ConstantPad2d_forward(handle, tensor.Handle);
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
            /// Pads the input tensor boundaries with a constant value.
            /// </summary>
            /// <param name="padding">The size of the padding.</param>
            /// <param name="value"></param>
            /// <returns></returns>
            public static ConstantPad2d ConstantPad2d(long padding, double value)
            {
                var handle = THSNN_ConstantPad2d_ctor(value, padding, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ConstantPad2d(handle, boxedHandle);
            }

            /// <summary>
            /// Pads the input tensor boundaries with a constant value.
            /// </summary>
            /// <param name="padding">The size of the padding:  (padding_left, padding_right, padding_top, padding_bottom)</param>
            /// <param name="value"></param>
            /// <returns></returns>
            public static ConstantPad2d ConstantPad2d((long, long, long, long) padding, double value)
            {
                var handle = THSNN_ConstantPad2d_ctor_tuple(value, padding.Item1, padding.Item2, padding.Item3, padding.Item4, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ConstantPad2d(handle, boxedHandle);
            }
        }
    }
}
