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
        public sealed class ConstantPad1d : torch.nn.Module<Tensor, Tensor>
        {
            internal ConstantPad1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ConstantPad1d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;
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
                var handle = THSNN_ConstantPad1d_ctor(value, padding, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ConstantPad1d(handle, boxedHandle);
            }

            /// <summary>
            /// Pads the input tensor boundaries with a constant value.
            /// </summary>
            /// <param name="padding">The size of the padding: (padding_right, padding_left).</param>
            /// <param name="value"></param>
            /// <returns></returns>
            public static ConstantPad1d ConstantPad1d((long, long) padding, double value)
            {
                var handle = THSNN_ConstantPad1d_ctor_tuple(value, padding.Item1, padding.Item2, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ConstantPad1d(handle, boxedHandle);
            }
        }
    }
}
