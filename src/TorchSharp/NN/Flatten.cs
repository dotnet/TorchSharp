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
        /// This class is used to represent a dropout module for 2d/3d convolutational layers.
        /// </summary>
        public sealed class Flatten : torch.nn.Module<Tensor, Tensor>
        {
            internal Flatten(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Flatten_forward(handle, tensor.Handle);
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
            /// Flattens a contiguous range of dims into a tensor. For use with Sequential.
            /// </summary>
            /// <param name="startDim">First dim to flatten (default = 1).</param>
            /// <param name="endDim">Last dim to flatten (default = -1).</param>
            /// <returns></returns>
            public static Flatten Flatten(long startDim = 1, long endDim = -1)
            {
                var handle = THSNN_Flatten_ctor(startDim, endDim, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Flatten(handle, boxedHandle);
            }
        }
    }
}
