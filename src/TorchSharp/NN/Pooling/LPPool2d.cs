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
        /// This class is used to represent a LPPool2D module.
        /// </summary>
        public sealed class LPPool2d : torch.nn.Module<Tensor, Tensor>
        {
            internal LPPool2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_LPPool2d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            /// Applies a 2D power-average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="norm_type">The LP norm (exponent)</param>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="strides">The stride of the window. Default value is kernel_size</param>
            /// <param name="ceil_mode">Use ceil instead of floor to compute the output shape</param>
            /// <returns></returns>
            public static LPPool2d LPPool2d(double norm_type, long[] kernel_size, long[] strides = null, bool ceil_mode = false)
            {
                unsafe {
                    fixed (long* pkernelSize = kernel_size, pstrides = strides) {
                        var handle = THSNN_LPPool2d_ctor(norm_type, (IntPtr)pkernelSize, kernel_size.Length, (IntPtr)pstrides, (strides == null ? 0 : strides.Length), ceil_mode, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new LPPool2d(handle, boxedHandle);
                    }
                }
            }

            /// <summary>
            /// Applies a 2D power-average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="norm_type">The LP norm (exponent)</param>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window.</param>
            /// <param name="ceil_mode">Use ceil instead of floor to compute the output shape</param>
            /// <returns></returns>
            public static LPPool2d LPPool2d(double norm_type, long kernel_size, long? stride = null, bool ceil_mode = false)
            {
                return stride.HasValue ?
                    LPPool2d(norm_type, new long[] { kernel_size, kernel_size }, new long[] { stride.Value, stride.Value }, ceil_mode) :
                    LPPool2d(norm_type, new long[] { kernel_size, kernel_size }, null, ceil_mode);
            }
        }
    }
}
