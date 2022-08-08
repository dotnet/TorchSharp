// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a LPPool2D module.
        /// </summary>
        public class LPPool2d : torch.nn.Module
        {
            internal LPPool2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LPPool2d_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_LPPool2d_forward(handle.DangerousGetHandle(), tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_LPPool2d_ctor(double norm_type, IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, [MarshalAs(UnmanagedType.U1)] bool ceil_mode, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 2D power-average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="norm_type">The LP norm (exponent)</param>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="strides">The stride of the window. Default value is kernel_size</param>
            /// <param name="ceil_mode">Use ceil instead of floor to compute the output shape</param>
            /// <returns></returns>
            static public LPPool2d LPPool2d(double norm_type, long[] kernel_size, long[] strides = null, bool ceil_mode = false)
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
            static public LPPool2d LPPool2d(double norm_type, long kernel_size, long? stride = null, bool ceil_mode = false)
            {
                return stride.HasValue ?
                    LPPool2d(norm_type, new long[] { kernel_size, kernel_size }, new long[] { stride.Value, stride.Value }, ceil_mode) :
                    LPPool2d(norm_type, new long[] { kernel_size, kernel_size }, null, ceil_mode);
            }
        }
    }
}
