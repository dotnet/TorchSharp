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
        /// This class is used to represent a LPPool1D module.
        /// </summary>
        public class LPPool1d : torch.nn.Module
        {
            internal LPPool1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LPPool1d_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_LPPool1d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            extern static IntPtr THSNN_LPPool1d_ctor(double norm_type, IntPtr pkernelSize, IntPtr pstrides, bool ceil_mode, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 1D power-average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="norm_type">The LP norm (exponent)</param>
            /// <param name="kernelSize">The size of the window</param>
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <param name="ceil_mode">Use ceil instead of floor to compute the output shape</param>
            /// <returns></returns>
            static public LPPool1d LPPool1d(double norm_type, long kernelSize, long? stride = null, bool ceil_mode = false)
            {
                return stride.HasValue ?
                    LPPool1d(norm_type, new long[] { kernelSize }, new long[] { stride.Value }, ceil_mode) :
                    LPPool1d(norm_type, new long[] { kernelSize }, null);
            }

            /// <summary>
            /// Applies a 1D power-average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="norm_type">The LP norm (exponent)</param>
            /// <param name="kernelSize">The size of the window</param>
            /// <param name="strides">The stride of the window. Default value is kernel_size</param>
            /// <param name="ceil_mode">Use ceil instead of floor to compute the output shape</param>
            /// <returns></returns>
            static private LPPool1d LPPool1d(double norm_type, long[] kernelSize, long[] strides = null, bool ceil_mode = false)
            {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides) {
                        var handle = THSNN_LPPool1d_ctor(norm_type, (IntPtr)pkernelSize, (IntPtr)pstrides, ceil_mode, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new LPPool1d(handle, boxedHandle);
                    }
                }
            }
        }
    }
}
