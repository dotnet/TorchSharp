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
        /// This class is used to represent a AvgPool1D module.
        /// </summary>
        public sealed class AvgPool1d : torch.nn.Module<Tensor, Tensor>
        {
            internal AvgPool1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            protected override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AvgPool1d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            /// Applies a 1D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the window</param>
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <returns></returns>
            public static AvgPool1d AvgPool1d(long kernelSize, long? stride = null)
            {
                return stride.HasValue ?
                    AvgPool1d(new long[] { kernelSize }, new long[] { stride.Value }) :
                    AvgPool1d(new long[] { kernelSize }, null);
            }

            /// <summary>
            /// Applies a 1D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the window</param>
            /// <param name="strides">The stride of the window. Default value is kernel_size</param>
            /// <returns></returns>
            private static AvgPool1d AvgPool1d(long[] kernelSize, long[] strides = null)
            {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides) {
                        var handle = THSNN_AvgPool1d_ctor((IntPtr)pkernelSize, (IntPtr)pstrides, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new AvgPool1d(handle, boxedHandle);
                    }
                }
            }
        }
    }
}
