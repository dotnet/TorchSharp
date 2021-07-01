// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    using impl;

    namespace impl
    {
        /// <summary>
        /// This class is used to represent a AvgPool2D module.
        /// </summary>
        public class AvgPool2d : torch.nn.Module
        {
            internal AvgPool2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_AvgPool2d_forward(IntPtr module, IntPtr tensor);

            public override TorchTensor forward(TorchTensor tensor)
            {
                var res = THSNN_AvgPool2d_forward(handle.DangerousGetHandle(), tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_AvgPool2d_ctor(IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 2D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the window</param>
            /// <param name="strides">The stride of the window. Default value is kernel_size</param>
            /// <returns></returns>

            static public AvgPool2d AvgPool2d(long[] kernelSize, long[] strides = null)
            {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides) {
                        var handle = THSNN_AvgPool2d_ctor((IntPtr)pkernelSize, kernelSize.Length, (IntPtr)pstrides, (strides == null ? 0 : strides.Length), out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new AvgPool2d(handle, boxedHandle);
                    }
                }
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 2D average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="x">The input signal tensor</param>
                /// <param name="kernelSize">The size of the window</param>
                /// <param name="strides">The stride of the window. Default value is kernel_size</param>
                /// <returns></returns>
                static public TorchTensor avg_pool2d(TorchTensor x, long[] kernelSize, long[] strides = null)
                {
                    using (var d = nn.AvgPool2d(kernelSize, strides)) {
                        return d.forward(x);
                    }
                }
            }
        }
    }
}
