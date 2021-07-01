// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;

namespace TorchSharp
{
    using impl;
    using static torch;

    namespace impl
    {
        /// <summary>
        /// This class is used to represent a AdaptiveAvgPool2D module.
        /// </summary>
        public class AdaptiveAvgPool2d : torch.nn.Module
        {
            internal AdaptiveAvgPool2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_AdaptiveAvgPool2d_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AdaptiveAvgPool2d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            extern static IntPtr THSNN_AdaptiveAvgPool2d_ctor(IntPtr psizes, int length, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size (H,W) of the image of the form H x W.</param>
            /// <returns></returns>
            static public AdaptiveAvgPool2d AdaptiveAvgPool2d(long[] outputSize)
            {
                unsafe {
                    fixed (long* pkernelSize = outputSize) {
                        var handle = THSNN_AdaptiveAvgPool2d_ctor((IntPtr)pkernelSize, outputSize.Length, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new AdaptiveAvgPool2d(handle, boxedHandle);
                    }
                }
            }
        }

        public static partial class functional
        {
            /// <summary>
            /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="x">The input signal tensor.</param>
            /// <param name="outputSize">The target output size (H,W) of the image of the form H x W.</param>
            /// <returns></returns>
            static public Tensor AdaptiveAvgPool2d(Tensor x, long[] outputSize)
            {
                using (var d = nn.AdaptiveAvgPool2d(outputSize)) {
                    return d.forward(x);
                }
            }
        }
    }
}
