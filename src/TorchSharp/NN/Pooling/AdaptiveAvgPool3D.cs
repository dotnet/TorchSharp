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
        /// This class is used to represent a AdaptiveAvgPool3D module.
        /// </summary>
        public class AdaptiveAvgPool3d : torch.nn.Module
        {
            internal AdaptiveAvgPool3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_AdaptiveAvgPool3d_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AdaptiveAvgPool3d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            extern static IntPtr THSNN_AdaptiveAvgPool3d_ctor(IntPtr psizes, int length, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size of the image of the form D x H x W.</param>
            /// <returns></returns>
            static public unsafe AdaptiveAvgPool3d AdaptiveAvgPool3d(long[] outputSize)
            {
                fixed (long* pkernelSize = outputSize) {
                    var handle = THSNN_AdaptiveAvgPool3d_ctor((IntPtr)pkernelSize, outputSize.Length, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new AdaptiveAvgPool3d(handle, boxedHandle);
                }
            }

            /// <summary>
            /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size (D,H,W) of the image of the form D x H x W.</param>
            /// <returns></returns>
            static public unsafe AdaptiveAvgPool3d AdaptiveAvgPool3d((long, long, long) outputSize)
            {
                long* pkernelSize = stackalloc long[3] { outputSize.Item1, outputSize.Item2, outputSize.Item3 };

                var handle = THSNN_AdaptiveAvgPool3d_ctor((IntPtr)pkernelSize, 3, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdaptiveAvgPool3d(handle, boxedHandle);
            }

            /// <summary>
            /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
            /// The output is of size D x H x W, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">The target output size (D,H,W) of the image of the form H x W.</param>
            /// <returns></returns>
            static public unsafe AdaptiveAvgPool3d AdaptiveAvgPool3d(long outputSize)
            {
                long* pkernelSize = stackalloc long[3] { outputSize, outputSize, outputSize};
                var handle = THSNN_AdaptiveAvgPool3d_ctor((IntPtr)pkernelSize, 3, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdaptiveAvgPool3d(handle, boxedHandle);
        }
        }
    }
}
