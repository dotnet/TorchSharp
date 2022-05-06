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
        /// This class is used to represent a AvgPool2D module.
        /// </summary>
        public class AvgPool2d : torch.nn.Module
        {
            internal AvgPool2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_AvgPool2d_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AvgPool2d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            extern static IntPtr THSNN_AvgPool2d_ctor(IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 2D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="strides">The stride of the window. Default value is kernel_size</param>
            /// <returns></returns>
            static public unsafe AvgPool2d AvgPool2d(long[] kernel_size, long[] strides = null)
            {
                fixed (long* pkernelSize = kernel_size, pstrides = strides) {
                    var handle = THSNN_AvgPool2d_ctor((IntPtr)pkernelSize, kernel_size.Length, (IntPtr)pstrides, (strides == null ? 0 : strides.Length), out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new AvgPool2d(handle, boxedHandle);
                }
            }

            /// <summary>
            /// Applies a 2D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window.</param>
            /// <returns></returns>
            static public unsafe AvgPool2d AvgPool2d((long,long) kernel_size, (long,long)? stride = null)
            {
                long svalue1 = (stride == null) ? kernel_size.Item1 : stride.Value.Item1;
                long svalue2 = (stride == null) ? kernel_size.Item2 : stride.Value.Item2;

                long* pkernelSize = stackalloc long[2] { kernel_size.Item1, kernel_size.Item2 };
                long* pstrides = stackalloc long[2] { svalue1, svalue2 };

                var handle = THSNN_AvgPool2d_ctor((IntPtr)pkernelSize, 2, (IntPtr)pstrides, 2, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AvgPool2d(handle, boxedHandle);
            }

            /// <summary>
            /// Applies a 2D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window.</param>
            /// <returns></returns>
            static public unsafe AvgPool2d AvgPool2d(long kernel_size, long? stride = null)
            {
                long svalue = (stride == null) ? kernel_size : stride.Value;

                long* pkernelSize = stackalloc long[2] { kernel_size, kernel_size };
                long* pstrides = stackalloc long[2] { svalue, svalue };

                var handle = THSNN_AvgPool2d_ctor((IntPtr)pkernelSize, 2, (IntPtr)pstrides, 2, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AvgPool2d(handle, boxedHandle);
            }
        }
    }
}
