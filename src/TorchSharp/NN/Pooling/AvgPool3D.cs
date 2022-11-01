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
        /// This class is used to represent a AvgPool3D module.
        /// </summary>
        public sealed class AvgPool3d : torch.nn.Module<Tensor, Tensor>
        {
            internal AvgPool3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AvgPool3d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            /// Applies a 3D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="strides">The stride of the window. Default value is kernel_size</param>
            /// <returns></returns>
            public static AvgPool3d AvgPool3d(long[] kernel_size, long[] strides = null)
            {
                unsafe {
                    fixed (long* pkernelSize = kernel_size, pstrides = strides) {
                        var handle = THSNN_AvgPool3d_ctor((IntPtr)pkernelSize, kernel_size.Length, (IntPtr)pstrides, (strides == null ? 0 : strides.Length), out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new AvgPool3d(handle, boxedHandle);
                    }
                }
            }

            /// <summary>
            /// Applies a 3D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <returns></returns>
            public static unsafe AvgPool3d AvgPool3d((long, long, long) kernel_size, (long, long, long)? stride = null)
            {
                long svalue1 = (stride == null) ? kernel_size.Item1 : stride.Value.Item1;
                long svalue2 = (stride == null) ? kernel_size.Item2 : stride.Value.Item2;
                long svalue3 = (stride == null) ? kernel_size.Item3 : stride.Value.Item3;

                long* pkernelSize = stackalloc long[3] { kernel_size.Item1, kernel_size.Item2, kernel_size.Item3 };
                long* pstrides = stackalloc long[3] { svalue1, svalue2, svalue3 };

                var handle = THSNN_AvgPool3d_ctor((IntPtr)pkernelSize, 3, (IntPtr)pstrides, 3, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AvgPool3d(handle, boxedHandle);
            }

            /// <summary>
            /// Applies a 3D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <returns></returns>
            public static unsafe AvgPool3d AvgPool3d(long kernel_size, long? stride = null)
            {
                long svalue = (stride == null) ? kernel_size : stride.Value;

                long* pkernelSize = stackalloc long[3] { kernel_size, kernel_size, kernel_size };
                long* pstrides = stackalloc long[3] { svalue, svalue, svalue };

                var handle = THSNN_AvgPool3d_ctor((IntPtr)pkernelSize, 3, (IntPtr)pstrides, 3, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AvgPool3d(handle, boxedHandle);
            }
        }
    }
}
