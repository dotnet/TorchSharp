// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a AvgPool1D module.
        /// </summary>
        public class AvgPool1d : torch.nn.Module
        {
            internal AvgPool1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_AvgPool1d_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
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
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_AvgPool1d_ctor(IntPtr pkernelSize, IntPtr pstrides, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 1D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the window</param>
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <returns></returns>
            static public AvgPool1d AvgPool1d(long kernelSize, long? stride = null)
            {
                return stride.HasValue ?
                    AvgPool1D(new long[] { kernelSize }, new long[] { stride.Value }) :
                    AvgPool1D(new long[] { kernelSize }, null);
            }

            static private AvgPool1d AvgPool1D(long[] kernelSize, long[] strides = null)
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
