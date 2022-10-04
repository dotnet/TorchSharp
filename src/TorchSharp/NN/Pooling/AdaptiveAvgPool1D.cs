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
        /// This class is used to represent a AdaptiveAvgPool1D module.
        /// </summary>
        public class AdaptiveAvgPool1d : torch.nn.Module<Tensor, Tensor>
        {
            internal AdaptiveAvgPool1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_AdaptiveAvgPool1d_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AdaptiveAvgPool1d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            extern static IntPtr THSNN_AdaptiveAvgPool1d_ctor(IntPtr psizes, int length, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 1D adaptive average pooling over an input signal composed of several input planes.
            /// The output size is H, for any input size.The number of output features is equal to the number of input planes.
            /// </summary>
            /// <param name="outputSize">the target output size H</param>
            /// <returns></returns>
            static public unsafe AdaptiveAvgPool1d AdaptiveAvgPool1d(long outputSize)
            {
                long* pkernelSize = stackalloc long[1] { outputSize };
                var handle = THSNN_AdaptiveAvgPool1d_ctor((IntPtr)pkernelSize, 1, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AdaptiveAvgPool1d(handle, boxedHandle);
            }
        }
    }
}
