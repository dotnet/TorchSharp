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
        /// This class is used to represent a log softmax module.
        /// </summary>
        public class LogSoftmax : torch.nn.Module
        {
            internal LogSoftmax(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LogSoftmax_forward(torch.nn.Module.HType handle, IntPtr tensor);

            public override TorchTensor forward(TorchTensor tensor)
            {
                var res = THSNN_LogSoftmax_forward(handle, tensor.Handle);
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
            extern static IntPtr THSNN_LogSoftmax_ctor(long dimension, out IntPtr pBoxedModule);

            static public LogSoftmax LogSoftmax(long dimension)
            {
                var handle = THSNN_LogSoftmax_ctor(dimension, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new LogSoftmax(handle, boxedHandle);
            }

            public static partial class functional
            {
                static public TorchTensor log_softmax(TorchTensor x, long dimension)
                {
                    using (var l = nn.LogSoftmax(dimension)) {
                        return l.forward(x);
                    }
                }
            }
        }
    }
}
