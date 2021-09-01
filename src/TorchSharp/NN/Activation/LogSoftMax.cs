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
        /// This class is used to represent a log softmax module.
        /// </summary>
        public class LogSoftmax : torch.nn.Module
        {
            internal LogSoftmax(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LogSoftmax_forward(torch.nn.Module.HType handle, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_LogSoftmax_forward(handle, tensor.Handle);
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
            extern static IntPtr THSNN_LogSoftmax_ctor(long dimension, out IntPtr pBoxedModule);

            static public LogSoftmax LogSoftmax(long dimension)
            {
                var handle = THSNN_LogSoftmax_ctor(dimension, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new LogSoftmax(handle, boxedHandle);
            }

            public static partial class functional
            {
                static public Tensor log_softmax(Tensor x, long dimension)
                {
                    using (var l = nn.LogSoftmax(dimension)) {
                        return l.forward(x);
                    }
                }
            }
        }
    }
}
