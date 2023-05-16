// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a log softmax module.
        /// </summary>
        public sealed class LogSoftmax : torch.nn.Module<Tensor, Tensor>
        {
            internal LogSoftmax(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

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
            public static LogSoftmax LogSoftmax(long dim)
            {
                var handle = THSNN_LogSoftmax_ctor(dim, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new LogSoftmax(handle, boxedHandle);
            }

            public static partial class functional
            {
                public static Tensor log_softmax(Tensor x, long dim)
                {
                    return torch.special.log_softmax(x, dim);
                }
            }
        }
    }
}
