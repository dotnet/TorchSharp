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
        /// This class is used to represent a ReLU module.
        /// </summary>
        public sealed class ReLU : torch.nn.Module<Tensor, Tensor>
        {
            internal ReLU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ReLU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(ReLU).Name;
            }
        }
    }
    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Rectified Linear Unit
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            public static ReLU ReLU(bool inplace = false)
            {
                var handle = THSNN_ReLU_ctor(inplace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ReLU(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Rectified Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor relu(Tensor x, bool inplace = false)
                {
                    using (var m = nn.ReLU(inplace)) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
