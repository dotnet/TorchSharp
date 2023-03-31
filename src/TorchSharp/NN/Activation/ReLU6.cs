// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;
    using TorchSharp.PInvoke;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a ReLU6 module.
        /// </summary>
        public sealed class ReLU6 : torch.nn.Module<Tensor, Tensor>
        {
            internal ReLU6(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = LibTorchSharp.THSNN_ReLU6_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(ReLU6).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Rectified Linear Unit
            ///
            /// This ReLU version caps positive values at 6.
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            public static ReLU6 ReLU6(bool inplace = false)
            {
                var handle = LibTorchSharp.THSNN_ReLU6_ctor(inplace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ReLU6(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Rectified Linear Unit
                ///
                /// This ReLU version caps positive values at 6.
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor relu6(Tensor x, bool inplace = false)
                {
                    using (var m = nn.ReLU6(inplace)) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
