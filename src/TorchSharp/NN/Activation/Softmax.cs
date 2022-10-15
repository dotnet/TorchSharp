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
        /// This class is used to represent a Softmax module.
        /// </summary>
        public sealed class Softmax : torch.nn.Module<Tensor, Tensor>
        {
            internal Softmax(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Softmax_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Softmax).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Softmax
            /// </summary>
            /// <param name="dim">A dimension along which Softmax will be computed (so every slice along dim will sum to 1)</param>
            /// <returns></returns>
            public static Softmax Softmax(long dim)
            {
                var handle = THSNN_Softmax_ctor(dim, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Softmax(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softmax
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="dim">A dimension along which Softmax will be computed (so every slice along dim will sum to 1)</param>
                /// <returns></returns>
                public static Tensor softmax(Tensor x, long dim)
                {
                    using (var m = nn.Softmax(dim)) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
