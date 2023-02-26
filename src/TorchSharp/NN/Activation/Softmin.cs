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
        /// This class is used to represent a Softmin module.
        /// </summary>
        public sealed class Softmin : torch.nn.Module<Tensor, Tensor>
        {
            internal Softmin(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Softmin_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Softmin).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Softmin
            /// </summary>
            /// <param name="dim">A dimension along which Softmin will be computed (so every slice along dim will sum to 1)</param>
            /// <returns></returns>
            public static Softmin Softmin(long dim)
            {
                var handle = THSNN_Softmin_ctor(dim, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Softmin(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softmin
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="dim">A dimension along which Softmin will be computed (so every slice along dim will sum to 1)</param>
                /// <returns></returns>
                public static Tensor softmin(Tensor x, long dim)
                {
                    using (var m = nn.Softmin(dim)) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
