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
        /// This class is used to represent a Tanhshrink module.
        /// </summary>
        public sealed class Tanhshrink : torch.nn.Module<Tensor, Tensor>
        {
            internal Tanhshrink(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Tanhshrink_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Tanhshrink).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Tanhshrink
            /// </summary>
            /// <returns></returns>
            public static Tanhshrink Tanhshrink()
            {
                var handle = THSNN_Tanhshrink_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tanhshrink(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Tanhshrink
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                public static Tensor Tanhshrink(Tensor x)
                {
                    using (var m = nn.Tanhshrink()) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
