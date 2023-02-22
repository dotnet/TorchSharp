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
        /// This class is used to represent a Tanh module.
        /// </summary>
        public sealed class Tanh : torch.nn.Module<Tensor, Tensor>
        {
            internal Tanh(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Tanh_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Tanh).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Tanh activation
            /// </summary>
            /// <returns></returns>
            public static Tanh Tanh()
            {
                var handle = THSNN_Tanh_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tanh(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Tanh activation
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                public static Tensor tanh(Tensor x)
                {
                    using (var m = nn.Tanh()) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
