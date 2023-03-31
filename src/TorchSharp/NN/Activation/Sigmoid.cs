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
        /// This class is used to represent a Sigmoid module.
        /// </summary>
        public sealed class Sigmoid : torch.nn.Module<Tensor, Tensor>
        {
            internal Sigmoid(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Sigmoid_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Sigmoid).Name;
            }
        }
    }
    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Sigmoid activation
            /// </summary>
            /// <returns></returns>
            public static Sigmoid Sigmoid()
            {
                var handle = THSNN_Sigmoid_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Sigmoid(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Sigmoid activation
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                public static Tensor Sigmoid(Tensor x)
                {
                    using (var m = nn.Sigmoid()) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
