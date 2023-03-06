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
        /// This class is used to represent a Mish module.
        /// </summary>
        public sealed class Mish : torch.nn.Module<Tensor, Tensor>
        {
            internal Mish(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Mish_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Mish).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// A Self Regularized Non-Monotonic Neural Activation Function.
            /// </summary>
            /// <returns></returns>
            public static Mish Mish()
            {
                var handle = THSNN_Mish_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Mish(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// A Self Regularized Non-Monotonic Neural Activation Function.
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                public static Tensor Mish(Tensor x)
                {
                    using (var m = nn.Mish()) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
