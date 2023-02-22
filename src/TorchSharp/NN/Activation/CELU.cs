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
        /// This class is used to represent a CELU module.
        /// </summary>
        public sealed class CELU : torch.nn.Module<Tensor, Tensor>
        {
            internal CELU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_CELU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(CELU).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Continuously Differentiable Exponential Linear Unit
            /// </summary>
            /// <param name="alpha">The α value for the CELU formulation. Default: 1.0</param>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            public static CELU CELU(double alpha = 1.0, bool inplace = false)
            {
                var handle = THSNN_CELU_ctor(alpha, inplace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new CELU(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Continuously Differentiable Exponential Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="alpha">The α value for the CELU formulation. Default: 1.0</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor celu(Tensor x, double alpha, bool inplace = false)
                {
                    using (var m = nn.CELU(alpha, inplace)) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
