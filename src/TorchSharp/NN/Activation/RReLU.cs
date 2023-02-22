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
        /// This class is used to represent a RReLU module.
        /// </summary>
        public sealed class RReLU : torch.nn.Module<Tensor, Tensor>
        {
            internal RReLU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_RReLU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(RReLU).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Randomized Rectified Linear Unit
            /// </summary>
            /// <param name="lower">Lower bound of the uniform distribution. Default: 1/8</param>
            /// <param name="upper">Upper bound of the uniform distribution. Default: 1/3</param>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            public static RReLU RReLU(double lower = one_eighth, double upper = one_third, bool inplace = false)
            {
                var handle = THSNN_RReLU_ctor(lower, upper, inplace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new RReLU(handle, boxedHandle);
            }

            private const double one_eighth = 1.0 / 8.0;
            private const double one_third = 1.0 / 3.0;

            public static partial class functional
            {
                /// <summary>
                /// Randomized Rectified Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="lower">Lower bound of the uniform distribution. Default: 1/8</param>
                /// <param name="upper">Upper bound of the uniform distribution. Default: 1/3</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor rrelu(Tensor x, double lower, double upper, bool inplace = false)
                {
                    using (var m = nn.RReLU(lower, upper, inplace)) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
