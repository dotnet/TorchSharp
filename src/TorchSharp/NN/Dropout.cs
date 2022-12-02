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
        /// This class is used to represent a dropout module.
        /// </summary>
        public sealed class Dropout : torch.nn.Module<Tensor, Tensor>
        {
            internal Dropout(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            /// <summary>
            /// Forward pass.
            /// </summary>
            /// <param name="tensor">Input tensor</param>
            /// <returns></returns>
            protected override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Dropout_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
            /// Each channel will be zeroed out independently on every forward call.
            /// </summary>
            /// <param name="p">Probability of an element to be zeroed. Default: 0.5</param>
            /// <param name="inplace">If set to true, will do this operation in-place. Default: false</param>
            /// <returns></returns>
            public static Dropout Dropout(double p = 0.5, bool inplace = false)
            {
                var handle = THSNN_Dropout_ctor(p, inplace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Dropout(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
                /// Each channel will be zeroed out independently on every forward call.
                /// </summary>
                /// <returns></returns>
                public static Tensor dropout(Tensor input, double p = 0.5, bool training = true, bool inplace = false)
                {
                    var res = THSNN_dropout(input.Handle, p, training, inplace);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
