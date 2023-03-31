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
        /// This class is used to represent a SELU module.
        /// </summary>
        public sealed class SELU : torch.nn.Module<Tensor, Tensor>
        {
            internal SELU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_SELU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(SELU).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Scaled Exponential Linear Unit
            /// </summary>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            public static SELU SELU(bool inplace = false)
            {
                var handle = THSNN_SELU_ctor(inplace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new SELU(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Scaled Exponential Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                public static Tensor selu(Tensor x, bool inplace = false)
                {
                    using (var m = nn.SELU(inplace)) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
