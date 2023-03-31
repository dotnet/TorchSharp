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
        /// This class is used to represent a Softmax2d module.
        /// </summary>
        public sealed class Softmax2d : torch.nn.Module<Tensor, Tensor>
        {
            internal Softmax2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Softmax2d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Softmax2d).Name;
            }
        }
    }
    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies Softmax over features to each spatial location
            /// </summary>
            /// <returns></returns>
            public static Softmax2d Softmax2d()
            {
                var handle = THSNN_Softmax2d_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Softmax2d(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies Softmax over features to each spatial location
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                public static Tensor softmax2d(Tensor x)
                {
                    using (var m = nn.Softmax2d()) {
                        return m.call(x);
                    }
                }
            }
        }
    }
}
