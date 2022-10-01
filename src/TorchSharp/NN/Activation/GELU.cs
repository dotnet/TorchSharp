// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a GELU module.
        /// </summary>
        public class GELU : torch.nn.Module<Tensor, Tensor>
        {
            internal GELU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_GELU_forward(torch.nn.Module<Tensor, Tensor>.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_GELU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(GELU).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_GELU_ctor(out IntPtr pBoxedModule);

            /// <summary>
            /// Gaussian Error Linear Units
            /// </summary>
            /// <returns></returns>
            static public GELU GELU()
            {
                var handle = THSNN_GELU_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new GELU(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Gaussian Error Linear Units
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                static public Tensor gelu(Tensor x)
                {
                    using (var m = nn.GELU()) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
