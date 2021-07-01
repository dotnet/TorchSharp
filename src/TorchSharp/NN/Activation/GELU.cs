// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    using impl;

    namespace impl
    {
        /// <summary>
        /// This class is used to represent a GELU module.
        /// </summary>
        public class GELU : torch.nn.Module
        {
            internal GELU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_GELU_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override TorchTensor forward(TorchTensor tensor)
            {
                var res = THSNN_GELU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new TorchTensor(res);
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
        }
        public static partial class functional
        {
            /// <summary>
            /// Gaussian Error Linear Units
            /// </summary>
            /// <param name="x">The input tensor</param>
            /// <returns></returns>
            static public TorchTensor GELU(TorchTensor x)
            {
                using (var m = nn.GELU()) {
                    return m.forward(x);
                }
            }
        }
    }
}
