// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using impl;

    namespace impl
    {
        /// <summary>
        /// This class is used to represent a Sigmoid module.
        /// </summary>
        public class Sigmoid : torch.nn.Module
        {
            internal Sigmoid(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Sigmoid_forward(torch.nn.Module.HType module, IntPtr tensor);

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
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Sigmoid_ctor(out IntPtr pBoxedModule);

            /// <summary>
            /// Sigmoid activation
            /// </summary>
            /// <returns></returns>
            static public Sigmoid Sigmoid()
            {
                var handle = THSNN_Sigmoid_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Sigmoid(handle, boxedHandle);
            }
        }
        public static partial class functional
        {
            /// <summary>
            /// Sigmoid activation
            /// </summary>
            /// <param name="x">The input tensor</param>
            /// <returns></returns>
            static public Tensor Sigmoid(Tensor x)
            {
                using (var m = nn.Sigmoid()) {
                    return m.forward(x);
                }
            }
        }
    }
}
