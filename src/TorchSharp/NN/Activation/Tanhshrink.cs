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
        /// This class is used to represent a Tanhshrink module.
        /// </summary>
        public class Tanhshrink : torch.nn.Module<Tensor, Tensor>
        {
            internal Tanhshrink(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Tanhshrink_forward(torch.nn.Module<Tensor, Tensor>.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Tanhshrink_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Tanhshrink).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Tanhshrink_ctor(out IntPtr pBoxedModule);

            /// <summary>
            /// Tanhshrink
            /// </summary>
            /// <returns></returns>
            static public Tanhshrink Tanhshrink()
            {
                var handle = THSNN_Tanhshrink_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tanhshrink(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Tanhshrink
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                static public Tensor Tanhshrink(Tensor x)
                {
                    using (var m = nn.Tanhshrink()) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
