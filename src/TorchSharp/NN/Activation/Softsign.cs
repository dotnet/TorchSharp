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
        /// This class is used to represent a Softsign module.
        /// </summary>
        public class Softsign : torch.nn.Module<Tensor, Tensor>
        {
            internal Softsign(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Softsign_forward(torch.nn.Module<Tensor, Tensor>.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Softsign_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Softsign).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Softsign_ctor(out IntPtr pBoxedModule);

            /// <summary>
            /// Softsign
            /// </summary>
            /// <returns></returns>
            static public Softsign Softsign()
            {
                var handle = THSNN_Softsign_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Softsign(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softsign
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                static public Tensor Softsign(Tensor x)
                {
                    using (var m = nn.Softsign()) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
