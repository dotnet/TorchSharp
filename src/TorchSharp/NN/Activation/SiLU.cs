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
        /// This class is used to represent a SiLU module.
        /// </summary>
        public class SiLU : torch.nn.Module<Tensor, Tensor>
        {
            internal SiLU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_SiLU_forward(torch.nn.Module<Tensor, Tensor>.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_SiLU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(SiLU).Name;
            }
        }
    }
    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_SiLU_ctor(out IntPtr pBoxedModule);

            /// <summary>
            /// Sigmoid-Weighted Linear Unit
            /// </summary>
            /// <returns></returns>
            /// <remarks>The native libreary does not take an 'inplace' option, even though the PyTorch documentation mentions the parameter.</remarks>
            static public SiLU SiLU()
            {
                var handle = THSNN_SiLU_ctor(out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new SiLU(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Sigmoid-Weighted Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <returns></returns>
                static public Tensor SiLU(Tensor x)
                {
                    using (var m = nn.SiLU()) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
