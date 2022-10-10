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
        /// This class is used to represent a Softshrink module.
        /// </summary>
        public sealed class Softshrink : torch.nn.Module<Tensor, Tensor>
        {
            internal Softshrink(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Softshrink_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Softshrink_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Softshrink).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Softshrink_ctor(double lambd, out IntPtr pBoxedModule);

            /// <summary>
            /// Softshrink
            /// </summary>
            /// <param name="lambda"> the λ value for the Softshrink formulation. Default: 0.5</param>
            /// <returns></returns>
            static public Softshrink Softshrink(double lambda = 0.5)
            {
                var handle = THSNN_Softshrink_ctor(lambda, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Softshrink(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softshrink
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="lambda">The λ value for the Softshrink formulation. Default: 0.5</param>
                /// <returns></returns>
                static public Tensor Softshrink(Tensor x, double lambda = 0.5)
                {
                    using (var m = nn.Softshrink(lambda)) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
