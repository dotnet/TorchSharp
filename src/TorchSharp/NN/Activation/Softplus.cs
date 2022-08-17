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
        /// This class is used to represent a Softplus module.
        /// </summary>
        public class Softplus : torch.nn.Module
        {
            internal Softplus(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Softplus_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Softplus_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(Softplus).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Softplus_ctor(double beta, double threshold, out IntPtr pBoxedModule);

            /// <summary>
            /// Softplus
            /// </summary>
            /// <param name="beta">The β value for the Softplus formulation.</param>
            /// <param name="threshold">Values above this revert to a linear function</param>
            /// <returns></returns>
            static public Softplus Softplus(double beta = 1.0, double threshold = 20.0)
            {
                var handle = THSNN_Softplus_ctor(beta, threshold, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Softplus(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softplus
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="beta">The β value for the Softplus formulation.</param>
                /// <param name="threshold">Values above this revert to a linear function</param>
                /// <returns></returns>
                static public Tensor softplus(Tensor x, double beta = 1.0, double threshold = 20.0)
                {
                    using (var m = nn.Softplus(beta, threshold)) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
