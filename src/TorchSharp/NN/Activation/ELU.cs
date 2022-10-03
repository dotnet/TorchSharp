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
        /// This class is used to represent a ELU module.
        /// </summary>
        public class ELU : torch.nn.Module<Tensor, Tensor>
        {
            internal ELU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_ELU_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ELU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(ELU).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_ELU_ctor(double alpha, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

            /// <summary>
            /// Exponential Linear Unit
            /// </summary>
            /// <param name="alpha">The α value for the ELU formulation. Default: 1.0</param>
            /// <param name="inplace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            static public ELU ELU(double alpha = 1.0, bool inplace = false)
            {
                var handle = THSNN_ELU_ctor(alpha, inplace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ELU(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Exponential Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="alpha">The α value for the ELU formulation. Default: 1.0</param>
                /// <param name="inplace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                static public Tensor elu(Tensor x, double alpha, bool inplace = false)
                {
                    using (var m = nn.ELU(alpha, inplace)) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
