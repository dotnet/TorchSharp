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
        /// This class is used to represent a CELU module.
        /// </summary>
        public class CELU : torch.nn.Module
        {
            internal CELU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_CELU_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_CELU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(CELU).Name;
            }
        }

    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_CELU_ctor(double alpha, [MarshalAs(UnmanagedType.U1)] bool inplace, out IntPtr pBoxedModule);

            /// <summary>
            /// Continuously Differentiable Exponential Linear Unit
            /// </summary>
            /// <param name="alpha">The α value for the CELU formulation. Default: 1.0</param>
            /// <param name="inPlace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            static public CELU CELU(double alpha = 1.0, bool inPlace = false)
            {
                var handle = THSNN_CELU_ctor(alpha, inPlace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new CELU(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Continuously Differentiable Exponential Linear Unit
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="alpha">The α value for the CELU formulation. Default: 1.0</param>
                /// <param name="inPlace">Do the operation in-place. Default: False</param>
                /// <returns></returns>
                static public Tensor celu(Tensor x, double alpha, bool inPlace = false)
                {
                    using (var m = nn.CELU(alpha, inPlace)) {
                        return m.forward(x);
                    }
                }
            }
        }
    }
}
