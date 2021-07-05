// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a LeakyReLU module.
        /// </summary>
        public class LeakyReLU : torch.nn.Module
        {
            internal LeakyReLU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LeakyReLU_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_LeakyReLU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(LeakyReLU).Name;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_LeakyReLU_ctor(double negative_slope, bool inplace, out IntPtr pBoxedModule);

            /// <summary>
            /// Continuously Differentiable Exponential Linear Unit
            /// </summary>
            /// <param name="negativeSlope">The α value for the LeakyReLU formulation. Default: 1.0</param>
            /// <param name="inPlace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            static public LeakyReLU LeakyReLU(double negativeSlope = 1.0, bool inPlace = false)
            {
                var handle = THSNN_LeakyReLU_ctor(negativeSlope, inPlace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new LeakyReLU(handle, boxedHandle);
            }
        }
        public static partial class functional
        {
            /// <summary>
            /// Continuously Differentiable Exponential Linear Unit
            /// </summary>
            /// <param name="x">The input tensor</param>
            /// <param name="negativeSlope">The α value for the LeakyReLU formulation. Default: 1.0</param>
            /// <param name="inPlace">Do the operation in-place. Default: False</param>
            /// <returns></returns>
            static public Tensor LeakyReLU(Tensor x, double negativeSlope, bool inPlace = false)
            {
                using (var m = nn.LeakyReLU(negativeSlope, inPlace)) {
                    return m.forward(x);
                }
            }
        }
    }
}
