// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    /// <summary>
    /// This class is used to represent a CELU module.
    /// </summary>
    public class CELU : nn.Module
    {
        internal CELU (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_CELU_forward (nn.Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_CELU_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        public override string GetName ()
        {
            return typeof (CELU).Name;
        }
    }

    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_CELU_ctor (double alpha, bool inplace, out IntPtr pBoxedModule);

        /// <summary>
        /// Continuously Differentiable Exponential Linear Unit
        /// </summary>
        /// <param name="alpha">The α value for the CELU formulation. Default: 1.0</param>
        /// <param name="inPlace">Do the operation in-place. Default: False</param>
        /// <returns></returns>
        static public CELU CELU (double alpha = 1.0, bool inPlace = false)
        {
            var handle = THSNN_CELU_ctor (alpha, inPlace, out var boxedHandle);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new CELU (handle, boxedHandle);
        }
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
        static public TorchTensor CELU (TorchTensor x, double alpha, bool inPlace = false)
        {
            using (var m =nn.CELU (alpha, inPlace)) {
                return m.forward (x);
            }
        }
    }

}
