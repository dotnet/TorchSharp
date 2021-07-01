// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    /// <summary>
    /// This class is used to represent a ConstantPad1d module.
    /// </summary>
    public class ConstantPad1d : torch.nn.Module
    {
        internal ConstantPad1d (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_ConstantPad1d_forward (torch.nn.Module.HType module, IntPtr tensor);

        /// <summary>
        /// Forward pass.
        /// </summary>
        /// <param name="tensor">Input tensor</param>
        /// <returns></returns>
        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_ConstantPad1d_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_ConstantPad1d_ctor (double value, long padding, out IntPtr pBoxedModule);

        /// <summary>
        /// Pads the input tensor using replication of the input boundary.
        /// </summary>
        /// <param name="padding">The size of the padding.</param>
        /// <param name="value"></param>
        /// <returns></returns>
        static public ConstantPad1d ConstantPad1d(long padding, double value)
        {
            var handle = THSNN_ConstantPad1d_ctor(value, padding, out var boxedHandle);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new ConstantPad1d(handle, boxedHandle);
        }
    }

    public static partial class functional
    {
        /// <summary>
        /// Pads the input tensor using replication of the input boundary.
        /// </summary>
        /// <param name="x">Input tensor</param>
        /// <param name="padding">The size of the padding: (padding_left , padding_right)</param>
        /// <param name="value"></param>
        /// <returns></returns>
        static public TorchTensor ConstantPad1d (TorchTensor x, long padding, double value)
        {
            using (var d =nn.ConstantPad1d (padding, value)) {
                return d.forward (x);
            }
        }
    }

}
