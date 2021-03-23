// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Conv1D : Module
    {
        internal Conv1D (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Conv1d_forward (Module.HType module, IntPtr tensor);

        public TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_Conv1d_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Conv1d_ctor (long inputChannel, long outputChannel, long kernelSize, long stride, long padding, out IntPtr pBoxedModule);

        /// <summary>
        /// Applies a 1D convolution over an input signal composed of several input planes.
        /// </summary>
        /// <param name="inputChannel">Number of channels in the input image</param>
        /// <param name="outputChannel">Number of channels produced by the convolution</param>
        /// <param name="kernelSize">Size of the convolving kernel</param>
        /// <param name="stride">Stride of the convolution. Default: 1</param>
        /// <param name="padding">Zero-padding added to both sides of the input. Default: 0</param>
        /// <returns>Tensor of shape (N,C_out,L_out)</returns>
        static public Conv1D Conv1D (long inputChannel, long outputChannel, long kernelSize, long stride = 1, long padding = 0)
        {
            var res = THSNN_Conv1d_ctor (inputChannel, outputChannel, kernelSize, stride, padding, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Conv1D (res, boxedHandle);
        }
    }
    public static partial class Functions
    {
        /// <summary>
        /// Applies a 1D convolution over an input signal composed of several input planes.
        /// </summary>
        /// <param name="x">Tensor of shape (N,C_in,L_in)</param>
        /// <param name="inputChannel">Number of channels in the input image</param>
        /// <param name="outputChannel">Number of channels produced by the convolution</param>
        /// <param name="kernelSize">Size of the convolving kernel</param>
        /// <param name="stride">Stride of the convolution. Default: 1</param>
        /// <param name="padding">Zero-padding added to both sides of the input. Default: 0</param>
        /// <returns>Tensor of shape (N,C_out,L_out)</returns>
        static public TorchTensor Conv1D (TorchTensor x, long inputChannel, long outputChannel, long kernelSize, long stride = 1, long padding = 0)
        {
            using (var d = Modules.Conv1D (inputChannel, outputChannel, kernelSize, stride, padding)) {
                return d.forward (x);
            }
        }
    }

}
