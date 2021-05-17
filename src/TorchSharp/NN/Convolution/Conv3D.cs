// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp.NN
{
    public class Conv3d : Module
    {
        internal Conv3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_Conv3d_forward(Module.HType module, IntPtr tensor);

        public override TorchTensor forward(TorchTensor tensor)
        {
            var res = THSNN_Conv3d_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_Conv3d_bias(Module.HType module);
        [DllImport("LibTorchSharp")]
        extern static void THSNN_Conv3d_set_bias(Module.HType module, IntPtr tensor);

        public TorchTensor? Bias {
            get {
                var res = THSNN_Conv3d_bias(handle);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return ((res == IntPtr.Zero) ? null : new TorchTensor(res));
            }
            set {
                THSNN_Conv3d_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                Torch.CheckForErrors();
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_Conv3d_weight(Module.HType module);
        [DllImport("LibTorchSharp")]
        extern static void THSNN_Conv3d_set_weight(Module.HType module, IntPtr tensor);

        public TorchTensor Weight {
            get {
                var res = THSNN_Conv3d_weight(handle);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
            set {
                THSNN_Conv3d_set_weight(handle, value.Handle);
                Torch.CheckForErrors();
            }
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_Conv3d_ctor(long inputChannel, long outputChannel, long kernelSize, long stride, long padding, long dilation, long paddingMode, long groups, bool bias, out IntPtr pBoxedModule);

        /// <summary>
        /// Applies a 2D convolution over an input signal composed of several input planes
        /// </summary>
        /// <param name="inputChannel">Number of channels in the input image</param>
        /// <param name="outputChannel">Number of channels produced by the convolution</param>
        /// <param name="kernelSize">Size of the convolving kernel</param>
        /// <param name="stride">Stride of the convolution. Default: 1</param>
        /// <param name="padding">Zero-padding added to both sides of the input. Default: 0</param>
        /// <param name="dilation">Spacing between kernel elements. Default: 1</param>
        /// <param name="paddingMode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
        /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
        /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
        /// <returns></returns>
        static public Conv3d Conv3d(long inputChannel, long outputChannel, long kernelSize, long stride = 1, long padding = 0, long dilation = 1, PaddingModes paddingMode = PaddingModes.Zeros, long groups = 1, bool bias = true)
        {
            var res = THSNN_Conv3d_ctor(inputChannel, outputChannel, kernelSize, stride, padding, dilation, (long)paddingMode, groups, bias, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Conv3d(res, boxedHandle);
        }
    }
    public static partial class Functions
    {
        /// <summary>
        /// Applies a 2D convolution over an input signal composed of several input planes
        /// </summary>
        /// <param name="x">Tensor representing an image, (N,C,H,W).</param>
        /// <param name="inputChannel">Number of channels in the input image</param>
        /// <param name="outputChannel">Number of channels produced by the convolution</param>
        /// <param name="kernelSize">Size of the convolving kernel</param>
        /// <param name="stride">Stride of the convolution. Default: 1</param>
        /// <param name="padding">Zero-padding added to both sides of the input. Default: 0</param>
        /// <param name="dilation">Spacing between kernel elements. Default: 1</param>
        /// <param name="paddingMode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
        /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
        /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
        /// <returns></returns>
        static public TorchTensor Conv3d(TorchTensor x, long inputChannel, long outputChannel, long kernelSize, long stride = 1, long padding = 0, long dilation = 1, PaddingModes paddingMode = PaddingModes.Zeros, long groups = 1, bool bias = true)
        {
            using (var d = Modules.Conv3d(inputChannel, outputChannel, kernelSize, stride, padding, dilation, paddingMode, groups, bias)) {
                return d.forward(x);
            }
        }
    }

}
