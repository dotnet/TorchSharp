// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    /// <summary>
    /// This class is used to represent a dropout module.
    /// </summary>
    public class PixelShuffle : nn.Module
    {
        internal PixelShuffle (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_PixelShuffle_forward (nn.Module.HType module, IntPtr tensor);

        /// <summary>
        /// Forward pass.
        /// </summary>
        /// <param name="tensor">Input tensor</param>
        /// <returns></returns>
        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_PixelShuffle_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_PixelShuffle_ctor (long upscaleFactor, out IntPtr pBoxedModule);

        /// <summary>
        /// Rearranges elements in a tensor of shape (*, C * r^2, H, W) to a tensor of shape(*, C, H * r, W * r), where r is an upscale factor.
        /// This is useful for implementing efficient sub-pixel convolution with a stride of 1/r.
        /// </summary>
        /// <param name="upscaleFactor">Factor to increase spatial resolution by</param>
        /// <returns></returns>
        static public PixelShuffle PixelShuffle (long upscaleFactor)
        {
            var handle = THSNN_PixelShuffle_ctor (upscaleFactor, out var boxedHandle);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new PixelShuffle(handle, boxedHandle);
        }
    }

    public static partial class functional
    {
        /// <summary>
        /// Rearranges elements in a tensor of shape (*, C * r^2, H, W) to a tensor of shape(*, C, H * r, W * r), where r is an upscale factor.
        /// This is useful for implementing efficient sub-pixel convolution with a stride of 1/r.
        /// </summary>
        /// <param name="x">Input tensor</param>
        /// <param name="upscaleFactor">Factor to increase spatial resolution by</param>
        /// <returns></returns>
        /// <returns></returns>
        static public TorchTensor PixelShuffle (TorchTensor x, long upscaleFactor)
        {
            using (var d =nn.PixelShuffle (upscaleFactor)) {
                return d.forward (x);
            }
        }
    }

}
