// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a ZeroPad2d module.
    /// </summary>
    public class ZeroPad2d : Module
    {
        internal ZeroPad2d (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_ZeroPad2d_forward (Module.HType module, IntPtr tensor);

        /// <summary>
        /// Forward pass.
        /// </summary>
        /// <param name="tensor">Input tensor</param>
        /// <returns></returns>
        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_ZeroPad2d_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_ZeroPad2d_ctor (long padding, out IntPtr pBoxedModule);

        /// <summary>
        /// Pads the input tensor boundaries with zero.
        /// </summary>
        /// <param name="padding">The size of the padding.</param>
        /// <returns></returns>
        static public ZeroPad2d ZeroPad2d(long padding)
        {
            var handle = THSNN_ZeroPad2d_ctor(padding, out var boxedHandle);
            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
            return new ZeroPad2d(handle, boxedHandle);
        }
    }

    public static partial class Functions
    {
        /// <summary>
        /// Pads the input tensor boundaries with zero.
        /// </summary>
        /// <param name="x">Input tensor</param>
        /// <param name="padding">The size of the padding: (padding_left , padding_right, padding_top, padding_bottom)</param>
        /// <returns></returns>
        static public TorchTensor ZeroPad2d (TorchTensor x, long padding)
        {
            using (var d = Modules.ZeroPad2d (padding)) {
                return d.forward (x);
            }
        }
    }

}
