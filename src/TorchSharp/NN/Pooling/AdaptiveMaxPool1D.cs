// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a AdaptiveMaxPool1D module.
    /// </summary>
    public class AdaptiveMaxPool1D : Module
    {
        internal AdaptiveMaxPool1D (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_AdaptiveMaxPool1d_forward (IntPtr module, IntPtr tensor);

        public TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_AdaptiveMaxPool1d_forward (handle.DangerousGetHandle (), tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_AdaptiveMaxPool1d_ctor (IntPtr psizes, int length, out IntPtr pBoxedModule);

        /// <summary>
        /// Applies a 1D adaptive max pooling over an input signal composed of several input planes.
        /// The output size is H, for any input size.The number of output features is equal to the number of input planes.
        /// </summary>
        /// <param name="outputSize">The target output size H.</param>
        /// <returns></returns>
        static public AdaptiveMaxPool1D AdaptiveMaxPool1D (long[] outputSize)
        {
            unsafe {
                fixed (long* pkernelSize = outputSize) {
                    var handle = THSNN_AdaptiveMaxPool1d_ctor ((IntPtr)pkernelSize, outputSize.Length, out var boxedHandle);
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new AdaptiveMaxPool1D (handle, boxedHandle);
                }
            }
        }
    }

    public static partial class Functions
    {
        /// <summary>
        /// Applies a 1D adaptive max pooling over an input signal composed of several input planes.
        /// The output size is H, for any input size.The number of output features is equal to the number of input planes.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="outputSize">The target output size H.</param>
        /// <returns></returns>
        static public TorchTensor AdaptiveMaxPool1D (TorchTensor x, long[] outputSize)
        {
            using (var d = Modules.AdaptiveMaxPool1D (outputSize)) {
                return d.forward (x);
            }
        }
    }
}
