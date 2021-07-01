// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    /// <summary>
    /// This class is used to represent a AdaptiveAvgPool1D module.
    /// </summary>
    public class AdaptiveAvgPool1d : torch.nn.Module
    {
        internal AdaptiveAvgPool1d (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_AdaptiveAvgPool1d_forward (IntPtr module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_AdaptiveAvgPool1d_forward (handle.DangerousGetHandle (), tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }

    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_AdaptiveAvgPool1d_ctor (IntPtr psizes, int length, out IntPtr pBoxedModule);

        /// <summary>
        /// Applies a 1D adaptive average pooling over an input signal composed of several input planes.
        /// The output size is H, for any input size.The number of output features is equal to the number of input planes.
        /// </summary>
        /// <param name="outputSize">the target output size H</param>
        /// <returns></returns>
        static public AdaptiveAvgPool1d AdaptiveAvgPool1d(long outputSize)
        {
            unsafe {
                fixed (long* pkernelSize = new long[] { outputSize }) {
                    var handle = THSNN_AdaptiveAvgPool1d_ctor ((IntPtr)pkernelSize, 1, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new AdaptiveAvgPool1d (handle, boxedHandle);
                }
            }
        }
    }

    public static partial class functional
    {
        static public TorchTensor AdaptiveAvgPool1d(TorchTensor x, long outputSize)
        {
            using (var d =nn.AdaptiveAvgPool1d(outputSize)) {
                return d.forward (x);
            }
        }
    }
}
