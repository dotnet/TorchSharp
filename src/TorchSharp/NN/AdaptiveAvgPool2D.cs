// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a AdaptiveAvgPool2D module.
    /// </summary>
    public class AdaptiveAvgPool2D : Module
    {
        internal AdaptiveAvgPool2D (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_AdaptiveAvgPool2d_forward (IntPtr module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_AdaptiveAvgPool2d_forward (handle.DangerousGetHandle (), tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_AdaptiveAvgPool2d_ctor (IntPtr psizes, int length, out IntPtr pBoxedModule);

        static public AdaptiveAvgPool2D AdaptiveAvgPool2D (long[] kernelSize)
        {
            unsafe {
                fixed (long* pkernelSize = kernelSize) {
                    var handle = THSNN_AdaptiveAvgPool2d_ctor ((IntPtr)pkernelSize, kernelSize.Length, out var boxedHandle);
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new AdaptiveAvgPool2D (handle, boxedHandle);
                }
            }
        }
    }

    public static partial class Functions
    {
        static public TorchTensor AdaptiveAvgPool2D (TorchTensor x, long[] kernelSize)
        {
            using (var d = Modules.AdaptiveAvgPool2D (kernelSize)) {
                return d.Forward (x);
            }
        }
    }
}
