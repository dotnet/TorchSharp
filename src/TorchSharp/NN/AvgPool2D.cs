// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a AvgPool2D module.
    /// </summary>
    public class AvgPool2D : Module
    {
        internal AvgPool2D (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_AvgPool2d_forward (IntPtr module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_AvgPool2d_forward (handle.DangerousGetHandle (), tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_AvgPool2d_ctor (IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, out IntPtr pBoxedModule);

        static public AvgPool2D AvgPool2D (long[] kernelSize, long[] strides = null)
        {
            unsafe {
                fixed (long* pkernelSize = kernelSize, pstrides = strides) {
                    var handle = THSNN_AvgPool2d_ctor ((IntPtr)pkernelSize, kernelSize.Length, (IntPtr)pstrides, (strides == null ? 0 : strides.Length), out var boxedHandle);
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new AvgPool2D (handle, boxedHandle);
                }
            }
        }
    }

    public static partial class Functions
    {
        static public TorchTensor AvgPool2D (TorchTensor x, long[] kernelSize, long[] strides = null)
        {
            using (var d = Modules.AvgPool2D (kernelSize, strides)) {
                return d.Forward (x);
            }
        }
    }
}
