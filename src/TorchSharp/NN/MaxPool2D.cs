// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a MaxPool2D module.
    /// </summary>
    public class MaxPool2D : Module
    {
        internal MaxPool2D (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_MaxPool2d_forward (Module.HType module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_MaxPool2d_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_MaxPool2d_ctor (IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, out IntPtr pBoxedModule);

        static public MaxPool2D MaxPool2D (long[] kernelSize, long[] strides = null)
        {
            unsafe {
                fixed (long* pkernelSize = kernelSize, pstrides = strides) {
                    var handle = THSNN_MaxPool2d_ctor ((IntPtr)pkernelSize, kernelSize.Length, (IntPtr)pstrides, (strides == null ? 0 : strides.Length), out var boxedHandle);
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new MaxPool2D (handle, boxedHandle);
                }
            }
        }
    }

    public static partial class Functions
    {
        static public TorchTensor MaxPool2D (TorchTensor x, long[] kernelSize, long[] strides = null)
        {
            using (var d = Modules.MaxPool2D (kernelSize, strides)) {
                return d.Forward (x);
            }
        }
    }
}
