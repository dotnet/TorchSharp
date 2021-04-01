// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a AvgPool3D module.
    /// </summary>
    public class AvgPool3d : Module
    {
        internal AvgPool3d (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_AvgPool3d_forward (IntPtr module, IntPtr tensor);

        public TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_AvgPool3d_forward (handle.DangerousGetHandle (), tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_AvgPool3d_ctor (IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, out IntPtr pBoxedModule);

        /// <summary>
        /// Applies a 3D average pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="kernelSize">The size of the window</param>
        /// <param name="strides">The stride of the window. Default value is kernel_size</param>
        /// <returns></returns>
        static public AvgPool3d AvgPool3D (long[] kernelSize, long[] strides = null)
        {
            unsafe {
                fixed (long* pkernelSize = kernelSize, pstrides = strides) {
                    var handle = THSNN_AvgPool3d_ctor ((IntPtr)pkernelSize, kernelSize.Length, (IntPtr)pstrides, (strides == null ? 0 : strides.Length), out var boxedHandle);
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new AvgPool3d (handle, boxedHandle);
                }
            }
        }
    }

    public static partial class Functions
    {
        /// <summary>
        /// Applies a 3D average pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="x">The input signal tensor</param>
        /// <param name="kernelSize">The size of the window</param>
        /// <param name="strides">The stride of the window. Default value is kernel_size</param>
        /// <returns></returns>
        static public TorchTensor AvgPool3D (TorchTensor x, long[] kernelSize, long[] strides = null)
        {
            using (var d = Modules.AvgPool3D (kernelSize, strides)) {
                return d.forward (x);
            }
        }
    }
}
