// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a AdaptiveAvgPool3D module.
    /// </summary>
    public class AdaptiveAvgPool3d : Module
    {
        internal AdaptiveAvgPool3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
        {
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_AdaptiveAvgPool3d_forward(IntPtr module, IntPtr tensor);

        public TorchTensor forward(TorchTensor tensor)
        {
            var res = THSNN_AdaptiveAvgPool3d_forward(handle.DangerousGetHandle(), tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_AdaptiveAvgPool3d_ctor(IntPtr psizes, int length, out IntPtr pBoxedModule);

        /// <summary>
        /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
        /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
        /// </summary>
        /// <param name="outputSize">The target output size of the image of the form H x W.</param>
        /// <returns></returns>
        static public AdaptiveAvgPool3d AdaptiveAvgPool3d(long[] outputSize)
        {
            unsafe {
                fixed (long* pkernelSize = outputSize) {
                    var handle = THSNN_AdaptiveAvgPool3d_ctor((IntPtr)pkernelSize, outputSize.Length, out var boxedHandle);
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new AdaptiveAvgPool3d(handle, boxedHandle);
                }
            }
        }
    }

    public static partial class Functions
    {
        /// <summary>
        /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
        /// The output is of size H x W, for any input size.The number of output features is equal to the number of input planes.
        /// </summary>
        /// <param name="x">The input signal tensor.</param>
        /// <param name="outputSize">The target output size of the image of the form H x W.</param>
        /// <returns></returns>
        static public TorchTensor AdaptiveAvgPool3d(TorchTensor x, long[] outputSize)
        {
            using (var d = Modules.AdaptiveAvgPool3d(outputSize)) {
                return d.forward(x);
            }
        }
    }
}
