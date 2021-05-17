// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a MaxPool1D module.
    /// </summary>
    public class MaxPool1d : Module
    {
        internal MaxPool1d (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_MaxPool1d_forward (Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_MaxPool1d_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_MaxPool1d_ctor (IntPtr pkernelSize, IntPtr pstrides, out IntPtr pBoxedModule);

        /// <summary>
        /// Applies a 1D max pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
        /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
        /// <returns></returns>
        static public MaxPool1d MaxPool1D (long kernelSize, long? stride = null)
        {
            return stride.HasValue ?
                MaxPool1d(new long[] { kernelSize }, new long[] { stride.Value }) :
                MaxPool1d(new long[] { kernelSize }, null);
        }

        static private MaxPool1d MaxPool1d(long[] kernelSize, long[] strides = null)
        {
            unsafe {
                fixed (long* pkernelSize = kernelSize, pstrides = strides) {
                    var handle = THSNN_MaxPool1d_ctor((IntPtr)pkernelSize, (IntPtr)pstrides, out var boxedHandle);
                    if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new MaxPool1d(handle, boxedHandle);
                }
            }
        }

    }

    public static partial class Functions
    {
        /// <summary>
        /// Applies a 1D max pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="x">Input signal</param>
        /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
        /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
        /// <returns></returns>
        static public TorchTensor MaxPool1d (TorchTensor x, long kernelSize, long? stride = null)
        {
            using (var d = Modules.MaxPool1D (kernelSize, stride)) {
                return d.forward (x);
            }
        }
    }
}
