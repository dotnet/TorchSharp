// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp
{
    /// <summary>
    /// This class is used to represent a MaxPool1D module.
    /// </summary>
    public class MaxPool1d : nn.Module
    {
        internal MaxPool1d (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_MaxPool1d_forward (nn.Module.HType module, IntPtr tensor);

        public override TorchTensor forward (TorchTensor tensor)
        {
            var res = THSNN_MaxPool1d_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_MaxPool1d_forward_with_indices(nn.Module.HType module, IntPtr tensor, out IntPtr indices);

        public (TorchTensor Values, TorchTensor Indices) forward_with_indices(TorchTensor tensor)
        {
            var res = THSNN_MaxPool1d_forward_with_indices(handle, tensor.Handle, out var indices);
            if (res == IntPtr.Zero || indices == IntPtr.Zero) { torch.CheckForErrors(); }
            return (new TorchTensor(res), new TorchTensor(indices));
        }
    }
    public static partial class nn
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_MaxPool1d_ctor (IntPtr pkernelSize, IntPtr pStrides, IntPtr pPadding, IntPtr pDilation, bool ceilMode, out IntPtr pBoxedModule);

        /// <summary>
        /// Applies a 1D max pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
        /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
        /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
        /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
        /// <param name="ceilMode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
        /// <returns></returns>
        static public MaxPool1d MaxPool1d (long kernelSize, long? stride = null, long? padding = null, long? dilation = null, bool ceilMode = false)
        {
            var pStride = stride.HasValue ? new long[] { stride.Value } : null;
            var pPadding = padding.HasValue ? new long[] { padding.Value } : null;
            var pDilation = dilation.HasValue ? new long[] { dilation.Value } : null;
            return MaxPool1d(new long[] { kernelSize }, pStride, pPadding, pDilation, ceilMode);
        }

        static private MaxPool1d MaxPool1d(long[] kernelSize, long[] strides = null, long[] padding = null, long[] dilation = null, bool ceilMode = false)
        {
            unsafe {
                fixed (long* pkernelSize = kernelSize, pstrides = strides, pPadding = padding, pDilation = dilation) {
                    var handle = THSNN_MaxPool1d_ctor((IntPtr)pkernelSize, (IntPtr)pstrides, (IntPtr)pPadding, (IntPtr)pDilation, ceilMode, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new MaxPool1d(handle, boxedHandle);
                }
            }
        }

    }

    public static partial class functional
    {
        /// <summary>
        /// Applies a 1D max pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="x">Input signal</param>
        /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
        /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
        /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
        /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
        /// <param name="ceilMode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
        /// <returns></returns>
        static public TorchTensor MaxPool1d (TorchTensor x, long kernelSize, long? stride = null, long? padding = null, long? dilation = null, bool ceilMode = false)
        {
            using (var d =nn.MaxPool1d(kernelSize, stride, padding, dilation, ceilMode)) {
                return d.forward (x);
            }
        }
    }
}
