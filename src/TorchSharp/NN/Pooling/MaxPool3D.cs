// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a MaxPool3D module.
        /// </summary>
        public class MaxPool3d : torch.nn.Module
        {
            internal MaxPool3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_MaxPool3d_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_MaxPool3d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_MaxPool3d_forward_with_indices(torch.nn.Module.HType module, IntPtr tensor, out IntPtr indices);

            public (Tensor Values, Tensor Indices) forward_with_indices(Tensor tensor)
            {
                var res = THSNN_MaxPool3d_forward_with_indices(handle, tensor.Handle, out var indices);
                if (res == IntPtr.Zero || indices == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(indices));
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_MaxPool3d_ctor(IntPtr pkernelSize, int kernelSizeLength, IntPtr pstrides, int stridesLength, IntPtr pPadding, int paddingLength, IntPtr pDilation, int dilationLength, bool ceilMode, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 3D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceilMode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            static public MaxPool3d MaxPool3d(long kernelSize, long? stride = null, long? padding = null, long? dilation = null, bool ceilMode = false)
            {
                var pStride = stride.HasValue ? new long[] { stride.Value, stride.Value, stride.Value } : null;
                var pPadding = padding.HasValue ? new long[] { padding.Value, padding.Value, padding.Value } : null;
                var pDilation = dilation.HasValue ? new long[] { dilation.Value, dilation.Value, dilation.Value } : null;
                return MaxPool3d(new long[] { kernelSize, kernelSize, kernelSize }, pStride, pPadding, pDilation, ceilMode);
            }
            /// <summary>
            /// Applies a 3D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
            /// <param name="strides">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceilMode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            static public MaxPool3d MaxPool3d(long[] kernelSize, long[] strides = null, long[] padding = null, long[] dilation = null, bool ceilMode = false)
            {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, pPadding = padding, pDilation = dilation) {
                        var handle = THSNN_MaxPool3d_ctor((IntPtr)pkernelSize, kernelSize.Length, (IntPtr)pstrides, (strides == null ? 0 : strides.Length), (IntPtr)pPadding, (padding == null ? 0 : padding.Length), (IntPtr)pDilation, (dilation == null ? 0 : dilation.Length), ceilMode, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new MaxPool3d(handle, boxedHandle);
                    }
                }
            }
        }
    }
}
