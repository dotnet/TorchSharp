// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a MaxPool1D module.
        /// </summary>
        public class MaxPool1d : torch.nn.Module
        {
            internal MaxPool1d(IntPtr handle, IntPtr boxedHandle, bool return_indices) : base(handle, boxedHandle)
            {
                _return_indices = return_indices;
            }

            private bool _return_indices;

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_MaxPool1d_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                if (_return_indices)
                    new InvalidOperationException("MaxPool1d is set to return indices, so you cannot call forward(Tensor). Use forward(object), instead.");
                var res = THSNN_MaxPool1d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_MaxPool1d_forward_with_indices(torch.nn.Module.HType module, IntPtr tensor, out IntPtr indices);

            public (Tensor Values, Tensor Indices) forward_with_indices(Tensor tensor)
            {
                var res = THSNN_MaxPool1d_forward_with_indices(handle, tensor.Handle, out var indices);
                if (res == IntPtr.Zero || indices == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(indices));
            }

            public override object forward(object input)
            {
                var tensor = (Tensor)input;

                if (_return_indices) {
                    var res = THSNN_MaxPool1d_forward_with_indices(handle, tensor.Handle, out var indices);
                    if (res == IntPtr.Zero || indices == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (new Tensor(res), new Tensor(indices));
                } else {
                    var res = THSNN_MaxPool1d_forward(handle, tensor.Handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            public object forward(object x, object y)
            {
                return null;
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_MaxPool1d_ctor(IntPtr pkernelSize, IntPtr pStrides, IntPtr pPadding, IntPtr pDilation, [MarshalAs(UnmanagedType.U1)] bool ceilMode, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 1D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="return_indices">If true, will return the argmax along with the max values. Useful for torch.nn.MaxUnpool1d later</param>
            /// <param name="ceilMode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            static public MaxPool1d MaxPool1d(long kernelSize, long? stride = null, long? padding = null, long? dilation = null, bool return_indices = false, bool ceilMode = false)
            {
                var pStride = stride.HasValue ? new long[] { stride.Value } : null;
                var pPadding = padding.HasValue ? new long[] { padding.Value } : null;
                var pDilation = dilation.HasValue ? new long[] { dilation.Value } : null;
                return MaxPool1d(new long[] { kernelSize }, pStride, pPadding, pDilation, return_indices, ceilMode);
            }

            /// <summary>
            /// Applies a 1D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
            /// <param name="strides">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="return_indices">If true, will return the argmax along with the max values. Useful for torch.nn.MaxUnpool1d later</param>
            /// <param name="ceilMode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            static private MaxPool1d MaxPool1d(long[] kernelSize, long[] strides = null, long[] padding = null, long[] dilation = null, bool return_indices = false, bool ceilMode = false)
            {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, pPadding = padding, pDilation = dilation) {
                        var handle = THSNN_MaxPool1d_ctor((IntPtr)pkernelSize, (IntPtr)pstrides, (IntPtr)pPadding, (IntPtr)pDilation, ceilMode, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new MaxPool1d(handle, boxedHandle, return_indices);
                    }
                }
            }
        }
    }
}
