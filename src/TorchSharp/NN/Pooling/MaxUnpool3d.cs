// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a MaxUnpool3d module.
        /// </summary>
        public sealed class MaxUnpool3d : torch.nn.Module<Tensor, Tensor, long[], Tensor>
        {
            internal MaxUnpool3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor, Tensor indices, long[] output_size = null)
            {
                unsafe {
                    fixed (long* pOutSize = output_size) {
                        var res = THSNN_MaxUnpool3d_forward(handle, tensor.Handle, indices.Handle, (IntPtr)pOutSize, output_size == null ? 0 : output_size.Length);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            public new Tensor call(Tensor tensor, Tensor indices, long[] output_size = null)
            {
                return base.call(tensor, indices, output_size);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 2D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <returns></returns>
            public static MaxUnpool3d MaxUnpool3d(long kernelSize, long? stride = null, long? padding = null)
            {
                var pStride = stride.HasValue ? new long[] { stride.Value, stride.Value, stride.Value } : null;
                var pPadding = padding.HasValue ? new long[] { padding.Value, padding.Value, padding.Value } : null;
                return MaxUnpool3d(new long[] { kernelSize, kernelSize, kernelSize }, pStride, pPadding);
            }

            /// <summary>
            /// Applies a 2D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <returns></returns>
            public static MaxUnpool3d MaxUnpool3d((long, long, long) kernelSize, (long, long, long)? stride = null, (long, long, long)? padding = null)
            {
                var pStride = stride.HasValue ? new long[] { stride.Value.Item1, stride.Value.Item2, stride.Value.Item3 } : null;
                var pPadding = padding.HasValue ? new long[] { padding.Value.Item1, padding.Value.Item2, padding.Value.Item3 } : null;
                return MaxUnpool3d(new long[] { kernelSize.Item1, kernelSize.Item2, kernelSize.Item3 }, pStride, pPadding);
            }

            /// <summary>
            /// Applies a 2D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
            /// <param name="strides">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <returns></returns>
            public static MaxUnpool3d MaxUnpool3d(long[] kernelSize, long[] strides = null, long[] padding = null)
            {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, pPadding = padding) {
                        var handle = THSNN_MaxUnpool3d_ctor((IntPtr)pkernelSize, kernelSize.Length, (IntPtr)pstrides, (strides == null ? 0 : strides.Length), (IntPtr)pPadding, (padding == null ? 0 : padding.Length), out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new MaxUnpool3d(handle, boxedHandle);
                    }
                }
            }
        }
    }
}
