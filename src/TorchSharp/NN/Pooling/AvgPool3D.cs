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
        /// This class is used to represent a AvgPool3D module.
        /// </summary>
        public sealed class AvgPool3d : torch.nn.Module<Tensor, Tensor>
        {
            internal AvgPool3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AvgPool3d_forward(handle.DangerousGetHandle(), tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 3D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="strides">The stride of the window. Default value is kernel_size</param>
            /// <param name="padding">implicit zero padding to be added on both sides</param>
            /// <param name="ceil_mode">Whether to use ceil instead of floor to compute the output shape</param>
            /// <param name="count_include_pad">Whether to include the zero-padding in the averaging calculation</param>
            /// <param name="divisor_override">If specified, it will be used as divisor, otherwise size of the pooling region will be used</param>
            public static AvgPool3d AvgPool3d(long[] kernel_size, long[] strides = null, long[] padding = null, bool ceil_mode = false, bool count_include_pad = true, long? divisor_override = null)
            {
                unsafe {
                    fixed (long* pkernelSize = kernel_size, pstrides = strides, ppadding = padding) {
                        var handle = THSNN_AvgPool3d_ctor((IntPtr)pkernelSize, kernel_size.Length, (IntPtr)pstrides, (strides == null ? 0 : strides.Length), (IntPtr)ppadding, (padding == null ? 0 : padding.Length), ceil_mode, count_include_pad, divisor_override.HasValue ? divisor_override.Value : 0, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new AvgPool3d(handle, boxedHandle);
                    }
                }
            }

            /// <summary>
            /// Applies a 3D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <param name="padding">implicit zero padding to be added on both sides</param>
            /// <param name="ceil_mode">Whether to use ceil instead of floor to compute the output shape</param>
            /// <param name="count_include_pad">Whether to include the zero-padding in the averaging calculation</param>
            /// <param name="divisor_override">If specified, it will be used as divisor, otherwise size of the pooling region will be used</param>
            public static unsafe AvgPool3d AvgPool3d((long, long, long) kernel_size, (long, long, long)? stride = null, (long, long, long)? padding = null, bool ceil_mode = false, bool count_include_pad = true, long? divisor_override = null)
            {
                long svalue1 = (stride == null) ? kernel_size.Item1 : stride.Value.Item1;
                long svalue2 = (stride == null) ? kernel_size.Item2 : stride.Value.Item2;
                long svalue3 = (stride == null) ? kernel_size.Item3 : stride.Value.Item3;

                long pvalue1 = (padding == null) ? 0 : stride.Value.Item1;
                long pvalue2 = (padding == null) ? 0 : stride.Value.Item2;
                long pvalue3 = (padding == null) ? 0 : stride.Value.Item3;

                long* pkernelSize = stackalloc long[3] { kernel_size.Item1, kernel_size.Item2, kernel_size.Item3 };
                long* pstrides = stackalloc long[3] { svalue1, svalue2, svalue3 };
                long* ppadding = stackalloc long[3] { pvalue1, pvalue2, pvalue3 };

                var handle = THSNN_AvgPool3d_ctor((IntPtr)pkernelSize, 3, (IntPtr)pstrides, 3, (IntPtr)ppadding, 3, ceil_mode, count_include_pad, divisor_override.HasValue ? divisor_override.Value : 0, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AvgPool3d(handle, boxedHandle);
            }

            /// <summary>
            /// Applies a 3D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <param name="padding">implicit zero padding to be added on both sides</param>
            /// <param name="ceil_mode">Whether to use ceil instead of floor to compute the output shape</param>
            /// <param name="count_include_pad">Whether to include the zero-padding in the averaging calculation</param>
            /// <param name="divisor_override">If specified, it will be used as divisor, otherwise size of the pooling region will be used</param>
            public static unsafe AvgPool3d AvgPool3d(long kernel_size, long? stride = null, long? padding = null, bool ceil_mode = false, bool count_include_pad = true, long? divisor_override = null)
            {
                long svalue = (stride == null) ? kernel_size : stride.Value;
                long pvalue = (stride == null) ? 0 : padding.Value;

                long* pkernelSize = stackalloc long[3] { kernel_size, kernel_size, kernel_size };
                long* pstrides = stackalloc long[3] { svalue, svalue, svalue };
                long* ppadding = stackalloc long[3] { pvalue, pvalue, pvalue };

                var handle = THSNN_AvgPool3d_ctor((IntPtr)pkernelSize, 3, (IntPtr)pstrides, 3, (IntPtr)ppadding, 3, ceil_mode, count_include_pad, divisor_override.HasValue ? divisor_override.Value : 0, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new AvgPool3d(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies 3D average-pooling operation in kT x kH x kW regions by step size sT x sH x sW steps.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSizes"></param>
                /// <param name="strides"></param>
                /// <param name="paddings"></param>
                /// <param name="ceil_mode"></param>
                /// <param name="count_include_pad"></param>
                /// <returns></returns>
                public static Tensor avg_pool3d(Tensor input, long[] kernelSizes,
                    long[] strides = null,
                    long[] paddings = null,
                    bool ceil_mode = false,
                    bool count_include_pad = true)
                {
                    strides = (strides == null) ? new long[] { 1 } : strides;
                    paddings = (paddings == null) ? new long[] { 0 } : paddings;
                    unsafe {
                        fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                            var res =
                                THSTensor_avg_pool3d(input.Handle,
                                    (IntPtr)pkernelSize, kernelSizes.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    ceil_mode,
                                    count_include_pad);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                public static Tensor avg_pool3d_backward(Tensor input, Tensor originalInput,
                    long[] kernelSizes,
                    long[] strides = null,
                    long[] paddings = null,
                    bool ceil_mode = false,
                    bool count_include_pad = true,
                    long divisorOverride = 0)
                {
                    strides = (strides == null) ? new long[] { 1 } : strides;
                    paddings = (paddings == null) ? new long[] { 0 } : paddings;
                    unsafe {
                        fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                            var res =
                                THSTensor_avg_pool3d_backward(input.Handle, originalInput.Handle,
                                    (IntPtr)pkernelSize, kernelSizes.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    ceil_mode,
                                    count_include_pad,
                                    divisorOverride);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }
            }
        }
    }
}
