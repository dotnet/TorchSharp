// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a AvgPool3D module.
        /// </summary>
        public sealed class AvgPool3d : ParameterLessModule<Tensor, Tensor>
        {
            internal AvgPool3d(long[] kernel_size, long[] stride = null, long[] padding = null, bool ceil_mode = false, bool count_include_pad = true, long? divisor_override = null) : base(nameof(AvgPool3d))
            {
                this.kernel_size = kernel_size;
                this.stride = stride;
                this.padding = padding;
                this.ceil_mode = ceil_mode;
                this.count_include_pad = count_include_pad;
                this.divisor_override = divisor_override;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
            }

            public long[] kernel_size { get; set; }
            public long[] stride { get; set; }
            public long[] padding { get; set; }
            public bool ceil_mode { get; set; }
            public bool count_include_pad { get; set; }
            public long? divisor_override { get; set; }
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
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <param name="padding">implicit zero padding to be added on both sides</param>
            /// <param name="ceil_mode">Whether to use ceil instead of floor to compute the output shape</param>
            /// <param name="count_include_pad">Whether to include the zero-padding in the averaging calculation</param>
            /// <param name="divisor_override">If specified, it will be used as divisor, otherwise size of the pooling region will be used</param>
            public static AvgPool3d AvgPool3d(long[] kernel_size, long[] stride = null, long[] padding = null, bool ceil_mode = false, bool count_include_pad = true, long? divisor_override = null)
            {
                return new AvgPool3d(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
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
                long[] kernelValue = new[] { kernel_size.Item1, kernel_size.Item2, kernel_size.Item3 };
                long[] strideValue = stride == null ? null : new[] { stride.Value.Item1, stride.Value.Item2, stride.Value.Item3 };
                long[] paddingValue = padding == null ? null : new[] { padding.Value.Item1, padding.Value.Item2, padding.Value.Item3 };
                return new AvgPool3d(kernelValue, strideValue, paddingValue, ceil_mode, count_include_pad, divisor_override);
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
                long[] kernelValue = new[] { kernel_size, kernel_size, kernel_size };
                long[] strideValue = stride == null ? null : new[] { stride.Value, stride.Value, stride.Value };
                long[] paddingValue = padding == null ? null : new[] { padding.Value, padding.Value, padding.Value };
                return new AvgPool3d(kernelValue, strideValue, paddingValue, ceil_mode, count_include_pad, divisor_override);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies 3D average-pooling operation in kT x kH x kW regions by step size sT x sH x sW steps.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="ceil_mode"></param>
                /// <param name="count_include_pad"></param>
                /// <param name="divisor_override"></param>
                /// <returns></returns>
                public static Tensor avg_pool3d(Tensor input, long[] kernel_size,
                    long[] stride = null,
                    long[] padding = null,
                    bool ceil_mode = false,
                    bool count_include_pad = true,
                    long? divisor_override = null)
                {
                    stride = (stride == null) ? kernel_size : stride;
                    padding = (padding == null) ? new long[] { 0 } : padding;
                    unsafe {
                        fixed (long* pkernel_size = kernel_size, pstrides = stride, ppadding = padding) {
                            var res =
                                THSTensor_avg_pool3d(input.Handle,
                                    (IntPtr)pkernel_size, kernel_size.Length,
                                    (IntPtr)pstrides, stride.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    ceil_mode,
                                    count_include_pad, divisor_override ?? 0);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                public static Tensor avg_pool3d_backward(Tensor input, Tensor originalInput,
                    long[] kernel_sizes,
                    long[] strides = null,
                    long[] paddings = null,
                    bool ceil_mode = false,
                    bool count_include_pad = true,
                    long? divisor_override = null)
                {
                    strides = (strides == null) ? kernel_sizes : strides;
                    paddings = (paddings == null) ? new long[] { 0 } : paddings;
                    unsafe {
                        fixed (long* pkernel_size = kernel_sizes, pstrides = strides, ppadding = paddings) {
                            var res =
                                THSTensor_avg_pool3d_backward(input.Handle, originalInput.Handle,
                                    (IntPtr)pkernel_size, kernel_sizes.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    ceil_mode,
                                    count_include_pad,
                                    divisor_override ?? 0);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }
            }
        }
    }
}
