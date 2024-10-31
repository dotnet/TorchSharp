// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using System.Data;
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a AvgPool2D module.
        /// </summary>
        public sealed class AvgPool2d : ParameterLessModule<Tensor, Tensor>
        {
            internal AvgPool2d(long[] kernel_size, long[] stride = null, long[] padding = null, bool ceil_mode = false, bool count_include_pad = true, long? divisor_override = null) : base(nameof(AvgPool2d))
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
                return torch.nn.functional.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
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
            /// Applies a 2D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <param name="padding">implicit zero padding to be added on both sides</param>
            /// <param name="ceil_mode">Whether to use ceil instead of floor to compute the output shape</param>
            /// <param name="count_include_pad">Whether to include the zero-padding in the averaging calculation</param>
            /// <param name="divisor_override">If specified, it will be used as divisor, otherwise size of the pooling region will be used</param>
            public static AvgPool2d AvgPool2d(long[] kernel_size, long[] stride = null, long[] padding = null, bool ceil_mode = false, bool count_include_pad = true, long? divisor_override = null)
            {
                return new AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
            }

            /// <summary>
            /// Applies a 2D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window.</param>
            /// <param name="padding">implicit zero padding to be added on both sides</param>
            /// <param name="ceil_mode">Whether to use ceil instead of floor to compute the output shape</param>
            /// <param name="count_include_pad">Whether to include the zero-padding in the averaging calculation</param>
            /// <param name="divisor_override">If specified, it will be used as divisor, otherwise size of the pooling region will be used</param>
            public static unsafe AvgPool2d AvgPool2d((long,long) kernel_size, (long,long)? stride = null, (long,long)? padding = null, bool ceil_mode = false, bool count_include_pad = true, long? divisor_override = null)
            {
                long[] kernelValue = new[] { kernel_size.Item1, kernel_size.Item2 };
                long[] strideValue = stride == null ? null : new[] { stride.Value.Item1, stride.Value.Item2 };
                long[] paddingValue = padding == null ? null : new[] { padding.Value.Item1, padding.Value.Item2 };
                return new AvgPool2d(kernelValue, strideValue, paddingValue, ceil_mode, count_include_pad, divisor_override);
            }

            /// <summary>
            /// Applies a 2D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window.</param>
            /// <param name="padding">implicit zero padding to be added on both sides</param>
            /// <param name="ceil_mode">Whether to use ceil instead of floor to compute the output shape</param>
            /// <param name="count_include_pad">Whether to include the zero-padding in the averaging calculation</param>
            /// <param name="divisor_override">If specified, it will be used as divisor, otherwise size of the pooling region will be used</param>
            public static AvgPool2d AvgPool2d(long kernel_size, long? stride = null, long? padding = null, bool ceil_mode = false, bool count_include_pad = true, long? divisor_override = null)
            {
                long[] kernelValue = new[] { kernel_size, kernel_size };
                long[] strideValue = stride == null ? null : new[] { stride.Value, stride.Value };
                long[] paddingValue = padding == null ? null : new[] { padding.Value, padding.Value };
                return new AvgPool2d(kernelValue, strideValue, paddingValue, ceil_mode, count_include_pad, divisor_override);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies 2D average-pooling operation in kH × kW regions by step size sH * sW steps. The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="ceil_mode"></param>
                /// <param name="count_include_pad"></param>
                /// <param name="divisor_override"></param>
                /// <returns></returns>
                public static Tensor avg_pool2d(Tensor input, long[] kernel_size,
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
                                THSTensor_avg_pool2d(input.Handle,
                                    (IntPtr)pkernel_size, kernel_size.Length,
                                    (IntPtr)pstrides, stride.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    ceil_mode,
                                    count_include_pad,
                                    divisor_override ?? 0);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                /// <summary>
                /// Applies 2D average-pooling operation in kH × kW regions by step size sH * sW steps. The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="ceil_mode"></param>
                /// <param name="count_include_pad"></param>
                /// <param name="divisor_override"></param>
                /// <returns></returns>
                public static unsafe Tensor avg_pool2d(Tensor input, long kernel_size,
                    long? stride = null,
                    long padding = 0,
                    bool ceil_mode = false,
                    bool count_include_pad = true,
                    long? divisor_override = null)
                {
                    long svalue = (stride == null) ? kernel_size : stride.Value;

                    long* pkernel_size = stackalloc long[2] { kernel_size, kernel_size };
                    long* pstrides = stackalloc long[2] { svalue, svalue };
                    long* ppadding = stackalloc long[2] { padding, padding };

                    var res =
                        THSTensor_avg_pool2d(input.Handle,
                            (IntPtr)pkernel_size, 2,
                            (IntPtr)pstrides, 2,
                            (IntPtr)ppadding, 2,
                            ceil_mode,
                            count_include_pad,
                            divisor_override ?? 0);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Applies 2D average-pooling operation in kH × kW regions by step size sH * sW steps. The number of output features is equal to the number of input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="ceil_mode"></param>
                /// <param name="count_include_pad"></param>
                /// <param name="divisor_override"></param>
                /// <returns></returns>
                public static unsafe Tensor avg_pool2d(Tensor input, (long, long) kernel_size,
                    (long, long)? stride = null,
                    (long, long)? padding = null,
                    bool ceil_mode = false,
                    bool count_include_pad = true,
                    long? divisor_override = null)
                {
                    long svalue1 = (stride == null) ? kernel_size.Item1 : stride.Value.Item1;
                    long svalue2 = (stride == null) ? kernel_size.Item2 : stride.Value.Item2;
                    long pvalue1 = padding != null ? padding.Value.Item1 : 0;
                    long pvalue2 = padding != null ? padding.Value.Item2 : 0;

                    long* pstrides = stackalloc long[2] { svalue1, svalue2 };
                    long* ppadding = stackalloc long[2] { pvalue1, pvalue2 };

                    long* pkernel_size = stackalloc long[2] { kernel_size.Item1, kernel_size.Item2 };

                    var res =
                        THSTensor_avg_pool2d(input.Handle,
                            (IntPtr)pkernel_size, 2,
                            (IntPtr)pstrides, 2,
                            (IntPtr)ppadding, 2,
                            ceil_mode,
                            count_include_pad,
                            divisor_override ?? 0);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                public static Tensor avg_pool2d_backward(Tensor input, Tensor originalInput,
                    long[] kernel_sizes,
                    long[] strides = null,
                    long[] paddings = null,
                    bool ceil_mode = false,
                    bool count_include_pad = true,
                    long? divisor_override = null)
                {
                    strides = (strides == null) ? new long[] { 1 } : strides;
                    paddings = (paddings == null) ? new long[] { 0 } : paddings;
                    unsafe {
                        fixed (long* pkernel_size = kernel_sizes, pstrides = strides, ppadding = paddings) {
                            var res =
                                THSTensor_avg_pool2d_backward(input.Handle, originalInput.Handle,
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
