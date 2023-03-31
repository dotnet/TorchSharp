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
        /// This class is used to represent a AvgPool1D module.
        /// </summary>
        public sealed class AvgPool1d : torch.nn.Module<Tensor, Tensor>
        {
            internal AvgPool1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_AvgPool1d_forward(handle.DangerousGetHandle(), tensor.Handle);
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
            /// Applies a 1D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the window</param>
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <param name="padding">implicit zero padding to be added on both sides</param>
            /// <param name="ceil_mode">Whether to use ceil instead of floor to compute the output shape</param>
            /// <param name="count_include_pad">Whether to include the zero-padding in the averaging calculation</param>
            /// <param name="divisor_override">If specified, it will be used as divisor, otherwise size of the pooling region will be used</param>
            public static AvgPool1d AvgPool1d(long kernelSize, long? stride = null, long padding = 0, bool ceil_mode = false, bool count_include_pad = true, long? divisor_override = null)
            {
                return stride.HasValue ?
                    AvgPool1d(new long[] { kernelSize }, new long[] { stride.Value }, new long[] { padding }, ceil_mode, count_include_pad, divisor_override.HasValue ? divisor_override.Value : 0) :
                    AvgPool1d(new long[] { kernelSize }, null, new long[] { padding }, ceil_mode, count_include_pad, divisor_override.HasValue ? divisor_override.Value : 0);
            }

            /// <summary>
            /// Applies a 1D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the window</param>
            /// <param name="strides">The stride of the window. Default value is kernel_size</param>
            /// <param name="padding">implicit zero padding to be added on both sides</param>
            /// <param name="ceil_mode">Whether to use ceil instead of floor to compute the output shape</param>
            /// <param name="count_include_pad">Whether to include the zero-padding in the averaging calculation</param>
            /// <param name="divisor_override">If specified, it will be used as divisor, otherwise size of the pooling region will be used</param>
            private static AvgPool1d AvgPool1d(long[] kernelSize, long[] strides = null, long[] padding = null, bool ceil_mode = false, bool count_include_pad = true, long? divisor_override = null)
            {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding) {
                        var handle = THSNN_AvgPool1d_ctor((IntPtr)pkernelSize, (IntPtr)pstrides, (IntPtr)ppadding, ceil_mode, count_include_pad, divisor_override.HasValue ? divisor_override.Value : 0, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new AvgPool1d(handle, boxedHandle);
                    }
                }
            }

            public static partial class functional
            {

                /// <summary>
                /// Applies a 1D average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSize"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="ceil_mode"></param>
                /// <param name="count_include_pad"></param>
                /// <returns></returns>
                public static Tensor avg_pool1d(Tensor input, long kernelSize, long? stride = null,
                    long? padding = null, bool ceil_mode = false, bool count_include_pad = true)
                {
                    var kernelSizes = new long[] { kernelSize };
                    var strides = new long[] { stride ?? 1 };
                    var paddings = new long[] { padding ?? 0 };
                    unsafe {
                        fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                            var res =
                                THSTensor_avg_pool1d(input.Handle,
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

            }
        }
    }
}
