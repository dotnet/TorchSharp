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
        /// This class is used to represent a AvgPool1D module.
        /// </summary>
        public sealed class AvgPool1d : ParameterLessModule<Tensor, Tensor>
        {
            internal AvgPool1d(long kernel_size, long? stride = null, long? padding = null, bool ceil_mode = false, bool count_include_pad = true) : base(nameof(AvgPool1d))
            {
                this.kernel_size = kernel_size;
                this.stride = stride;
                this.padding = padding;
                this.ceil_mode = ceil_mode;
                this.count_include_pad = count_include_pad;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad);
            }

            public long kernel_size { get; set; }
            public long? stride { get; set; }
            public long? padding { get; set; }
            public bool ceil_mode { get; set; }
            public bool count_include_pad { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 1D average pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the window</param>
            /// <param name="stride">The stride of the window. Default value is kernel_size</param>
            /// <param name="padding">implicit zero padding to be added on both sides</param>
            /// <param name="ceil_mode">Whether to use ceil instead of floor to compute the output shape</param>
            /// <param name="count_include_pad">Whether to include the zero-padding in the averaging calculation</param>
            public static AvgPool1d AvgPool1d(long kernel_size, long? stride = null, long padding = 0, bool ceil_mode = false, bool count_include_pad = true)
            {
                return new AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad);
            }

            public static partial class functional
            {

                /// <summary>
                /// Applies a 1D average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="ceil_mode"></param>
                /// <param name="count_include_pad"></param>
                /// <returns></returns>
                public static Tensor avg_pool1d(Tensor input, long kernel_size, long? stride = null,
                    long? padding = null, bool ceil_mode = false, bool count_include_pad = true)
                {
                    var kernel_sizes = new long[] { kernel_size };
                    var strides = new long[] { stride ?? kernel_size };
                    var paddings = new long[] { padding ?? 0 };
                    unsafe {
                        fixed (long* pkernel_size = kernel_sizes, pstrides = strides, ppadding = paddings) {
                            var res =
                                THSTensor_avg_pool1d(input.Handle,
                                    (IntPtr)pkernel_size, kernel_sizes.Length,
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
