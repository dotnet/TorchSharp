// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a MaxPool1D module.
        /// </summary>
        public sealed class MaxPool1d : ParameterLessModule<Tensor, Tensor>
        {
            internal MaxPool1d(long kernel_size, long? stride = null, long? padding = null, long? dilation = null, bool ceil_mode = false) : base(nameof(MaxPool1d))
            {
                this.kernel_size = kernel_size;
                this.stride = stride;
                this.padding = padding;
                this.dilation = dilation;
                this.ceil_mode = ceil_mode;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode);
            }

            public (Tensor Values, Tensor Indices) forward_with_indices(Tensor input)
            {
                return torch.nn.functional.max_pool1d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode);
            }

            public long kernel_size { get; set; }
            public long? stride { get; set; }
            public long? padding { get; set; }
            public long? dilation { get; set; }
            public bool ceil_mode { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 1D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceil_mode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            public static MaxPool1d MaxPool1d(long kernel_size, long? stride = null, long? padding = null, long? dilation = null, bool ceil_mode = false)
            {
                return new MaxPool1d(kernel_size, stride, padding, dilation, ceil_mode);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 1D max pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static Tensor max_pool1d(Tensor input, long kernel_size, long? stride = null,
                    long? padding = null, long? dilation = null, bool ceil_mode = false)
                {
                    var kernel_sizes = new long[] { kernel_size };
                    var strides = new long[] { stride ?? kernel_size };
                    var paddings = new long[] { padding ?? 0 };
                    var dilations = new long[] { dilation ?? 1 };
                    unsafe {
                        fixed (long* pkernel_size = kernel_sizes, pstrides = strides, ppadding = paddings, pdilation = dilations) {
                            var res =
                                THSTensor_max_pool1d(input.Handle,
                                    (IntPtr)pkernel_size, kernel_sizes.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    (IntPtr)pdilation, dilations.Length,
                                    ceil_mode);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                /// <summary>
                /// Applies a 1D max pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static (Tensor output, Tensor indices) max_pool1d_with_indices(Tensor input, long kernel_size, long? stride = null,
                    long? padding = null, long? dilation = null, bool ceil_mode = false)
                {
                    var kernel_sizes = new long[] { kernel_size };
                    var strides = new long[] { stride ?? kernel_size };
                    var paddings = new long[] { padding ?? 0 };
                    var dilations = new long[] { dilation ?? 1 };
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        unsafe {
                            fixed (long* pkernel_size = kernel_sizes, pstrides = strides, ppadding = paddings, pdilation = dilations) {
                                THSTensor_max_pool1d_with_indices(input.Handle,
                                    pa.CreateArray,
                                    (IntPtr)pkernel_size, kernel_sizes.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    (IntPtr)pdilation, dilations.Length,
                                    ceil_mode);
                                torch.CheckForErrors();
                            }
                        }
                        ptrArray = pa.Array;
                    }
                    return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
                }
            }
        }
    }
}
