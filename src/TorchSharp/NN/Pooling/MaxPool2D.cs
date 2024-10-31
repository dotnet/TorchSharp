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
        /// This class is used to represent a MaxPool2D module.
        /// </summary>
        public sealed class MaxPool2d : ParameterLessModule<Tensor, Tensor>
        {
            internal MaxPool2d(long[] kernel_size, long[] stride = null, long[] padding = null, long[] dilation = null, bool ceil_mode = false) : base(nameof(MaxPool2d))
            {
                this.kernel_size = kernel_size;
                this.stride = stride;
                this.padding = padding;
                this.dilation = dilation;
                this.ceil_mode = ceil_mode;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode);
            }

            public (Tensor Values, Tensor Indices) forward_with_indices(Tensor input)
            {
                return torch.nn.functional.max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode);
            }

            public long[] kernel_size { get; set; }
            public long[] stride { get; set; }
            public long[] padding { get; set; }
            public long[] dilation { get; set; }
            public bool ceil_mode { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 2D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceil_mode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            public static MaxPool2d MaxPool2d(long kernel_size, long? stride = null, long? padding = null, long? dilation = null, bool ceil_mode = false)
            {
                long[] kernelValue = new[] { kernel_size, kernel_size };
                long[] strideValue = stride.HasValue ? new[] { stride.Value, stride.Value } : kernelValue.ToArray();
                long[] paddingValue = padding.HasValue ? new[] { padding.Value, padding.Value } : new[] { 0L, 0L };
                long[] dilationValue = dilation.HasValue ? new[] { dilation.Value, dilation.Value } : new[] { 1L, 1L };

                return new MaxPool2d(kernelValue, strideValue, paddingValue, dilationValue, ceil_mode);
            }

            /// <summary>
            /// Applies a 2D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceil_mode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            public static unsafe MaxPool2d MaxPool2d((long, long) kernel_size, (long, long)? stride = null, (long, long)? padding = null, (long, long)? dilation = null, bool ceil_mode = false)
            {
                long[] kernelValue = new[] { kernel_size.Item1, kernel_size.Item2 };
                long[] strideValue = stride.HasValue ? new[] { stride.Value.Item1, stride.Value.Item2 } : kernelValue.ToArray();
                long[] paddingValue = padding.HasValue ? new[] { padding.Value.Item1, padding.Value.Item2 } : new[] { 0L, 0L };
                long[] dilationValue = dilation.HasValue ? new[] { dilation.Value.Item1, dilation.Value.Item2 } : new[] { 1L, 1L };

                return new MaxPool2d(kernelValue, strideValue, paddingValue, dilationValue, ceil_mode);
            }

            /// <summary>
            /// Applies a 2D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceil_mode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            public static MaxPool2d MaxPool2d(long[] kernel_size, long[] stride = null, long[] padding = null, long[] dilation = null, bool ceil_mode = false)
            {
                return new MaxPool2d(kernel_size, stride, padding, dilation, ceil_mode);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 2D max pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static Tensor max_pool2d(Tensor input, long[] kernel_size, long[] strides = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    strides = strides ?? kernel_size;
                    padding = padding ?? kernel_size.Select(x => 0L).ToArray();
                    dilation = dilation ?? kernel_size.Select(x => 1L).ToArray();
                    unsafe {
                        fixed (long* pkernel_size = kernel_size, pstrides = strides, ppadding = padding, pdilation = dilation) {
                            var res =
                                THSTensor_max_pool2d(input.Handle,
                                    (IntPtr)pkernel_size, kernel_size.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    (IntPtr)pdilation, dilation.Length,
                                    ceil_mode);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                /// <summary>
                /// Applies a 2D max pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static unsafe Tensor max_pool2d(Tensor input, long kernel_size, long? stride = null,
                    long? padding = null, long? dilation = null, bool ceil_mode = false)
                {
                    long svalue = stride.HasValue ? stride.Value : kernel_size;
                    long pvalue = padding.HasValue ? padding.Value : 0;
                    long dvalue = dilation.HasValue ? dilation.Value : 1;

                    long* pStride = stackalloc long[2] { svalue, svalue };
                    long* pPadding = stackalloc long[2] { pvalue, pvalue };
                    long* pDilation = stackalloc long[2] { dvalue, dvalue };

                    long* pkernel_size = stackalloc long[2] { kernel_size, kernel_size };

                    var res = THSTensor_max_pool2d(input.Handle,
                                    (IntPtr)pkernel_size, 2,
                                    (IntPtr)pStride, 2,
                                    (IntPtr)pPadding, 2,
                                    (IntPtr)pDilation, 2,
                                    ceil_mode);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Applies a 2D max pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static unsafe Tensor max_pool2d(Tensor input, (long, long) kernel_size, (long, long)? stride = null,
                    (long, long)? padding = null, (long, long)? dilation = null, bool ceil_mode = false)
                {
                    long svalue1 = stride != null ? stride.Value.Item1 : kernel_size.Item1;
                    long svalue2 = stride != null ? stride.Value.Item2 : kernel_size.Item2;
                    long pvalue1 = padding != null ? padding.Value.Item1 : 0;
                    long pvalue2 = padding != null ? padding.Value.Item2 : 0;
                    long dvalue1 = dilation != null ? dilation.Value.Item1 : 1;
                    long dvalue2 = dilation != null ? dilation.Value.Item2 : 1;

                    long* pStride = stackalloc long[2] { svalue1, svalue2 };
                    long* pPadding = stackalloc long[2] { pvalue1, pvalue2 };
                    long* pDilation = stackalloc long[2] { dvalue1, dvalue2 };

                    long* pkernel_size = stackalloc long[2] { kernel_size.Item1, kernel_size.Item2 };

                    var res = THSTensor_max_pool2d(input.Handle,
                                    (IntPtr)pkernel_size, 2,
                                    (IntPtr)pStride, 2,
                                    (IntPtr)pPadding, 2,
                                    (IntPtr)pDilation, 2,
                                    ceil_mode);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Applies a 2D max pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static (Tensor output, Tensor indices) max_pool2d_with_indices(Tensor input, long[] kernel_size, long[] strides = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    strides = strides ?? kernel_size;
                    padding = padding ?? kernel_size.Select(x => 0L).ToArray();
                    dilation = dilation ?? kernel_size.Select(x => 1L).ToArray();
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        unsafe {
                            fixed (long* pkernel_size = kernel_size, pstrides = strides, ppadding = padding, pdilation = dilation) {
                                THSTensor_max_pool2d_with_indices(input.Handle,
                                    pa.CreateArray,
                                    (IntPtr)pkernel_size, kernel_size.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    (IntPtr)pdilation, dilation.Length,
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
