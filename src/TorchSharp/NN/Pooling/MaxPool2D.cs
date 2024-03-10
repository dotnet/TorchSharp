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
        public sealed class MaxPool2d : ParamLessModule<Tensor, Tensor>
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
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static Tensor max_pool2d(Tensor input, long[] kernel_size, long[] stride = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    var ret = max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode); 
                    ret.Indices.Dispose();
                    return ret.Values;
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
                public static Tensor max_pool2d(Tensor input, long kernel_size, long? stride = null,
                    long? padding = null, long? dilation = null, bool ceil_mode = false)
                {
                    long[] kernelValue = new[] { kernel_size, kernel_size };
                    long[] strideValue = stride.HasValue ? new[] { stride.Value, stride.Value } : kernelValue.ToArray();
                    long[] paddingValue = padding.HasValue ? new[] { padding.Value, padding.Value } : new[] { 0L, 0L };
                    long[] dilationValue = dilation.HasValue ? new[] { dilation.Value, dilation.Value } : new[] { 1L, 1L };

                    var ret = max_pool2d_with_indices(input, kernelValue, strideValue, paddingValue, dilationValue, ceil_mode);
                    ret.Indices.Dispose();
                    return ret.Values;
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
                    long[] kernelValue = new[] { kernel_size.Item1, kernel_size.Item2 };
                    long[] strideValue = stride.HasValue ? new[] { stride.Value.Item1, stride.Value.Item2 } : kernelValue.ToArray();
                    long[] paddingValue = padding.HasValue ? new[] { padding.Value.Item1, padding.Value.Item2 } : new[] { 0L, 0L };
                    long[] dilationValue = dilation.HasValue ? new[] { dilation.Value.Item1, dilation.Value.Item2 } : new[] { 1L, 1L };

                    var ret = max_pool2d_with_indices(input, kernelValue, strideValue, paddingValue, dilationValue, ceil_mode);
                    ret.Indices.Dispose();
                    return ret.Values;
                }

                /// <summary>
                /// Applies a 2D max pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSize"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static (Tensor Values, Tensor Indices) max_pool2d_with_indices(Tensor input, long[] kernelSize, long[] strides = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    strides ??= kernelSize;
                    padding ??= kernelSize.Select(x => 0L).ToArray();
                    dilation ??= kernelSize.Select(x => 1L).ToArray();
                    unsafe {
                        fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                            var resOutput = THSTensor_max_pool2d_with_indices(input.Handle,
                                (IntPtr)pkernelSize, kernelSize.Length,
                                (IntPtr)pstrides, strides.Length,
                                (IntPtr)ppadding, padding.Length,
                                (IntPtr)pdilation, dilation.Length,
                                ceil_mode, out var resIndices);

                            if (resOutput == IntPtr.Zero || resIndices == IntPtr.Zero) { torch.CheckForErrors(); }
                            return (new Tensor(resOutput), new Tensor(resIndices));
                        }
                    }
                }
            }
        }
    }
}
