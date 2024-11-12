// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Google.Protobuf.WellKnownTypes;
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a MaxPool3D module.
        /// </summary>
        public sealed class MaxPool3d : ParameterLessModule<Tensor, Tensor>
        {
            internal MaxPool3d(long[] kernel_size, long[] stride = null, long[] padding = null, long[] dilation = null, bool ceil_mode = false) : base(nameof(MaxPool3d))
            {
                this.kernel_size = kernel_size;
                this.stride = stride;
                this.padding = padding;
                this.dilation = dilation;
                this.ceil_mode = ceil_mode;
            }

            public override Tensor forward(Tensor input)
            {
                return torch.nn.functional.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode);
            }

            public (Tensor Values, Tensor Indices) forward_with_indices(Tensor input)
            {
                return torch.nn.functional.max_pool3d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode);
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
            /// Applies a 3D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceil_mode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            public static MaxPool3d MaxPool3d(long kernel_size, long? stride = null, long? padding = null, long? dilation = null, bool ceil_mode = false)
            {
                var pStride = stride.HasValue ? new long[] { stride.Value, stride.Value, stride.Value } : null;
                var pPadding = padding.HasValue ? new long[] { padding.Value, padding.Value, padding.Value } : null;
                var pDilation = dilation.HasValue ? new long[] { dilation.Value, dilation.Value, dilation.Value } : null;
                return MaxPool3d(new long[] { kernel_size, kernel_size, kernel_size }, pStride, pPadding, pDilation, ceil_mode);
            }

            /// <summary>
            /// Applies a 3D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceil_mode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            public static MaxPool3d MaxPool3d((long, long, long) kernel_size, (long, long, long)? stride = null, (long, long, long)? padding = null, (long, long, long)? dilation = null, bool ceil_mode = false)
            {
                var pStride = stride.HasValue ? new long[] { stride.Value.Item1, stride.Value.Item2, stride.Value.Item3 } : null;
                var pPadding = padding.HasValue ? new long[] { padding.Value.Item1, padding.Value.Item2, padding.Value.Item3 } : null;
                var pDilation = dilation.HasValue ? new long[] { dilation.Value.Item1, dilation.Value.Item2, dilation.Value.Item3 } : null;
                return MaxPool3d(new long[] { kernel_size.Item1, kernel_size.Item2, kernel_size.Item3 }, pStride, pPadding, pDilation, ceil_mode);
            }

            /// <summary>
            /// Applies a 3D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceil_mode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            public static MaxPool3d MaxPool3d(long[] kernel_size, long[] stride = null, long[] padding = null, long[] dilation = null, bool ceil_mode = false)
            {
                return new MaxPool3d(kernel_size, stride, padding, dilation, ceil_mode);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 3D max pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static Tensor max_pool3d(Tensor input, long[] kernel_size, long[] strides = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    strides = strides ?? kernel_size;
                    padding = padding ?? kernel_size.Select(x => 0L).ToArray();
                    dilation = dilation ?? kernel_size.Select(x => 1L).ToArray();
                    unsafe {
                        fixed (long* pkernel_size = kernel_size, pstrides = strides, ppadding = padding, pdilation = dilation) {
                            var res =
                                THSTensor_max_pool3d(input.Handle,
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
                /// Applies a 3D max pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernel_size"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static (Tensor output, Tensor indices) max_pool3d_with_indices(Tensor input, long[] kernel_size, long[] strides = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    strides = strides ?? kernel_size;
                    padding = padding ?? kernel_size.Select(x => 0L).ToArray();
                    dilation = dilation ?? kernel_size.Select(x => 1L).ToArray();
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        unsafe {
                            fixed (long* pkernel_size = kernel_size, pstrides = strides, ppadding = padding, pdilation = dilation) {
                                THSTensor_max_pool3d_with_indices(input.Handle,
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
