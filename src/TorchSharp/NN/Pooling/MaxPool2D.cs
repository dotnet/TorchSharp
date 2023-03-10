// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a MaxPool2D module.
        /// </summary>
        public sealed class MaxPool2d : torch.nn.Module<Tensor, Tensor>
        {
            internal MaxPool2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_MaxPool2d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
            public (Tensor Values, Tensor Indices) forward_with_indices(Tensor tensor)
            {
                var res = THSNN_MaxPool2d_forward_with_indices(handle, tensor.Handle, out var indices);
                if (res == IntPtr.Zero || indices == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(indices));
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
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceilMode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            public static unsafe MaxPool2d MaxPool2d(long kernelSize, long? stride = null, long? padding = null, long? dilation = null, bool ceilMode = false)
            {
                long svalue = stride.HasValue ? stride.Value : kernelSize;
                long pvalue = padding.HasValue ? padding.Value : 0;
                long dvalue = dilation.HasValue ? dilation.Value : 1;

                long* pStride = stackalloc long[2] { svalue, svalue };
                long* pPadding = stackalloc long[2] { pvalue, pvalue };
                long* pDilation = stackalloc long[2] { dvalue, dvalue };

                long* pkernelSize = stackalloc long[2] { kernelSize, kernelSize };

                var handle = THSNN_MaxPool2d_ctor((IntPtr)pkernelSize, 2, (IntPtr)pStride, 2, (IntPtr)pPadding, 2, (IntPtr)pDilation, 2, ceilMode, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new MaxPool2d(handle, boxedHandle);
            }

            /// <summary>
            /// Applies a 2D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceilMode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            public static unsafe MaxPool2d MaxPool2d((long, long) kernelSize, (long, long)? stride = null, (long, long)? padding = null, (long, long)? dilation = null, bool ceilMode = false)
            {
                long svalue1 = stride != null ? stride.Value.Item1 : kernelSize.Item1;
                long svalue2 = stride != null ? stride.Value.Item2 : kernelSize.Item2;
                long pvalue1 = padding != null ? padding.Value.Item1 : 0;
                long pvalue2 = padding != null ? padding.Value.Item2 : 0;
                long dvalue1 = dilation != null ? dilation.Value.Item1 : 1;
                long dvalue2 = dilation != null ? dilation.Value.Item2 : 1;

                long* pStride = stackalloc long[2] { svalue1, svalue2 };
                long* pPadding = stackalloc long[2] { pvalue1, pvalue2 };
                long* pDilation = stackalloc long[2] { dvalue1, dvalue2 };

                long* pkernelSize = stackalloc long[2] { kernelSize.Item1, kernelSize.Item2 };

                var handle = THSNN_MaxPool2d_ctor((IntPtr)pkernelSize, 2, (IntPtr)pStride, 2, (IntPtr)pPadding, 2, (IntPtr)pDilation, 2, ceilMode, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new MaxPool2d(handle, boxedHandle);
            }

            /// <summary>
            /// Applies a 2D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
            /// <param name="strides">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceilMode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            public static unsafe MaxPool2d MaxPool2d(long[] kernelSize, long[] strides = null, long[] padding = null, long[] dilation = null, bool ceilMode = false)
            {
                fixed (long* pkernelSize = kernelSize, pstrides = strides, pPadding = padding, pDilation = dilation) {
                    var handle = THSNN_MaxPool2d_ctor((IntPtr)pkernelSize, kernelSize.Length, (IntPtr)pstrides, (strides == null ? 0 : strides.Length), (IntPtr)pPadding, (padding == null ? 0 : padding.Length), (IntPtr)pDilation, (dilation == null ? 0 : dilation.Length), ceilMode, out var boxedHandle);
                    if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new MaxPool2d(handle, boxedHandle);
                }
            }

            public static partial class functional
            {
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
                public static Tensor max_pool2d(Tensor input, long[] kernelSize, long[] strides = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    strides = strides ?? kernelSize.Select(x => 1L).ToArray();
                    padding = padding ?? kernelSize.Select(x => 0L).ToArray();
                    dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
                    unsafe {
                        fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                            var res =
                                THSTensor_max_pool2d(input.Handle,
                                    (IntPtr)pkernelSize, kernelSize.Length,
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
                /// <param name="kernelSize"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static unsafe Tensor max_pool2d(Tensor input, long kernelSize, long? stride = null,
                    long? padding = null, long? dilation = null, bool ceil_mode = false)
                {
                    long svalue = stride.HasValue ? stride.Value : kernelSize;
                    long pvalue = padding.HasValue ? padding.Value : 0;
                    long dvalue = dilation.HasValue ? dilation.Value : 1;

                    long* pStride = stackalloc long[2] { svalue, svalue };
                    long* pPadding = stackalloc long[2] { pvalue, pvalue };
                    long* pDilation = stackalloc long[2] { dvalue, dvalue };

                    long* pkernelSize = stackalloc long[2] { kernelSize, kernelSize };

                    var res = THSTensor_max_pool2d(input.Handle,
                                    (IntPtr)pkernelSize, 2,
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
                /// <param name="kernelSize"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static unsafe Tensor max_pool2d(Tensor input, (long, long) kernelSize, (long, long)? stride = null,
                    (long, long)? padding = null, (long, long)? dilation = null, bool ceil_mode = false)
                {
                    long svalue1 = stride != null ? stride.Value.Item1 : kernelSize.Item1;
                    long svalue2 = stride != null ? stride.Value.Item2 : kernelSize.Item2;
                    long pvalue1 = padding != null ? padding.Value.Item1 : 0;
                    long pvalue2 = padding != null ? padding.Value.Item2 : 0;
                    long dvalue1 = dilation != null ? dilation.Value.Item1 : 1;
                    long dvalue2 = dilation != null ? dilation.Value.Item2 : 1;

                    long* pStride = stackalloc long[2] { svalue1, svalue2 };
                    long* pPadding = stackalloc long[2] { pvalue1, pvalue2 };
                    long* pDilation = stackalloc long[2] { dvalue1, dvalue2 };

                    long* pkernelSize = stackalloc long[2] { kernelSize.Item1, kernelSize.Item2 };

                    var res = THSTensor_max_pool2d(input.Handle,
                                    (IntPtr)pkernelSize, 2,
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
                /// <param name="kernelSize"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static (Tensor output, Tensor indices) max_pool2d_with_indices(Tensor input, long[] kernelSize, long[] strides = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    strides = strides ?? kernelSize.Select(x => 1L).ToArray();
                    padding = padding ?? kernelSize.Select(x => 0L).ToArray();
                    dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        unsafe {
                            fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                                THSTensor_max_pool2d_with_indices(input.Handle,
                                    pa.CreateArray,
                                    (IntPtr)pkernelSize, kernelSize.Length,
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
