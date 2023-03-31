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
        /// This class is used to represent a MaxPool1D module.
        /// </summary>
        public sealed class MaxPool1d : torch.nn.Module<Tensor, Tensor>
        {
            internal MaxPool1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_MaxPool1d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public (Tensor Values, Tensor Indices) forward_with_indices(Tensor tensor)
            {
                var res = THSNN_MaxPool1d_forward_with_indices(handle, tensor.Handle, out var indices);
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
            /// Applies a 1D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernelSize">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <param name="dilation">The stride between elements within a sliding window, must be > 0.</param>
            /// <param name="ceilMode">If true, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.</param>
            /// <returns></returns>
            public static MaxPool1d MaxPool1d(long kernelSize, long? stride = null, long? padding = null, long? dilation = null, bool ceilMode = false)
            {
                var pStride = stride.HasValue ? new long[] { stride.Value } : null;
                var pPadding = padding.HasValue ? new long[] { padding.Value } : null;
                var pDilation = dilation.HasValue ? new long[] { dilation.Value } : null;
                return MaxPool1d(new long[] { kernelSize }, pStride, pPadding, pDilation, ceilMode);
            }

            private static MaxPool1d MaxPool1d(long[] kernelSize, long[] strides = null, long[] padding = null, long[] dilation = null, bool ceilMode = false)
            {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, pPadding = padding, pDilation = dilation) {
                        var handle = THSNN_MaxPool1d_ctor((IntPtr)pkernelSize, (IntPtr)pstrides, (IntPtr)pPadding, (IntPtr)pDilation, ceilMode, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new MaxPool1d(handle, boxedHandle);
                    }
                }
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 1D max pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSize"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static Tensor max_pool1d(Tensor input, long kernelSize, long? stride = null,
                    long? padding = null, long? dilation = null, bool ceil_mode = false)
                {
                    var kernelSizes = new long[] { kernelSize };
                    var strides = new long[] { stride ?? 1 };
                    var paddings = new long[] { padding ?? 0 };
                    var dilations = new long[] { dilation ?? 1 };
                    unsafe {
                        fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings, pdilation = dilations) {
                            var res =
                                THSTensor_max_pool1d(input.Handle,
                                    (IntPtr)pkernelSize, kernelSizes.Length,
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
                /// <param name="kernelSize"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static (Tensor output, Tensor indices) max_pool1d_with_indices(Tensor input, long kernelSize, long? stride = null,
                    long? padding = null, long? dilation = null, bool ceil_mode = false)
                {
                    var kernelSizes = new long[] { kernelSize };
                    var strides = new long[] { stride ?? 1 };
                    var paddings = new long[] { padding ?? 0 };
                    var dilations = new long[] { dilation ?? 1 };
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        unsafe {
                            fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings, pdilation = dilations) {
                                THSTensor_max_pool1d_with_indices(input.Handle,
                                    pa.CreateArray,
                                    (IntPtr)pkernelSize, kernelSizes.Length,
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
