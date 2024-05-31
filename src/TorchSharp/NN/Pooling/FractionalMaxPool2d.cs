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
        /// This class is used to represent a FractionalMaxPool2D module.
        /// </summary>
        public sealed class FractionalMaxPool2d : torch.nn.Module<Tensor, Tensor>
        {
            internal FractionalMaxPool2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_FractionalMaxPool2d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public (Tensor Values, Tensor Indices) forward_with_indices(Tensor tensor)
            {
                var res = THSNN_FractionalMaxPool2d_forward_with_indices(handle, tensor.Handle, out var indices);
                if (res == IntPtr.Zero || indices == IntPtr.Zero) { torch.CheckForErrors(); }
                return (new Tensor(res), new Tensor(indices));
            }

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 2D fractional max pooling over an input signal composed of several input planes.
            ///
            /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
            /// see: https://arxiv.org/abs/1412.6071
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
            /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
            /// <returns></returns>
            public static FractionalMaxPool2d FractionalMaxPool2d(long kernel_size, long? output_size = null, double? output_ratio = null)
            {
                var pSize = output_size.HasValue ? new long[] { output_size.Value, output_size.Value } : null;
                var pRatio = output_ratio.HasValue ? new double[] { output_ratio.Value, output_ratio.Value } : null;
                return FractionalMaxPool2d(new long[] { kernel_size, kernel_size }, pSize, pRatio);
            }

            /// <summary>
            /// Applies a 2D fractional max pooling over an input signal composed of several input planes.
            ///
            /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
            /// see: https://arxiv.org/abs/1412.6071
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
            /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
            /// <returns></returns>
            public static FractionalMaxPool2d FractionalMaxPool2d((long, long) kernel_size, (long, long)? output_size = null, (double, double)? output_ratio = null)
            {
                var pSize = output_size.HasValue ? new long[] { output_size.Value.Item1, output_size.Value.Item2 } : null;
                var pRatio = output_ratio.HasValue ? new double[] { output_ratio.Value.Item1, output_ratio.Value.Item2 } : null;
                return FractionalMaxPool2d(new long[] { kernel_size.Item1, kernel_size.Item2 }, pSize, pRatio);
            }

            /// <summary>
            /// Applies a 2D fractional max pooling over an input signal composed of several input planes.
            ///
            /// Fractional MaxPooling is described in detail in the paper Fractional MaxPooling by Ben Graham,
            /// see: https://arxiv.org/abs/1412.6071
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="output_size">The target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH</param>
            /// <param name="output_ratio">If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)</param>
            /// <returns></returns>
            public static FractionalMaxPool2d FractionalMaxPool2d(long[] kernel_size, long[] output_size = null, double[] output_ratio = null)
            {
                if (kernel_size == null || kernel_size.Length != 2)
                    throw new ArgumentException("Kernel size must contain two elements.");
                if (output_size != null && output_size.Length != 2)
                    throw new ArgumentException("output_size must contain two elements.");
                if (output_ratio != null && output_ratio.Length != 2)
                    throw new ArgumentException("output_ratio must contain two elements.");
                if (output_size == null && output_ratio == null)
                    throw new ArgumentNullException("Only one of output_size and output_ratio may be specified.");
                if (output_size != null && output_ratio != null)
                    throw new ArgumentNullException("FractionalMaxPool2d requires specifying either an output size, or a pooling ratio.");

                unsafe {
                    fixed (long* pkernelSize = kernel_size, pSize = output_size) {
                        fixed (double* pRatio = output_ratio) {
                            var handle = THSNN_FractionalMaxPool2d_ctor(
                                (IntPtr)pkernelSize, kernel_size.Length,
                                (IntPtr)pSize, (output_size == null ? 0 : output_size.Length),
                                (IntPtr)pRatio, (output_ratio == null ? 0 : output_ratio.Length),
                                out var boxedHandle);
                            if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new FractionalMaxPool2d(handle, boxedHandle);
                        }
                    }
                }
            }
        }
    }
}
