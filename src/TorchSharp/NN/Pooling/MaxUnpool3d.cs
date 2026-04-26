// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Microsoft.VisualBasic;
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a MaxUnpool3D module.
        /// </summary>
        public sealed class MaxUnpool3d : ParameterLessModule<Tensor, Tensor, long[], Tensor>
        {
            internal MaxUnpool3d(long[] kernel_size, long[] stride = null, long[] padding = null) : base(nameof(MaxUnpool3d))
            {
                this.kernel_size = kernel_size;
                this.stride = stride;
                this.padding = padding;
            }

            public override Tensor forward(Tensor tensor, Tensor indices, long[] output_size = null)
            {
                return torch.nn.functional.max_unpool3d(tensor, indices, kernel_size, stride, padding, output_size);
            }

            public new Tensor call(Tensor tensor, Tensor indices, long[] output_size = null)
            {
                return base.call(tensor, indices, output_size);
            }

            public long[] kernel_size { get; set; }
            public long[] stride { get; set; }
            public long[] padding { get; set; }
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
            /// <returns></returns>
            public static MaxUnpool3d MaxUnpool3d(long kernel_size, long? stride = null, long? padding = null)
            {
                var pStride = stride.HasValue ? new long[] { stride.Value, stride.Value, stride.Value } : null;
                var pPadding = padding.HasValue ? new long[] { padding.Value, padding.Value, padding.Value } : null;
                return new MaxUnpool3d(new[] { kernel_size, kernel_size, kernel_size }, pStride, pPadding);
            }

            /// <summary>
            /// Applies a 3D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <returns></returns>
            public static MaxUnpool3d MaxUnpool3d((long, long, long) kernel_size, (long, long, long)? stride = null, (long, long, long)? padding = null)
            {
                var pStride = stride.HasValue ? new long[] { stride.Value.Item1, stride.Value.Item2, stride.Value.Item3 } : null;
                var pPadding = padding.HasValue ? new long[] { padding.Value.Item1, padding.Value.Item2, padding.Value.Item3 } : null;
                return new MaxUnpool3d(new[] { kernel_size.Item1, kernel_size.Item2, kernel_size.Item3 }, pStride, pPadding);
            }

            /// <summary>
            /// Applies a 3D max pooling over an input signal composed of several input planes.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <returns></returns>
            public static MaxUnpool3d MaxUnpool3d(long[] kernel_size, long[] stride = null, long[] padding = null)
            {
                return new MaxUnpool3d(kernel_size, stride, padding);
            }

            public static partial class functional
            {
                /// <summary>
                /// Computes a partial inverse of MaxPool3d.
                /// </summary>
                /// <param name="input">the input Tensor to invert</param>
                /// <param name="indices">the indices given out by :class:`~torch.nn.MaxPool3d`</param>
                /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
                /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
                /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
                /// <param name="output_size">(optional): The targeted output size</param>
                /// <returns></returns>
                public static Tensor max_unpool3d(Tensor input, Tensor indices, long[] kernel_size, long[] stride = null, long[] padding = null, long[] output_size = null)
                {
                    stride ??= Array.Empty<long>();
                    padding ??= Array.Empty<long>();
                    output_size ??= Array.Empty<long>();

                    unsafe {
                        fixed (long* pkernels = kernel_size, pstrides = stride, ppaddings = padding, poutputSize = output_size) {
                            var res = THSTensor_max_unpool3d(input.Handle, indices.Handle, (IntPtr)pkernels, kernel_size.Length, (IntPtr)poutputSize, output_size.Length, (IntPtr)ppaddings, padding.Length, (IntPtr)pstrides, stride.Length);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }
            }
        }
    }
}
