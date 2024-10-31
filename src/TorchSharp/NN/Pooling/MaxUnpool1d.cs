// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using System.Runtime.CompilerServices;
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a MaxUnpool1D module.
        /// </summary>
        public sealed class MaxUnpool1d : ParameterLessModule<Tensor, Tensor, long[], Tensor>
        {
            internal MaxUnpool1d(long kernel_size, long? stride = null, long? padding = null) : base(nameof(MaxUnpool1d))
            {
                this.kernel_size = kernel_size;
                this.stride = stride;
                this.padding = padding;
            }

            public override Tensor forward(Tensor tensor, Tensor indices, long[] output_size = null)
            {
                return torch.nn.functional.max_unpool1d(tensor, indices, kernel_size, stride, padding, output_size);
            }

            public new Tensor call(Tensor tensor, Tensor indices, long[] output_size = null)
            {
                return base.call(tensor, indices, output_size);
            }

            public long kernel_size { get; set; }
            public long? stride { get; set; }
            public long? padding { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Computes a partial inverse of :class:`MaxPool1d`.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
            /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
            /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
            /// <returns></returns>
            public static MaxUnpool1d MaxUnpool1d(long kernel_size, long? stride = null, long? padding = null)
            {
                return new MaxUnpool1d(kernel_size, stride, padding);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 1D max pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">the input Tensor to invert</param>
                /// <param name="indices">the indices given out by :class:`~torch.nn.MaxPool1d`</param>
                /// <param name="kernel_size">The size of the sliding window, must be > 0.</param>
                /// <param name="stride">The stride of the sliding window, must be > 0. Default value is kernel_size.</param>
                /// <param name="padding">Implicit negative infinity padding to be added on both sides, must be >= 0 and less than or equal to kernel_size / 2</param>
                /// <param name="output_size">(optional): The targeted output size</param>
                /// <returns></returns>
                public static Tensor max_unpool1d(Tensor input, Tensor indices, long kernel_size, long? stride = null, long? padding = null, long[] output_size = null)
                {
                    long[] kernels = new[] { kernel_size };
                    long[] strides = stride.HasValue ? new[] { stride.Value } : Array.Empty<long>();
                    long[] paddings = padding.HasValue ? new[] { padding.Value } : Array.Empty<long>();
                    output_size ??= Array.Empty<long>();

                    unsafe {
                        fixed (long* pkernels = kernels, pstrides = strides, ppaddings = paddings, poutputSize = output_size) {
                            var res = THSTensor_max_unpool1d(input.Handle, indices.Handle, (IntPtr)pkernels, kernels.Length, (IntPtr)poutputSize, output_size.Length, (IntPtr)ppaddings, paddings.Length, (IntPtr)pstrides, strides.Length);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }
            }
        }
    }
}
