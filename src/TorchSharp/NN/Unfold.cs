// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class Unfold : ParameterLessModule<Tensor, Tensor>
        {
            internal Unfold((long, long) kernel_size, (long, long) dilation, (long, long) padding, (long, long) stride) : base(nameof(Unfold))
            {
                this.kernel_size = kernel_size;
                this.dilation = dilation;
                this.padding = padding;
                this.stride = stride;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.unfold(tensor, kernel_size, dilation, padding, stride);
            }

            public (long, long) kernel_size { get; set; }
            public (long, long) dilation { get; set; }
            public (long, long) padding { get; set; }
            public (long, long) stride { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Extracts sliding local blocks from a batched input tensor.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding blocks</param>
            /// <param name="dilation">A parameter that controls the stride of elements within the neighborhood.</param>
            /// <param name="padding">Implicit zero padding to be added on both sides of input.</param>
            /// <param name="stride">The stride of the sliding blocks in the input spatial dimensions.</param>
            /// <remarks>Currently, only 4-D input tensors (batched image-like tensors) are supported.</remarks>
            public static Unfold Unfold(long kernel_size, long dilation = 1, long padding = 0, long stride = 1)
            {
                return new Unfold((kernel_size, kernel_size), (dilation, dilation), (padding, padding), (stride, stride));
            }

            /// <summary>
            /// Extracts sliding local blocks from a batched input tensor.
            /// </summary>
            /// <param name="kernel_size">The size of the sliding blocks</param>
            /// <param name="dilation">A parameter that controls the stride of elements within the neighborhood.</param>
            /// <param name="padding">Implicit zero padding to be added on both sides of input.</param>
            /// <param name="stride">The stride of the sliding blocks in the input spatial dimensions.</param>
            /// <remarks>Currently, only 4-D input tensors (batched image-like tensors) are supported.</remarks>
            public static Unfold Unfold((long, long) kernel_size, (long, long)? dilation = null, (long, long)? padding = null, (long, long)? stride = null)
            {
                dilation ??= (1, 1);
                stride ??= (1, 1);
                padding ??= (0, 0);

                return new Unfold(kernel_size, dilation.Value, padding.Value, stride.Value);
            }

            public static partial class functional
            {
                /// <summary>
                /// Extracts sliding local blocks from a batched input tensor.
                /// </summary>
                /// <param name="input">The input tensor. Must be 4-D (batched images).</param>
                /// <param name="kernel_size">The size of the sliding blocks</param>
                /// <param name="dilation">A parameter that controls the stride of elements within the neighborhood.</param>
                /// <param name="padding">Implicit zero padding to be added on both sides of input.</param>
                /// <param name="stride">The stride of the sliding blocks in the input spatial dimensions.</param>
                public static Tensor unfold(Tensor input, long kernel_size, long dilation = 1, long padding = 0, long stride = 1)
                {
                    var res = THSNN_unfold(input.Handle, kernel_size, kernel_size, stride, stride, padding, padding, dilation, dilation);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Extracts sliding local blocks from a batched input tensor.
                /// </summary>
                /// <param name="input">The input tensor. Must be 4-D (batched images).</param>
                /// <param name="kernel_size">The size of the sliding blocks</param>
                /// <param name="dilation">A parameter that controls the stride of elements within the neighborhood.</param>
                /// <param name="padding">Implicit zero padding to be added on both sides of input.</param>
                /// <param name="stride">The stride of the sliding blocks in the input spatial dimensions.</param>
                public static Tensor unfold(Tensor input, (long, long) kernel_size, (long, long)? dilation = null, (long, long)? padding = null, (long, long)? stride = null)
                {
                    dilation ??= (1, 1);
                    stride ??= (1, 1);
                    padding ??= (0, 0);

                    var res = THSNN_unfold(input.Handle,
                        kernel_size.Item1, kernel_size.Item2,
                        stride.Value.Item1, stride.Value.Item2,
                        padding.Value.Item1, padding.Value.Item2,
                        dilation.Value.Item1, dilation.Value.Item2);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
