// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

#nullable enable
namespace TorchSharp
{
    using System.Security.Cryptography;
    using Modules;

    namespace Modules
    {
        public sealed class Fold : torch.nn.Module<Tensor, Tensor>
        {
            internal Fold((long, long) output_size, (long, long) kernel_size, (long, long) dilation, (long, long) padding, (long, long) stride) : base(nameof(Fold))
            {
                this.outputSize = output_size;
                this.kernelSize = kernel_size;
                this.dilation = dilation;
                this.padding = padding;
                this.stride = stride;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.fold(tensor, outputSize , kernelSize, dilation, padding, stride);
            }

            private (long, long) outputSize;
            private (long, long) kernelSize;
            private (long, long) dilation;
            private (long, long) padding;
            private (long, long) stride;
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Combines an array of sliding local blocks into a large containing tensor.
            /// </summary>
            /// <param name="output_size">Describes the spatial shape of the large containing tensor of the sliding local blocks.</param>
            /// <param name="kernel_size">The size of the sliding blocks</param>
            /// <param name="dilation">A parameter that controls the stride of elements within the neighborhood.</param>
            /// <param name="padding">Implicit zero padding to be added on both sides of input.</param>
            /// <param name="stride">The stride of the sliding blocks in the input spatial dimensions.</param>
            /// <remarks>Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.</remarks>
            public unsafe static Fold Fold(long output_size, long kernel_size, long dilation = 1, long padding = 0, long stride = 1)
            {
                return new Fold((output_size, output_size), (kernel_size, kernel_size), (dilation, dilation), (padding, padding), (stride, stride));
            }

            /// <summary>
            /// Combines an array of sliding local blocks into a large containing tensor.
            /// </summary>
            /// <param name="output_size">Describes the spatial shape of the large containing tensor of the sliding local blocks.</param>
            /// <param name="kernel_size">The size of the sliding blocks</param>
            /// <param name="dilation">A parameter that controls the stride of elements within the neighborhood.</param>
            /// <param name="padding">Implicit zero padding to be added on both sides of input.</param>
            /// <param name="stride">The stride of the sliding blocks in the input spatial dimensions.</param>
            /// <remarks>Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.</remarks>
            public unsafe static Fold Fold((long, long) output_size, (long, long) kernel_size, (long, long)? dilation = null, (long, long)? padding = null, (long, long)? stride = null)
            {
                dilation ??= (1, 1);
                stride ??= (1, 1);
                padding ??= (0, 0);

                return new Fold(output_size, kernel_size, dilation.Value, padding.Value, stride.Value);
            }

            public static partial class functional
            {
                /// <summary>
                /// Combines an array of sliding local blocks into a large containing tensor.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="output_size">Describes the spatial shape of the large containing tensor of the sliding local blocks.</param>
                /// <param name="kernel_size">The size of the sliding blocks</param>
                /// <param name="dilation">A parameter that controls the stride of elements within the neighborhood.</param>
                /// <param name="padding">Implicit zero padding to be added on both sides of input.</param>
                /// <param name="stride">The stride of the sliding blocks in the input spatial dimensions.</param>
                /// <remarks>Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.</remarks>
                public unsafe static Tensor fold(Tensor input, long output_size, long kernel_size, long dilation = 1, long padding = 0, long stride = 1)
                {
                    var res = THSNN_fold(input.Handle, output_size, output_size, kernel_size, kernel_size, stride, stride, padding, padding, dilation, dilation);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                /// <summary>
                /// Combines an array of sliding local blocks into a large containing tensor.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="output_size">Describes the spatial shape of the large containing tensor of the sliding local blocks.</param>
                /// <param name="kernel_size">The size of the sliding blocks</param>
                /// <param name="dilation">A parameter that controls the stride of elements within the neighborhood.</param>
                /// <param name="padding">Implicit zero padding to be added on both sides of input.</param>
                /// <param name="stride">The stride of the sliding blocks in the input spatial dimensions.</param>
                /// <remarks>Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.</remarks>
                public unsafe static Tensor fold(Tensor input, (long,long) output_size, (long, long) kernel_size, (long, long)? dilation = null, (long, long)? padding = null, (long, long)? stride = null)
                {
                    dilation ??= (1, 1);
                    stride ??= (1, 1);
                    padding ??= (0, 0);

                    var res = THSNN_fold(input.Handle,
                        output_size.Item1, output_size.Item2,
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
