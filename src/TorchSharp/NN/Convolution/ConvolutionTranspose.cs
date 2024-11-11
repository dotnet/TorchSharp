// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using System.Collections.Generic;
    using System.Linq;
    using Modules;
    using TorchSharp.Utils;

    namespace Modules
    {
        public abstract class ConvolutionTranspose : Convolution
        {
            protected ConvolutionTranspose(string name, long in_channels, long out_channels, long[] kernel_size, long[] stride, long[]? padding, Padding? padding_type, long[] dilation, bool transposed, long[] output_padding, long groups, bool bias, PaddingModes padding_mode, Device? device = null, ScalarType? dtype = null) : base(name, in_channels, out_channels, kernel_size, stride, padding, padding_type, dilation, transposed, output_padding, groups, bias, padding_mode, device, dtype)
            {
            }

            public override Tensor forward(Tensor input)
            {
                return this.forward(input, null);
            }
            public abstract Tensor forward(Tensor input, long[]? output_size);
            
            protected long[] _output_padding(Tensor input, long[]? output_size, long[] kernel_size, long[] stride, long[] padding, long[] dilation, long num_spatial_dims)
            {
                if (output_size is null)
                    return new[] { this.output_padding[0] };

                var hasBatchDim = input.dim() == num_spatial_dims + 2;
                var numNonSpatialDims = hasBatchDim ? 2 : 1;
                if (output_size.Length == numNonSpatialDims + num_spatial_dims)
                    output_size = output_size.Skip(numNonSpatialDims).ToArray();
                if (output_size.Length != num_spatial_dims)
                    throw new ArgumentException($"ConvTranspose{num_spatial_dims}D: for {input.dim()}D input, output_size must have {num_spatial_dims} or {numNonSpatialDims + num_spatial_dims} elements (got {output_size.Length})", nameof(input));

                var minSizes = new List<long>();
                var maxSizes = new List<long>();
                for (int d = 0; d < num_spatial_dims; d++) {
                    var dimSize = (input.size(d + numNonSpatialDims) - 1) * stride[d] - 2 * padding[d]
                                    + dilation[d] * (kernel_size[d] - 1) + 1;
                    minSizes.Add(dimSize);
                    maxSizes.Add(dimSize + stride[d] - 1);
                }

                for (int i = 0; i < output_size.Length; i++) {
                    var size = output_size[i];
                    var minSize = minSizes[i];
                    var maxSize = maxSizes[i];
                    if (size < minSize || size > maxSize)
                        throw new ArgumentException($"Requested an output size of {output_size}, but valid sizes range from {minSize} to {maxSize}");
                }

                return Enumerable.Range(0, (int)num_spatial_dims).Select(d => output_size[d] - minSizes[d]).ToArray();
            }

        }
    }
}