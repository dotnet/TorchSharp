// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/a4f53308b2d0f1aa9191686e326f45c26053f686/torchvision/ops/drop_block.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {

        public static partial class ops
        {
            /// <summary>
            /// Implements DropBlock2d from `"DropBlock: A regularization method for convolutional networks"
            /// https://arxiv.org/abs/1810.12890
            /// </summary>
            /// <param name="input">The input tensor of 4 dimensions with the first one beinng its batch.</param>
            /// <param name="p">Probability of an element to be dropped.</param>
            /// <param name="block_size">Size of the block to drop.</param>
            /// <param name="inplace">If set to true, will do this operation in-place</param>
            /// <param name="eps">A small value added to the denominator for numerical stability.</param>
            /// <param name="training">Apply dropblock if is true</param>
            /// <returns>The randomly zeroed [N, C, H, W] tensor after dropblock</returns>
            /// <exception cref="ArgumentOutOfRangeException">If p is not in the range [0,1]</exception>
            /// <exception cref="ArgumentException">If the input tensor is not 4-dimensional</exception>
            public static Tensor drop_block2d(Tensor input, double p, long block_size, bool inplace = false, double eps = 1e-6, bool training = true)
            {
                if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p));
                if (input.ndim != 4) throw new ArgumentException($"input should be 4 dimensional. Got {input.ndim} dimensions.");
                if (!training || p == 0) return input.alias();

                var (N, C, H, W) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);

                block_size = Math.Min(block_size, Math.Min(W, H));
                // compute the gamma of Bernoulli distribution
                var gamma = (p * H * W) / ((block_size * block_size) * ((H - block_size + 1) * (W - block_size + 1)));
                var noise = torch.empty(N, C, H - block_size + 1, W - block_size + 1, dtype: input.dtype, device: input.device);
                noise.bernoulli_(gamma);

                var pad = block_size / 2;
                noise = torch.nn.functional.pad(noise, (pad, pad, pad, pad), value: 0);
                noise = torch.nn.functional.max_pool2d(noise, stride: 1, kernel_size: block_size, padding: block_size / 2);
                noise = 1 - noise;

                var normalize_scale = noise.numel() / (eps + noise.sum());
                if (inplace)
                    input.mul_(noise).mul_(normalize_scale);
                else
                    input = input * noise * normalize_scale;
                return input;
            }

            /// <summary>
            /// Implements DropBlock3d from `"DropBlock: A regularization method for convolutional networks"
            /// https://arxiv.org/abs/1810.12890
            /// </summary>
            /// <param name="input">The input tensor of 5 dimensions with the first one beinng its batch.</param>
            /// <param name="p">Probability of an element to be dropped.</param>
            /// <param name="block_size">Size of the block to drop.</param>
            /// <param name="inplace">If set to true, will do this operation in-place</param>
            /// <param name="eps">A small value added to the denominator for numerical stability.</param>
            /// <param name="training">Apply dropblock if is true</param>
            /// <returns>The randomly zeroed [N, C, D, H, W] tensor after dropblock</returns>
            /// <exception cref="ArgumentOutOfRangeException">If p is not in the range [0,1]</exception>
            /// <exception cref="ArgumentException">If the input tensor is not 5-dimensional</exception>
            public static Tensor drop_block3d(Tensor input, double p, long block_size, bool inplace = false, double eps = 1e-6, bool training = true)
            {
                if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p));
                if (input.ndim != 5) throw new ArgumentException($"input should be 5-dimensional. Got {input.ndim} dimensions.");
                if (!training || p == 0) return input.alias();

                var (N, C, D, H, W) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3], input.shape[4]);

                block_size = Math.Min(Math.Min(block_size, D), Math.Min(W, H));
                // compute the gamma of Bernoulli distribution
                var gamma = (p * D * H * W) / ((block_size * block_size * block_size) * ((D - block_size + 1) * (H - block_size + 1) * (W - block_size + 1)));
                var noise = torch.empty(new[] { N, C, D - block_size + 1, H - block_size + 1, W - block_size + 1 }, dtype: input.dtype, device: input.device);
                noise.bernoulli_(gamma);

                var pad = block_size / 2;
                var padding = new[] { pad, pad, pad, pad, pad, pad };
                noise = torch.nn.functional.pad(noise, padding, value: 0);
                noise = torch.nn.functional.max_pool3d(noise, strides: new long[] { 1, 1, 1 }, kernel_size: new[] { block_size, block_size, block_size }, padding: new long[] { pad });
                noise = 1 - noise;

                var normalize_scale = noise.numel() / (eps + noise.sum());
                if (inplace)
                    input.mul_(noise).mul_(normalize_scale);
                else
                    input = input * noise * normalize_scale;
                return input;
            }

            /// <summary>
            /// Implements DropBlock2d from `"DropBlock: A regularization method for convolutional networks"
            /// https://arxiv.org/abs/1810.12890
            /// </summary>
            public static DropBlock2d DropBlock2d(double p, long block_size, bool inplace = false, double eps = 1e-6) => new DropBlock2d(p, block_size, inplace, eps);

            /// <summary>
            /// Implements DropBlock3d from `"DropBlock: A regularization method for convolutional networks"
            /// https://arxiv.org/abs/1810.12890
            /// </summary>
            public static DropBlock3d DropBlock3d(double p, long block_size, bool inplace = false, double eps = 1e-6) => new DropBlock3d(p, block_size, inplace, eps);
        }
    }

    namespace Modules
    {
        public class DropBlock2d : ParameterLessModule<Tensor,Tensor>
        {
            public DropBlock2d(double p, long block_size, bool inplace = false, double eps = 1e-6) : base(nameof(DropBlock2d))
            {
                this.p = p;
                this.block_size = block_size;
                this.inplace = inplace;
                this.eps = eps;
            }


            public override Tensor forward(Tensor input)
            {
                return torchvision.ops.drop_block2d(input, p, block_size, inplace, eps, training);
            }

            private bool inplace;
            private double p;
            private long block_size;
            private double eps;
        }

        public class DropBlock3d : ParameterLessModule<Tensor, Tensor>
        {
            public DropBlock3d(double p, long block_size, bool inplace = false, double eps = 1e-6) : base(nameof(DropBlock3d))
            {
                this.p = p;
                this.block_size = block_size;
                this.inplace = inplace;
                this.eps = eps;
            }


            public override Tensor forward(Tensor input)
            {
                return torchvision.ops.drop_block3d(input, p, block_size, inplace, eps, training);
            }

            private bool inplace;
            private double p;
            private long block_size;
            private double eps;
        }
    }
}
