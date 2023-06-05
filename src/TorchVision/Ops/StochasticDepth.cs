// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/a4f53308b2d0f1aa9191686e326f45c26053f686/torchvision/ops/stochastic_depth.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {

        public static partial class ops
        {
            public static StochasticDepth StochasticDepth(double p, StochasticDepth.Mode mode) => new StochasticDepth(p, mode);

            /// <summary>
            /// Implements the Stochastic Depth from “Deep Networks with Stochastic Depth” used for randomly dropping residual branches of residual architectures.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="p">Probability of the input to be zeroed.</param>
            /// <param name="mode">"batch" or "row". "batch" randomly zeroes the entire input, "row" zeroes randomly selected rows from the batch.</param>
            /// <param name="training">Apply stochastic depth if is True</param>
            public static Tensor stochastic_depth(Tensor input, double p, StochasticDepth.Mode mode, bool training = true)
            {
                if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p));
                if (!training || p == 0.0) return input.alias();

                var survival_rate = 1 - p;
                var size = new List<long>();

                if (mode == torchvision.StochasticDepth.Mode.Row) {
                    size.Add(input.shape[0]);
                    for (var i = 0; i < input.ndim-1; i++)
                        size.Add(1);
                } else {
                    for (var i = 0; i < input.ndim; i++)
                        size.Add(1);
                }

                var noise = torch.empty(size.ToArray(), dtype: input.dtype, device: input.device);
                noise.bernoulli_(survival_rate);

                if (survival_rate > 0) {
                    noise.div_(survival_rate);
                }
                return input * noise;
            }
        }

        /// <summary>
        /// Implements the Stochastic Depth from “Deep Networks with Stochastic Depth” used for randomly dropping residual branches of residual architectures.
        /// </summary>
        public class StochasticDepth : nn.Module<Tensor, Tensor>
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="p">Probability of the input to be zeroed.</param>
            /// <param name="mode">"batch" or "row". "batch" randomly zeroes the entire input, "row" zeroes randomly selected rows from the batch.</param>
            public StochasticDepth(double p, Mode mode) : base(nameof(StochasticDepth))
            {
                this.p = p;
                this.mode = mode;
            }

            public override Tensor forward(Tensor input)
            {
                return ops.stochastic_depth(input, p, mode, training);
            }

            public enum Mode { Batch, Row }

            private double p;
            private Mode mode;
        }
    }
}
