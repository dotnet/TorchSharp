// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Text;

using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class Uniform : torch.distributions.Distribution
        {

            public override Tensor mean => (high - low) / 2;

            public override Tensor variance => (high - low).pow(2) / 12;

            public Uniform(Tensor low, Tensor high, torch.Generator generator = null) : base(generator, low.size())
            {
                var lowHigh = torch.broadcast_tensors(low, high);
                this.low = lowHigh[0];
                this.high = lowHigh[1];
            }

            private Tensor high;
            private Tensor low;

            public override Tensor rsample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                var rand = torch.rand(shape, dtype: low.dtype, device: low.device, generator: generator);
                return low + rand * (high - low);
            }

            public override Tensor log_prob(Tensor value)
            {
                var lb = low.le(value).type_as(low);
                var ub = high.gt(value).type_as(low);
                return torch.log(lb.mul(ub)) - torch.log(high - low);
            }

            public override Tensor cdf(Tensor value)
            {
                return (value - low) / (high - low).clamp(0, 1);
            }

            public override Tensor icdf(Tensor value)
            {
                return value * (high - low) + low;
            }

            public override Tensor entropy()
            {
                return (high - low).log();
            }

            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Uniform))
                    throw new ArgumentException("expand(): 'instance' must be a Uniform distribution");

                var newDistribution = ((instance == null) ?
                    new Uniform(low.expand(batch_shape), high.expand(batch_shape)) :
                    instance) as Uniform;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance)
                {
                    newDistribution.low = low.expand(batch_shape);
                    newDistribution.high = high.expand(batch_shape);
                }
                return newDistribution;
            }
        }

    }
    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Generates uniformly distributed random samples from the half-open interval [low, high[.
            /// </summary>
            /// <param name="low">Lower bound (inclusive)</param>
            /// <param name="high">Upper bound (exclusive)</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Uniform Uniform(Tensor low, Tensor high, torch.Generator generator = null)
            {
                return new Uniform(low, high, generator);
            }
        }
    }
}
