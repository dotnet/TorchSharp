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
        public class Exponential : torch.distributions.ExponentialFamily
        {

            public override Tensor mean => rate.reciprocal();

            public override Tensor variance => rate.pow(2);

            public override Tensor stddev => rate.reciprocal();


            public Exponential(Tensor rate)
            {
                var locScale = torch.broadcast_tensors(rate);
                this.batch_shape = rate.size();
                this.rate = locScale[0];
            }

            private Tensor rate;

            public override Tensor rsample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                return rate.new_empty(shape).exponential_() / rate;
            }

            public override Tensor log_prob(Tensor value)
            {
                return rate.log() - rate * value;
            }

            public override Tensor entropy()
            {
                return 1 - rate.log();
            }

            public override Tensor cdf(Tensor value)
            {
                return 1 - torch.exp(-rate * value);
            }

            public override Tensor icdf(Tensor value)
            {
                return -torch.log(1 - value) / rate;
            }

            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Exponential))
                    throw new ArgumentException("expand(): 'instance' must be a Exponential distribution");

                var r = rate.expand(batch_shape);

                var newDistribution = ((instance == null) ? new Exponential(r) : instance) as Exponential;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.rate = r;
                }
                return newDistribution;
            }

            protected override IList<Tensor> NaturalParams => new Tensor[] { -rate };

            protected override Tensor MeanCarrierMeasure => torch.tensor(0, dtype:rate.dtype, device: rate.device);

            protected override Tensor LogNormalizer(params Tensor[] parameters) => -torch.log(-parameters[0]);
        }

    }

    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a Exponential distribution parameterized by `rate`.
            /// </summary>
            /// <param name="rate">rate = 1 / scale of the distribution (often referred to as 'β')</param>
            /// <returns></returns>
            public static Exponential Exponential(Tensor rate)
            {
                return new Exponential(rate);
            }
        }
    }
}
