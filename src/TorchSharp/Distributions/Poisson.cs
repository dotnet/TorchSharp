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
        public class Poisson : torch.distributions.ExponentialFamily
        {

            public override Tensor mean => rate;

            public override Tensor variance => rate;

            public Poisson(Tensor rate)
            {
                var locScale = torch.broadcast_tensors(rate);
                this.batch_shape = rate.size();
                this.rate = locScale[0];
            }

            private Tensor rate;

            public override Tensor rsample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                using(torch.no_grad())
                    return torch.poisson(rate.expand(shape));
            }

            public override Tensor log_prob(Tensor value)
            {
                var bcast = torch.broadcast_tensors(rate, value);
                var r = bcast[0];
                var v = bcast[1];
                return r.log() * value - r - (value + 1).lgamma();
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
                if (instance != null && !(instance is Poisson))
                    throw new ArgumentException("expand(): 'instance' must be a Poisson distribution");

                var r = rate.expand(batch_shape);

                var newDistribution = ((instance == null) ? new Poisson(r) : instance) as Poisson;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.rate = r;
                }
                return newDistribution;
            }

            protected override IList<Tensor> NaturalParams => new Tensor[] { rate.log() };

            protected override Tensor MeanCarrierMeasure => throw new NotImplementedException();

            protected override Tensor LogNormalizer(params Tensor[] parameters) => parameters[0].exp();
        }

    }

    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a Poisson distribution parameterized by `rate`.
            /// </summary>
            /// <param name="rate">rate = 1 / scale of the distribution (often referred to as 'β')</param>
            /// <returns></returns>
            public static Poisson Poisson(Tensor rate)
            {
                return new Poisson(rate);
            }
        }
    }
}
