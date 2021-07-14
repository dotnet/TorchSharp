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
        public class Gamma : torch.distributions.ExponentialFamily
        {

            public override Tensor mean => concentration / rate;

            public override Tensor variance => concentration / rate.pow(2);


            public Gamma(Tensor concentration, Tensor rate)
            {
                var locScale = torch.broadcast_tensors(concentration, rate);
                this.concentration = locScale[0];
                this.rate = locScale[1];
                this.batch_shape = this.concentration.size();
            }

            protected Tensor concentration;
            private Tensor rate;

            public override Tensor rsample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                var value = torch._standard_gamma(concentration.expand(shape)) / rate.expand(shape);
                return value.detach().clamp_(min: torch.finfo(value.dtype).tiny);
            }

            public override Tensor log_prob(Tensor value)
            {
                value = torch.as_tensor(value, dtype: rate.dtype, device: rate.device);
                return concentration * rate.log() + (concentration - 1) * value.log() - rate * value - torch.lgamma(concentration);
            }

            public override Tensor entropy()
            {
                return concentration - rate.log() + concentration.lgamma() + (1.0 - concentration) * concentration.digamma();
            }

            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Gamma))
                    throw new ArgumentException("expand(): 'instance' must be a Gamma distribution");

                var c = concentration.expand(batch_shape);
                var r = rate.expand(batch_shape);

                var newDistribution = ((instance == null) ? new Gamma(c, r) : instance) as Gamma;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.concentration = c;
                    newDistribution.rate = r;
                }
                return newDistribution;
            }

            protected override IList<Tensor> NaturalParams => new Tensor[] { concentration - 1, rate };

            protected override Tensor MeanCarrierMeasure => torch.tensor(0, dtype:rate.dtype, device: rate.device);

            protected override Tensor LogNormalizer(params Tensor[] parameters)
            {
                var x = parameters[0];
                var y = parameters[1];

                return torch.lgamma(x + 1) + (x + 1) * torch.log(-y.reciprocal());
            }
        }

    }

    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a Gamma distribution parameterized by shape `concentration` and `rate`.
            /// </summary>
            /// <param name="concentration">Shape parameter of the distribution (often referred to as 'α')</param>
            /// <param name="rate">rate = 1 / scale of the distribution (often referred to as 'β')</param>
            /// <returns></returns>
            public static Gamma Gamma(Tensor concentration, Tensor rate)
            {
                return new Gamma(concentration, rate);
            }
        }
    }
}
