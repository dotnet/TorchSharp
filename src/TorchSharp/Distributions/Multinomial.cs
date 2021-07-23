using System;
using System.Collections.Generic;
using System.Linq;

using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class Multinomial : torch.distributions.Distribution
        {

            public override Tensor mean => total_count * probs;

            public override Tensor variance => total_count * probs * (1 - probs);

            public Multinomial(int total_count, Tensor p = null, Tensor l = null) 
            {
                this.total_count = total_count;
                this.categorical = new Categorical(p, l);
                this.batch_shape = this.categorical.batch_shape;
                var ps = this.categorical.param_shape;
                this.event_shape = new long[] { ps[ps.Length-1] };
            }

            private Multinomial(int total_count, Categorical categorical)
            {
                this.total_count = total_count;
                this.categorical = categorical;
                this.batch_shape = categorical.batch_shape;
                var ps = categorical.param_shape;
                this.event_shape = new long[] { ps[ps.Length - 1] };
            }

            public Tensor probs => categorical.probs;

            public Tensor logits => categorical.logits;


            public long[] param_shape => categorical.param_shape;

            private int total_count;
            private Categorical categorical;

            public override Tensor rsample(params long[] sample_shape)
            {
                var cShape = new List<long>(); cShape.Add(total_count); cShape.AddRange(sample_shape);

                var samples = categorical.sample(cShape.ToArray());
                var shifted_idx = Enumerable.Range(0, (int)samples.dim()).ToList();
                var tc = shifted_idx[0];
                shifted_idx.RemoveAt(0);
                shifted_idx.Add(tc);
                samples = samples.permute(shifted_idx.Select(i => (long)i).ToArray());
                var counts = samples.new_zeros(ExtendedShape(sample_shape));
                counts.scatter_add_(-1, samples, torch.ones_like(samples));
                return counts.type_as(probs);
            }

            public override Tensor log_prob(Tensor value)
            {
                var bcast = torch.broadcast_tensors(logits, value);
                var l = bcast[0].clone();
                value = bcast[1];
                var log_factorial_n = torch.lgamma(value.sum(-1) + 1);
                var log_factorial_xs = torch.lgamma(value + 1).sum(-1);
                l[(value == 0) & (l == float.NegativeInfinity)] = torch.tensor(0.0f);
                var log_powers = (logits * value).sum(-1);
                return log_factorial_n - log_factorial_xs + log_powers;
            }

            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Multinomial))
                    throw new ArgumentException("expand(): 'instance' must be a Multinomial distribution");

                var newDistribution = ((instance == null) ?
                    new Multinomial(total_count, categorical.expand(batch_shape) as Categorical) :
                    instance) as Multinomial;

                if (newDistribution == instance) {
                    newDistribution.total_count = total_count;
                    newDistribution.categorical = categorical.expand(batch_shape) as Categorical;
                    newDistribution.batch_shape = newDistribution.categorical.batch_shape;
                    var ps = newDistribution.categorical.param_shape;
                    newDistribution.event_shape = new long[] { ps[ps.Length - 1] };
                }

                return newDistribution;
            }

            public override Tensor entropy()
            {
                throw new NotImplementedException();
            }
        }

    }
    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a Multinomial distribution parameterized by `probs` or `logits` (but not both).
            /// `total_count` must be broadcastable with `probs`/`logits`.
            /// </summary>
            /// <param name="total_count">Number of Bernoulli trials</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <returns></returns>
            public static Multinomial Multinomial(int total_count, Tensor probs = null, Tensor logits = null)
            {
                return new Multinomial(total_count, probs, logits);
            }
        }
    }
}
