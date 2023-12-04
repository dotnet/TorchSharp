// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A Multinomial distribution parameterized by `probs` or `logits` (but not both).
        /// `total_count` must be broadcastable with `probs`/`logits`.
        /// </summary>
        public class Multinomial : torch.distributions.Distribution
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => total_count * probs;

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance => WrappedTensorDisposeScope(() => total_count * probs * (1 - probs));

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="total_count">Number of Bernoulli trials</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Multinomial(int total_count, Tensor probs = null, Tensor logits = null, torch.Generator generator = null) : base(generator) 
            {
                this.total_count = total_count;
                this.categorical = new Categorical(probs, logits);
                this.batch_shape = this.categorical.batch_shape;
                var ps = this.categorical.param_shape;
                this.event_shape = new long[] { ps[ps.Length-1] };
            }

            private Multinomial(int total_count, Categorical categorical, torch.Generator generator = null) : base(generator)
            {
                this.total_count = total_count;
                this.categorical = categorical;
                this.batch_shape = categorical.batch_shape;
                var ps = categorical.param_shape;
                this.event_shape = new long[] { ps[ps.Length - 1] };
            }

            /// <summary>
            /// Event probabilities
            /// </summary>
            public Tensor probs => categorical.probs;

            /// <summary>
            /// Event log-odds
            /// </summary>
            public Tensor logits => categorical.logits;


            public long[] param_shape => categorical.param_shape;

            private int total_count;
            private Categorical categorical;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    categorical?.Dispose();
                }
                base.Dispose(disposing);
            }

            /// <summary>
            ///  Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples
            ///  if the distribution parameters are batched.
            /// </summary>
            /// <param name="sample_shape">The sample shape.</param>
            public override Tensor rsample(params long[] sample_shape)
            {
                var cShape = new List<long>(); cShape.Add(total_count); cShape.AddRange(sample_shape);

                using var _ = NewDisposeScope();
                var samples = categorical.sample(cShape.ToArray());
                var shifted_idx = Enumerable.Range(0, (int)samples.dim()).ToList();
                var tc = shifted_idx[0];
                shifted_idx.RemoveAt(0);
                shifted_idx.Add(tc);
                samples = samples.permute(shifted_idx.Select(i => (long)i).ToArray());
                var counts = samples.new_zeros(ExtendedShape(sample_shape));
                counts.scatter_add_(-1, samples, torch.ones_like(samples));
                return counts.type_as(probs).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                using var _ = NewDisposeScope();
                var bcast = torch.broadcast_tensors(logits, value);
                var l = bcast[0].clone();
                value = bcast[1];
                var log_factorial_n = torch.lgamma(value.sum(-1) + 1);
                var log_factorial_xs = torch.lgamma(value + 1).sum(-1);
                l[(value == 0) & (l == float.NegativeInfinity)] = torch.tensor(0.0f);
                var log_powers = (logits * value).sum(-1);
                return (log_factorial_n - log_factorial_xs + log_powers).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
            /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
            /// memory for the expanded distribution instance.
            /// </summary>
            /// <param name="batch_shape">Tthe desired expanded size.</param>
            /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Multinomial))
                    throw new ArgumentException("expand(): 'instance' must be a Multinomial distribution");

                var newDistribution = ((instance == null) ?
                    new Multinomial(total_count, categorical.expand(batch_shape) as Categorical, generator) :
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

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            /// <returns></returns>
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
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Multinomial Multinomial(int total_count, Tensor probs = null, Tensor logits = null, torch.Generator generator = null)
            {
                return new Multinomial(total_count, probs, logits, generator);
            }

            /// <summary>
            /// Creates an equal-probability multinomial distribution parameterized by the number of categories.
            /// `total_count` must be broadcastable with `probs`/`logits`.
            /// </summary>
            /// <param name="total_count">Number of Bernoulli trials</param>
            /// <param name="categories">The number of categories.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Multinomial Multinomial(int total_count, int categories, torch.Generator generator = null)
            {
                var probs = torch.tensor(1.0 / categories).expand(categories);
                return new Multinomial(total_count, probs, null, generator);
            }
        }
    }
}
