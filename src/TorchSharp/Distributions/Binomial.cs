// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A Binomial distribution parameterized by total_count and either probs or logits (but not both).
        /// </summary>
        public class Binomial : torch.distributions.Distribution
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => total_count * probs;

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance {
                get {
                    // Individual sub-expressions is slightly faster than using DisposeScope
                    using var t0 = total_count * probs;
                    using var t1 = (1 - probs);
                    return t0 * t1;
                }
            }

            /// <summary>
            /// Mode of the negative binomial distribution.
            /// </summary>
            public override Tensor mode {
                get {
                    // Individual sub-expressions is slightly faster than using DisposeScope
                    using var t0 = total_count + 1;
                    using var t1 = t0 * _probs;
                    return t1.floor_().clamp(max: total_count);
                }
            }

            public Binomial(Tensor total_count, Tensor p = null, Tensor l = null, torch.Generator generator = null) : base(generator)
            {
                this.batch_shape = p is null ? l.size() : p.size();
                this._probs = p ?? LogitsToProbs(l, true).DetachFromDisposeScope();
                this._logits = l ?? ProbsToLogits(p, true).DetachFromDisposeScope();
                this.generator = generator;

                var broadcast = (p is null) ? torch.broadcast_tensors(total_count, l) : torch.broadcast_tensors(total_count, p);
                this.total_count = broadcast[0].type_as(p ?? l).DetachFromDisposeScope();
            }

            /// <summary>
            /// Event probabilities
            /// </summary>
            public Tensor probs {
                get {
                    return _probs;
                }
            }

            /// <summary>
            /// Event log-odds
            /// </summary>
            public Tensor logits {
                get {
                    return _logits;
                }
            }

            private Tensor _probs;
            private Tensor _logits;
            private Tensor total_count;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    _probs?.Dispose();
                    _logits?.Dispose();
                    total_count?.Dispose();
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
                var shape = ExtendedShape(sample_shape);
                return torch.binomial(total_count.expand(shape), probs.expand(shape), generator);
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                var log_factorial_n = torch.lgamma(total_count + 1);
                var log_factorial_k = torch.lgamma(value + 1);
                var log_factorial_nmk = torch.lgamma(total_count - value + 1);

                var normalize_term = (total_count * ClampByZero(logits) + total_count * torch.log1p(torch.exp(-torch.abs(logits))) - log_factorial_n);
                return value * logits - log_factorial_k - log_factorial_nmk - normalize_term;
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy()
            {
                return torch.nn.functional.binary_cross_entropy_with_logits(logits, probs, reduction: nn.Reduction.None);
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
                if (instance != null && !(instance is Binomial))
                    throw new ArgumentException("expand(): 'instance' must be a Binomial distribution");

                var newDistribution = ((instance == null) ?
                    new Binomial(total_count.expand(batch_shape), p: _probs?.expand(batch_shape), l: logits?.expand(batch_shape), generator) :
                    instance) as Binomial;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.total_count = total_count.expand(batch_shape);
                    newDistribution._probs = _probs?.expand(batch_shape);
                    newDistribution._logits = _logits?.expand(batch_shape);
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
            /// Creates a Binomial distribution parameterized by `probs` or `logits` (but not both).
            /// `total_count` must be broadcastable with `probs`/`logits`.
            /// </summary>
            /// <param name="total_count">Number of Bernoulli trials</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Binomial Binomial(Tensor total_count, Tensor probs = null, Tensor logits = null, torch.Generator generator = null)
            {
                return new Binomial(total_count, probs, logits);
            }

            /// <summary>
            /// Creates a Binomial distribution parameterized by `probs` or `logits` (but not both).
            /// </summary>
            /// <param name="total_count">Number of Bernoulli trials</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Binomial Binomial(int total_count, float? probs, float? logits, torch.Generator generator = null)
            {
                if (probs.HasValue && !logits.HasValue)
                    return new Binomial(torch.tensor(total_count), torch.tensor(probs.Value), null);
                else if (!probs.HasValue && logits.HasValue)
                    return new Binomial(torch.tensor(total_count), null, torch.tensor(logits.Value));
                else
                    throw new ArgumentException("One and only one of 'probs' and logits should be provided.");
            }


            /// <summary>
            /// Creates a Binomial distribution parameterized by `probs` or `logits` (but not both).
            /// </summary>
            /// <param name="total_count">Number of Bernoulli trials</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Binomial Binomial(int total_count, double? probs, double? logits, torch.Generator generator = null)
            {
                if (probs.HasValue && !logits.HasValue)
                    return new Binomial(torch.tensor(total_count), torch.tensor(probs.Value), null, generator);
                else if (!probs.HasValue && logits.HasValue)
                    return new Binomial(torch.tensor(total_count), null, torch.tensor(logits.Value), generator);
                else
                    throw new ArgumentException("One and only one of 'probs' and logits should be provided.");
            }

        }
    }
}
