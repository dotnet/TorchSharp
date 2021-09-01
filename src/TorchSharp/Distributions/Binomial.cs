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
        public class Binomial : torch.distributions.Distribution
        {

            public override Tensor mean => total_count * probs;

            public override Tensor variance => total_count * probs * (1 - probs);

            public Binomial(Tensor total_count, Tensor p = null, Tensor l = null) 
            {
                this.batch_shape = p is null ? l.size() : p.size();
                this._probs = p;
                this._logits = l;

                var broadcast = (p is null) ? torch.broadcast_tensors(total_count, l) : torch.broadcast_tensors(total_count, p);
                this.total_count = broadcast[0].type_as(p ?? l);
            }

            public Tensor probs {
                get {
                    return _probs ?? LogitsToProbs(_logits, true);
                }
            }
            public Tensor logits {
                get {
                    return _logits ?? ProbsToLogits(_probs);
                }
            }

            private Tensor _probs;
            private Tensor _logits;
            private Tensor total_count;

            public override Tensor rsample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                return torch.binomial(total_count.expand(shape), probs.expand(shape));
            }

            public override Tensor log_prob(Tensor value)
            {
                var log_factorial_n = torch.lgamma(total_count + 1);
                var log_factorial_k = torch.lgamma(value + 1);
                var log_factorial_nmk = torch.lgamma(total_count - value + 1);

                var normalize_term = (total_count * ClampByZero(logits) + total_count * torch.log1p(torch.exp(-torch.abs(logits))) - log_factorial_n);
                return value * logits - log_factorial_k - log_factorial_nmk - normalize_term;
            }

            public override Tensor entropy()
            {
                return torch.nn.functional.binary_cross_entropy_with_logits(logits, probs, reduction: nn.Reduction.None);
            }

            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Binomial))
                    throw new ArgumentException("expand(): 'instance' must be a Binomial distribution");

                var newDistribution = ((instance == null) ?
                    new Binomial(total_count.expand(batch_shape), p: _probs?.expand(batch_shape), l: logits?.expand(batch_shape)) :
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
            /// <returns></returns>
            public static Binomial Binomial(Tensor total_count, Tensor probs = null, Tensor logits = null)
            {
                return new Binomial(total_count, probs, logits);
            }
        }
    }
}
