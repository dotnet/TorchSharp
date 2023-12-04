// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A Geometric distribution parameterized by probs,
        /// where probs is the probability of success of Bernoulli trials.
        ///
        /// It represents the probability that in k+1 Bernoulli trials, the
        /// first k trials failed, before seeing a success.
        /// </summary>
        public class Geometric : torch.distributions.Distribution
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => 1 / (probs - 1);

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance => (1.0f / probs - 1.0f) / probs;

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="probs">The probability of sampling '1'. Must be in range (0, 1]</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Geometric(Tensor probs = null, Tensor logits = null, torch.Generator generator = null) : base(generator) 
            {
                this.batch_shape = probs is null ? logits.size() : probs.size();
                this._probs = probs?.alias().DetachFromDisposeScope();
                this._logits = logits?.alias().DetachFromDisposeScope();
            }

            /// <summary>
            /// Event probabilities
            /// </summary>
            public Tensor probs {
                get {
                    return _probs ?? LogitsToProbs(_logits, true);
                }
            }

            /// <summary>
            /// Event log-odds
            /// </summary>
            public Tensor logits {
                get {
                    return _logits ?? ProbsToLogits(_probs);
                }
            }

            private Tensor _probs;
            private Tensor _logits;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    _probs?.Dispose();
                    _logits?.Dispose();
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
                using var _ = torch.NewDisposeScope();
                var shape = ExtendedShape(sample_shape);
                var tiny = torch.finfo(probs.dtype).tiny;
                using (torch.no_grad()) {
                    var u = probs.new_empty(shape).uniform_(tiny, 1, generator: generator);
                    return (u.log() / (-probs).log1p()).floor().MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                using var _ = torch.NewDisposeScope();
                var bcast = torch.broadcast_tensors(value, probs);
                value = bcast[0];
                var p = bcast[1].clone();
                p[(p == 1) & (value == 0)] = torch.tensor(0);
                return (value * (-p).log1p() + probs.log()).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            /// <returns></returns>
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
                if (instance != null && !(instance is Geometric))
                    throw new ArgumentException("expand(): 'instance' must be a Geometric distribution");

                var newDistribution = ((instance == null) ?
                    new Geometric(probs: _probs?.expand(batch_shape), logits: logits?.expand(batch_shape), generator) :
                    instance) as Geometric;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
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
            /// Creates a Geometric distribution parameterized by probs,
            /// where probs is the probability of success of Bernoulli trials.
            ///
            /// It represents the probability that in k+1 Bernoulli trials, the
            /// first k trials failed, before seeing a success.
            /// </summary>
            /// <param name="probs">The probability of sampling '1'. Must be in range (0, 1]</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Geometric Geometric(Tensor probs = null, Tensor logits = null, torch.Generator generator = null)
            {
                return new Geometric(probs, logits, generator);
            }
        }
    }
}
