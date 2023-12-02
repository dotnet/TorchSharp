// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A Bernoulli distribution parameterized by `probs` or `logits` (but not both).
        /// </summary>
        public class OneHotCategorical : torch.distributions.Distribution
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => _categorical._probs;

            /// <summary>
            /// The mode of the distribution.
            /// </summary>
            public override Tensor mode {
                get {
                    var probs = _categorical.probs;
                    var mode = probs.argmax(-1);
                    return torch.nn.functional.one_hot(mode, num_classes: probs.shape[probs.ndim - 1]).to(probs);
                }
            }

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance => probs * (1 - _categorical.probs);

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="p"></param>
            /// <param name="l"></param>
            /// <param name="generator"></param>
            public OneHotCategorical(Tensor p = null, Tensor l = null, torch.Generator generator = null) : base(generator)
            {
                _categorical = new Categorical(p, l, generator);

                if ((p is null && logits is null) || (p is not null && l is not null))
                    throw new ArgumentException("One and only one of 'probs' and logits should be provided.");

                this.batch_shape = p is null ? l.size() : p.size();
                this._probs = p?.alias().DetachFromDisposeScope();
                this._logits = l?.alias().DetachFromDisposeScope();
            }

            /// <summary>
            /// The probability of sampling 1
            /// </summary>
            public Tensor probs {
                get {
                    return _probs ?? LogitsToProbs(_logits, true);
                }
            }

            /// <summary>
            /// The log-odds of sampling 1
            /// </summary>
            public Tensor logits {
                get {
                    return _logits ?? ProbsToLogits(_probs);
                }
            }

            private Categorical _categorical;
            private Tensor _probs;
            private Tensor _logits;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    _probs?.Dispose();
                    _logits?.Dispose();
                    _categorical?.Dispose();
                }
                base.Dispose(disposing);
            }

            /// <summary>
            ///  Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples
            ///  if the distribution parameters are batched.
            /// </summary>
            /// <param name="sample_shape">The sample shape.</param>
            /// <returns></returns>
            public override Tensor rsample(params long[] sample_shape)
            {
                var probs = _categorical.probs;
                var num_events = _categorical.num_events;
                var indices = _categorical.sample(sample_shape);
                return torch.nn.functional.one_hot(indices, num_events).to(probs);
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            /// <returns></returns>

            public override Tensor log_prob(Tensor value)
            {
                using var _ = torch.NewDisposeScope();
                var indices = value.max(-1).indexes;
                return _categorical.log_prob(indices).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy()
            {
                return _categorical.entropy();
            }

            /// <summary>
            /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
            /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
            /// memory for the expanded distribution instance.
            /// </summary>
            /// <param name="batch_shape">Tthe desired expanded size.</param>
            /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
            /// <returns></returns>
            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Bernoulli))
                    throw new ArgumentException("expand(): 'instance' must be a Bernoulli distribution");

                var p = _probs?.expand(batch_shape);
                var l = _logits?.expand(batch_shape);

                var newDistribution = ((instance == null) ? new OneHotCategorical(p, l, generator) : instance) as OneHotCategorical;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution._probs = p;
                    newDistribution._logits = l;
                    newDistribution._categorical = _categorical.expand(batch_shape) as Categorical;
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
            /// Creates a Bernoulli distribution parameterized by `probs` or `logits` (but not both).
            /// </summary>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static OneHotCategorical OneHotCategorical(Tensor probs = null, Tensor logits = null, torch.Generator generator = null)
            {
                return new OneHotCategorical(probs, logits, generator);
            }

            /// <summary>
            /// Creates a Bernoulli distribution parameterized by `probs` or `logits` (but not both).
            /// </summary>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static OneHotCategorical OneHotCategorical(float? probs, float? logits, torch.Generator generator = null)
            {
                if (probs.HasValue && !logits.HasValue)
                    return new OneHotCategorical(torch.tensor(probs.Value), null, generator);
                else if (!probs.HasValue && logits.HasValue)
                    return new OneHotCategorical(null, torch.tensor(logits.Value), generator);
                else
                    throw new ArgumentException("One and only one of 'probs' and logits should be provided.");
            }


            /// <summary>
            /// Creates a Bernoulli distribution parameterized by `probs` or `logits` (but not both).
            /// </summary>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static OneHotCategorical OneHotCategorical(double? probs, double? logits, torch.Generator generator = null)
            {
                if (probs.HasValue && !logits.HasValue)
                    return new OneHotCategorical(torch.tensor(probs.Value), null, generator);
                else if (!probs.HasValue && logits.HasValue)
                    return new OneHotCategorical(null, torch.tensor(logits.Value), generator);
                else
                    throw new ArgumentException("One and only one of 'probs' and 'logits' should be non-null");
            }
        }
    }
}
