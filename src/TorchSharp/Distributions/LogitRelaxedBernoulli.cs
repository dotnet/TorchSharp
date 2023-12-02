// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// Creates a LogitRelaxedBernoulli distribution parameterized by `probs` or 'logits` (but not both),
        /// which is the logit of a RelaxedBernoulli distribution.
        /// </summary>
        public class LogitRelaxedBernoulli : torch.distributions.Distribution
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => probs;

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance => probs * (1 - probs);

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs"></param>
            /// <param name="logits"></param>
            /// <param name="generator"></param>
            public LogitRelaxedBernoulli(Tensor temperature, Tensor probs = null, Tensor logits = null, torch.Generator generator = null) : base(generator)
            {
                _temperature = temperature.alias().DetachFromDisposeScope();

                if ((probs is null && logits is null) || (probs is not null && logits is not null))
                    throw new ArgumentException("One and only one of 'probs' and logits should be provided.");

                this.batch_shape = probs is null ? logits.size() : probs.size();
                this._probs = probs?.alias().DetachFromDisposeScope();
                this._logits = logits?.alias().DetachFromDisposeScope();
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

            private Tensor _temperature;
            private Tensor _probs;
            private Tensor _logits;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    _probs?.Dispose();
                    _logits?.Dispose();
                    _temperature?.Dispose();
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
                using var _ = NewDisposeScope();
                var shape = ExtendedShape(sample_shape);
                var probs = ClampProbs(_probs.expand(shape));
                var uniforms = ClampProbs(torch.rand(shape, dtype: probs.dtype, device: probs.device));
                return ((uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / _temperature).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            /// <returns></returns>

            public override Tensor log_prob(Tensor value)
            {
                using var _ = NewDisposeScope();
                var logitsValue = torch.broadcast_tensors(logits, value);
                var diff = logitsValue[0] - logitsValue[1].mul(_temperature);
                return (_temperature.log() + diff - 2 * diff.exp().log1p()).MoveToOuterDisposeScope();
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
            /// <returns></returns>
            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Bernoulli))
                    throw new ArgumentException("expand(): 'instance' must be a Bernoulli distribution");

                var p = _probs?.expand(batch_shape);
                var l = _logits?.expand(batch_shape);

                var newDistribution = ((instance == null) ? new LogitRelaxedBernoulli(_temperature, p, l, generator) : instance) as LogitRelaxedBernoulli;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution._probs = p;
                    newDistribution._logits = l;
                    newDistribution._temperature = _temperature;
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
            /// Creates a LogitRelaxedBernoulli distribution parameterized by `probs` or 'logits` (but not both),
            /// which is the logit of a RelaxedBernoulli distribution.
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static LogitRelaxedBernoulli LogitRelaxedBernoulli(Tensor temperature, Tensor probs = null, Tensor logits = null, torch.Generator generator = null)
            {
                return new LogitRelaxedBernoulli(temperature, probs, logits, generator);
            }

            /// <summary>
            /// Creates a LogitRelaxedBernoulli distribution parameterized by `probs` or 'logits` (but not both),
            /// which is the logit of a RelaxedBernoulli distribution.
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static LogitRelaxedBernoulli LogitRelaxedBernoulli(Tensor temperature, float? probs, float? logits, torch.Generator generator = null)
            {
                if (probs.HasValue && !logits.HasValue)
                    return new LogitRelaxedBernoulli(temperature, torch.tensor(probs.Value), null, generator);
                else if (!probs.HasValue && logits.HasValue)
                    return new LogitRelaxedBernoulli(temperature, null, torch.tensor(logits.Value), generator);
                else
                    throw new ArgumentException("One and only one of 'probs' and logits should be provided.");
            }


            /// <summary>
            /// Creates a LogitRelaxedBernoulli distribution parameterized by `probs` or 'logits` (but not both),
            /// which is the logit of a RelaxedBernoulli distribution.
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static LogitRelaxedBernoulli LogitRelaxedBernoulli(Tensor temperature, double? probs, double? logits, torch.Generator generator = null)
            {
                if (probs.HasValue && !logits.HasValue)
                    return new LogitRelaxedBernoulli(temperature, torch.tensor(probs.Value), null, generator);
                else if (!probs.HasValue && logits.HasValue)
                    return new LogitRelaxedBernoulli(temperature, null, torch.tensor(logits.Value), generator);
                else
                    throw new ArgumentException("One and only one of 'probs' and 'logits' should be non-null");
            }
        }
    }
}
