// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;
    using static torch.distributions;

    namespace Modules
    {
        /// <summary>
        /// Creates a ExpRelaxedCategorical parameterized by `temperature`, and either `probs` or `logits` (but not both).
        /// Returns the log of a point in the simplex.
        /// </summary>
        public class ExpRelaxedCategorical : Distribution
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs">the probability of sampling `1`</param>
            /// <param name="logits">the log-odds of sampling `1`</param>
            /// <param name="generator"></param>
            internal ExpRelaxedCategorical(Tensor temperature, Tensor probs = null, Tensor logits = null, torch.Generator generator = null) :
                base(generator)
            {
                this._categorical = Categorical(probs, logits, generator);
                this._probs = probs?.alias().DetachFromDisposeScope();
                this._logits = logits?.alias().DetachFromDisposeScope();
                this._temperature = temperature.alias().DetachFromDisposeScope();
                base._init(_categorical.batch_shape, _categorical.event_shape);
            }

            private Tensor _probs;
            private Tensor _logits;
            private Tensor _temperature;
            private Categorical _categorical;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    _probs?.Dispose();
                    _logits?.Dispose();
                    _temperature?.Dispose();
                    _categorical?.Dispose();
                }
                base.Dispose(disposing);
            }

            /// <summary>
            /// The probability of sampling 1
            /// </summary>
            public Tensor probs {
                get {
                    return _categorical.probs;
                }
            }

            /// <summary>
            /// The log-odds of sampling 1
            /// </summary>
            public Tensor logits {
                get {
                    return _categorical.logits;
                }
            }

            public Tensor temperature {
                get {
                    return _temperature;
                }
            }

            public override Tensor mean => new Tensor(IntPtr.Zero);

            public override Tensor mode => base.mode;

            public override Tensor variance => new Tensor(IntPtr.Zero);

            public override Tensor stddev => base.stddev;

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
                if (instance != null && !(instance is ExpRelaxedCategorical))
                    throw new ArgumentException("expand(): 'instance' must be a ExpRelaxedCategorical distribution");

                var newDistribution = ((instance == null) ? new ExpRelaxedCategorical(temperature, _probs, _logits, generator) : instance) as ExpRelaxedCategorical;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution._temperature = _temperature;
                    newDistribution._categorical = _categorical.expand(batch_shape) as Categorical;
                }
                return newDistribution;
            }

            /// <summary>
            ///  The shape of the input parameter.
            /// </summary>
            public long[] param_shape {
                get {
                    return _categorical.param_shape;
                }
            }

            public override Tensor sample(params long[] sample_shape)
            {
                return rsample(sample_shape);
            }

            public override Tensor rsample(params long[] sample_shape)
            {
                using var _ = torch.NewDisposeScope();
                var shape = ExtendedShape(sample_shape);
                var uniforms = ClampProbs(torch.rand(shape, dtype: _logits.dtype, device: _logits.device));
                var gumbels = -((-(uniforms.log())).log());
                var scores = (_logits + gumbels) / _temperature;
                return (scores - scores.logsumexp(dim: -1, keepdim: true)).MoveToOuterDisposeScope();
            }

            public override Tensor log_prob(Tensor value)
            {
                using var _ = torch.NewDisposeScope();
                float K = _categorical.num_events;
                var logitsValue = broadcast_tensors(_logits, value);
                var logits = logitsValue[0];
                value = logitsValue[1];
                var log_scale = (torch.full_like(_temperature, K).lgamma() - _temperature.log().mul(-(K - 1)));
                var score = logits - value.mul(_temperature);
                score = (score - score.logsumexp(dim: -1, keepdim: true)).sum(-1);
                return (score + log_scale).MoveToOuterDisposeScope();
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
            /// Creates a ExpRelaxedCategorical parameterized by `temperature`, and either `probs` or `logits` (but not both).
            /// Returns the log of a point in the simplex.
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static ExpRelaxedCategorical ExpRelaxedCategorical(Tensor temperature, Tensor probs = null, Tensor logits = null, torch.Generator generator = null)
            {
                return new ExpRelaxedCategorical(temperature, probs, logits, generator);
            }

            /// <summary>
            /// Creates a ExpRelaxedCategorical parameterized by `temperature`, and either `probs` or `logits` (but not both).
            /// Returns the log of a point in the simplex.
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static ExpRelaxedCategorical ExpRelaxedCategorical(Tensor temperature, float? probs, float? logits, torch.Generator generator = null)
            {
                if (probs.HasValue && !logits.HasValue)
                    return new ExpRelaxedCategorical(temperature, torch.tensor(probs.Value), null, generator);
                else if (!probs.HasValue && logits.HasValue)
                    return new ExpRelaxedCategorical(temperature, null, torch.tensor(logits.Value), generator);
                else
                    throw new ArgumentException("One and only one of 'probs' and logits should be provided.");
            }


            /// <summary>
            /// Creates a ExpRelaxedCategorical parameterized by `temperature`, and either `probs` or `logits` (but not both).
            /// Returns the log of a point in the simplex.
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static ExpRelaxedCategorical ExpRelaxedCategorical(Tensor temperature, double? probs, double? logits, torch.Generator generator = null)
            {
                if (probs.HasValue && !logits.HasValue)
                    return new ExpRelaxedCategorical(temperature, torch.tensor(probs.Value), null, generator);
                else if (!probs.HasValue && logits.HasValue)
                    return new ExpRelaxedCategorical(temperature, null, torch.tensor(logits.Value), generator);
                else
                    throw new ArgumentException("One and only one of 'probs' and 'logits' should be non-null");
            }
        }
    }
}
