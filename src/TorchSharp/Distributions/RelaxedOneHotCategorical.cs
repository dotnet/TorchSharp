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
        /// Creates a RelaxedOneHotCategorical distribution, parametrized by `temperature`, and either `probs` or `logits` (but not both).
        /// This is a relaxed version of the `OneHotCategorical` distribution, so its samples are on simplex, and are reparametrizable..
        /// </summary>
        public class RelaxedOneHotCategorical : TransformedDistribution
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs">the probability of sampling `1`</param>
            /// <param name="logits">the log-odds of sampling `1`</param>
            /// <param name="generator"></param>
            public RelaxedOneHotCategorical(Tensor temperature, Tensor probs = null, Tensor logits = null, torch.Generator generator = null) :
                base(ExpRelaxedCategorical(temperature, probs, logits, generator), new distributions.transforms.ExpTransform(), generator)
            {
                this._probs = probs?.alias().DetachFromDisposeScope();
                this._logits = logits?.alias().DetachFromDisposeScope();
            }

            private Tensor _probs;
            private Tensor _logits;

            private ExpRelaxedCategorical base_dist => this.base_distribution as ExpRelaxedCategorical;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    _probs?.Dispose();
                    _logits?.Dispose();
                    base_dist?.Dispose();
                }
                base.Dispose(disposing);
            }

            /// <summary>
            /// The probability of sampling 1
            /// </summary>
            public Tensor probs {
                get {
                    return base_dist.probs;
                }
            }

            /// <summary>
            /// The log-odds of sampling 1
            /// </summary>
            public Tensor logits {
                get {
                    return base_dist.logits;
                }
            }

            public Tensor temperature {
                get {
                    return base_dist.logits;
                }
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
                if (instance != null && !(instance is RelaxedOneHotCategorical))
                    throw new ArgumentException("expand(): 'instance' must be a RelaxedOneHotCategorical distribution");

                var newDistribution = ((instance == null) ? new RelaxedOneHotCategorical(temperature, _probs, _logits, generator) : instance) as RelaxedOneHotCategorical;

                newDistribution.batch_shape = batch_shape;
                return base.expand(batch_shape, newDistribution);
            }
        }

    }
    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a RelaxedOneHotCategorical distribution, parametrized by `temperature`, and either `probs` or `logits` (but not both).
            /// This is a relaxed version of the `OneHotCategorical` distribution, so its samples are on simplex, and are reparametrizable..
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static RelaxedOneHotCategorical RelaxedOneHotCategorical(Tensor temperature, Tensor probs = null, Tensor logits = null, torch.Generator generator = null)
            {
                return new RelaxedOneHotCategorical(temperature, probs, logits, generator);
            }

            /// <summary>
            /// Creates a RelaxedOneHotCategorical distribution, parametrized by `temperature`, and either `probs` or `logits` (but not both).
            /// This is a relaxed version of the `OneHotCategorical` distribution, so its samples are on simplex, and are reparametrizable..
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static RelaxedOneHotCategorical RelaxedOneHotCategorical(Tensor temperature, float? probs, float? logits, torch.Generator generator = null)
            {
                if (probs.HasValue && !logits.HasValue)
                    return new RelaxedOneHotCategorical(temperature, torch.tensor(probs.Value), null, generator);
                else if (!probs.HasValue && logits.HasValue)
                    return new RelaxedOneHotCategorical(temperature, null, torch.tensor(logits.Value), generator);
                else
                    throw new ArgumentException("One and only one of 'probs' and logits should be provided.");
            }


            /// <summary>
            /// Creates a RelaxedOneHotCategorical distribution, parametrized by `temperature`, and either `probs` or `logits` (but not both).
            /// This is a relaxed version of the `OneHotCategorical` distribution, so its samples are on simplex, and are reparametrizable..
            /// </summary>
            /// <param name="temperature">Relaxation temperature</param>
            /// <param name="probs">The probability of sampling '1'</param>
            /// <param name="logits">The log-odds of sampling '1'</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static RelaxedOneHotCategorical RelaxedOneHotCategorical(Tensor temperature, double? probs, double? logits, torch.Generator generator = null)
            {
                if (probs.HasValue && !logits.HasValue)
                    return new RelaxedOneHotCategorical(temperature, torch.tensor(probs.Value), null, generator);
                else if (!probs.HasValue && logits.HasValue)
                    return new RelaxedOneHotCategorical(temperature, null, torch.tensor(logits.Value), generator);
                else
                    throw new ArgumentException("One and only one of 'probs' and 'logits' should be non-null");
            }
        }
    }
}
