// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A Gamma distribution parameterized by shape `concentration` and `rate`.
        /// </summary>
        public class Gamma : torch.distributions.ExponentialFamily
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => WrappedTensorDisposeScope(() => concentration / rate);

            public override Tensor mode => WrappedTensorDisposeScope(() => ((concentration - 1) / rate).clamp_(min: 0));

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance => WrappedTensorDisposeScope(() => concentration / rate.pow(2));

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="concentration">Shape parameter of the distribution (often referred to as 'α')</param>
            /// <param name="rate">rate = 1 / scale of the distribution (often referred to as 'β')</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Gamma(Tensor concentration, Tensor rate, torch.Generator generator = null) : base(generator)
            {
                var locScale = torch.broadcast_tensors(concentration, rate);
                this.concentration = locScale[0].DetachFromDisposeScope();
                this.rate = locScale[1].DetachFromDisposeScope();
                this.batch_shape = this.concentration.size();
            }

            protected Tensor concentration;
            private Tensor rate;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    rate?.Dispose();
                    concentration?.Dispose();
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
                var value = torch._standard_gamma(concentration.expand(shape), generator: generator) / rate.expand(shape);
                return value.detach().clamp_(min: torch.finfo(value.dtype).tiny).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                using var _ = torch.NewDisposeScope();
                value = torch.as_tensor(value, dtype: rate.dtype, device: rate.device);
                var result = concentration * rate.log() + (concentration - 1) * value.log() - rate * value - torch.lgamma(concentration);
                return result.MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy() =>
                WrappedTensorDisposeScope(() =>
                    concentration - rate.log() + concentration.lgamma() + (1.0 - concentration) * concentration.digamma());

            public override Tensor cdf(Tensor value) =>
                WrappedTensorDisposeScope(() => torch.special.gammainc(concentration, rate * value));

            /// <summary>
            /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
            /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
            /// memory for the expanded distribution instance.
            /// </summary>
            /// <param name="batch_shape">Tthe desired expanded size.</param>
            /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Gamma))
                    throw new ArgumentException("expand(): 'instance' must be a Gamma distribution");

                var c = concentration.expand(batch_shape);
                var r = rate.expand(batch_shape);

                var newDistribution = ((instance == null) ? new Gamma(c, r, generator) : instance) as Gamma;

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
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Gamma Gamma(Tensor concentration, Tensor rate, torch.Generator generator = null)
            {
                return new Gamma(concentration, rate, generator);
            }
        }
    }
}
