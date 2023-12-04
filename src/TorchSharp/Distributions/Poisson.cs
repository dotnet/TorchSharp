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
        /// A Poisson distribution parameterized by `rate`.
        /// </summary>
        public class Poisson : distributions.ExponentialFamily
        {
            public override Tensor mean => rate;

            public override Tensor variance => rate;

            public override Tensor mode => rate.floor();

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="rate">rate = 1 / scale of the distribution (often referred to as 'β')</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Poisson(Tensor rate, Generator generator = null) : base(generator)
            {
                var locScale = broadcast_tensors(rate);
                batch_shape = rate.size();
                this.rate = locScale[0].alias().DetachFromDisposeScope();
            }

            private Tensor rate;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    rate?.Dispose();
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
                using var _ = NewDisposeScope();
                var shape = ExtendedShape(sample_shape);
                using(no_grad())
                    return torch.poisson(rate.expand(shape), generator: generator).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                using var _ = NewDisposeScope();
                var bcast = broadcast_tensors(rate, value);
                var r = bcast[0];
                var v = bcast[1];
                return (value.xlogy(r) - r - (value + 1).lgamma()).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns the cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor cdf(Tensor value)
            {
                return WrappedTensorDisposeScope(() => 1 - exp(-rate * value));
            }

            /// <summary>
            /// Returns the inverse cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor icdf(Tensor value)
            {
                return WrappedTensorDisposeScope(() => -log(1 - value) / rate);
            }

            /// <summary>
            /// Returns tensor containing all values supported by a discrete distribution. The result will enumerate over dimension 0, so the shape
            /// of the result will be `(cardinality,) + batch_shape + event_shape` (where `event_shape = ()` for univariate distributions).
            ///
            /// Note that this enumerates over all batched tensors in lock-step `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
            /// along dim 0, but with the remaining batch dimensions being singleton dimensions, `[[0], [1], ..`
            /// </summary>
            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Poisson))
                    throw new ArgumentException("expand(): 'instance' must be a Poisson distribution");

                var r = rate.expand(batch_shape);

                var newDistribution = ((instance == null) ? new Poisson(r, generator) : instance) as Poisson;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.rate = r;
                }
                return newDistribution;
            }

            protected override IList<Tensor> NaturalParams => new Tensor[] { rate.log() };

            protected override Tensor MeanCarrierMeasure => new Tensor(IntPtr.Zero);

            protected override Tensor LogNormalizer(params Tensor[] parameters) => parameters[0].exp();
        }

    }

    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a Poisson distribution parameterized by `rate`.
            /// </summary>
            /// <param name="rate">rate = 1 / scale of the distribution (often referred to as 'β')</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Poisson Poisson(Tensor rate, Generator generator = null)
            {
                return new Poisson(rate, generator);
            }

            /// <summary>
            /// Creates a Poisson distribution parameterized by `rate`.
            /// </summary>
            /// <param name="rate">rate = 1 / scale of the distribution (often referred to as 'β')</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Poisson Poisson(float rate, Generator generator = null)
            {
                return new Poisson(tensor(rate), generator);
            }

            /// <summary>
            /// Creates a Poisson distribution parameterized by `rate`.
            /// </summary>
            /// <param name="rate">rate = 1 / scale of the distribution (often referred to as 'β')</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Poisson Poisson(double rate, Generator generator = null)
            {
                return new Poisson(tensor(rate), generator);
            }
        }
    }
}
