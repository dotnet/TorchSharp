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
        /// An Exponential distribution parameterized by `rate`.
        /// </summary>
        public class Exponential : torch.distributions.ExponentialFamily
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => rate.reciprocal();

            public override Tensor mode => torch.zeros_like(rate);

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance => rate.pow(2);

            /// <summary>
            /// The standard deviation of the distribution
            /// </summary>
            public override Tensor stddev => rate.reciprocal();


            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="rate">rate = 1 / scale of the distribution (often referred to as 'β')</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Exponential(Tensor rate, torch.Generator generator = null) : base(generator)
            {
                var locScale = torch.broadcast_tensors(rate);
                this.batch_shape = rate.size();
                this.rate = locScale[0].DetachFromDisposeScope();
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
                var shape = ExtendedShape(sample_shape);
                return rate.new_empty(shape).exponential_(generator: generator) / rate;
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value) => WrappedTensorDisposeScope(() => rate.log() - rate * value);

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            /// <returns></returns>
            public override Tensor entropy() => WrappedTensorDisposeScope(() => 1 - rate.log());

            /// <summary>
            /// Returns the cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor cdf(Tensor value) => WrappedTensorDisposeScope(() => 1 - torch.exp(-rate * value));

            /// <summary>
            /// Returns the inverse cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor icdf(Tensor value) => WrappedTensorDisposeScope(() => -torch.log(1 - value) / rate);

            /// <summary>
            /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
            /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
            /// memory for the expanded distribution instance.
            /// </summary>
            /// <param name="batch_shape">Tthe desired expanded size.</param>
            /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Exponential))
                    throw new ArgumentException("expand(): 'instance' must be a Exponential distribution");

                var r = rate.expand(batch_shape);

                var newDistribution = ((instance == null) ? new Exponential(r, generator) : instance) as Exponential;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.rate = r;
                }
                return newDistribution;
            }

            protected override IList<Tensor> NaturalParams => new Tensor[] { -rate };

            protected override Tensor MeanCarrierMeasure => torch.tensor(0, dtype:rate.dtype, device: rate.device);

            protected override Tensor LogNormalizer(params Tensor[] parameters) => -torch.log(-parameters[0]);
        }

    }

    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a Exponential distribution parameterized by `rate`.
            /// </summary>
            /// <param name="rate">rate = 1 / scale of the distribution (often referred to as 'β')</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Exponential Exponential(Tensor rate, torch.Generator generator = null)
            {
                return new Exponential(rate, generator);
            }
        }
    }
}
