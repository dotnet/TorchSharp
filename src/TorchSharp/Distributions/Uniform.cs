// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// Generates uniformly distributed random samples from the half-open interval [low, high[.
        /// </summary>
        public class Uniform : torch.distributions.Distribution
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean =>
                WrappedTensorDisposeScope(() => (high + low) / 2);

            public override Tensor mode => double.NaN * high;

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance =>
                WrappedTensorDisposeScope(() => (high - low).pow(2) / 12);

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="low">Lower bound (inclusive)</param>
            /// <param name="high">Upper bound (exclusive)</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Uniform(Tensor low, Tensor high, torch.Generator generator = null) : base(generator, low.size())
            {
                var lowHigh = torch.broadcast_tensors(low, high);
                this.low = lowHigh[0].DetachFromDisposeScope();
                this.high = lowHigh[1].DetachFromDisposeScope();
            }

            private Tensor high;
            private Tensor low;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    low?.Dispose();
                    high?.Dispose();
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
                var rand = torch.rand(shape, dtype: low.dtype, device: low.device, generator: generator);
                return (low + rand * (high - low)).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                using var _ = torch.NewDisposeScope();
                var lb = low.le(value).type_as(low);
                var ub = high.gt(value).type_as(low);
                return (torch.log(lb.mul(ub)) - torch.log(high - low)).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns the cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor cdf(Tensor value)
            {
                return torch.WrappedTensorDisposeScope(() => ((value - low) / (high - low)).clamp_(0, 1));
            }

            /// <summary>
            /// Returns the inverse cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor icdf(Tensor value)
            {
                return torch.WrappedTensorDisposeScope(() => (value * (high - low)).add_(low));
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy()
            {
                return torch.WrappedTensorDisposeScope(() => (high - low).log_());
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
                if (instance != null && !(instance is Uniform))
                    throw new ArgumentException("expand(): 'instance' must be a Uniform distribution");

                var newDistribution = ((instance == null) ?
                    new Uniform(low.expand(batch_shape), high.expand(batch_shape), generator) :
                    instance) as Uniform;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance)
                {
                    newDistribution.low = low.expand(batch_shape);
                    newDistribution.high = high.expand(batch_shape);
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
            /// Generates uniformly distributed random samples from the half-open interval [low, high[.
            /// </summary>
            /// <param name="low">Lower bound (inclusive)</param>
            /// <param name="high">Upper bound (exclusive)</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Uniform Uniform(Tensor low, Tensor high, torch.Generator generator = null)
            {
                return new Uniform(low, high, generator);
            }
        }
    }
}
