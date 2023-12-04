// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A Cauchy (Lorentz) distribution. The distribution of the ratio of
        /// independent normally distributed random variables with means `0` follows a Cauchy distribution.
        /// </summary>
        public class Cauchy : torch.distributions.Distribution
        {
            /// <summary>
            /// The mode of the distribution.
            /// </summary>
            public override Tensor mode => loc;

            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => _mean;

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance => _variance;

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Half width at half maximum.</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Cauchy(Tensor loc, Tensor scale, torch.Generator generator = null) : base(generator)
            {
                var locScale = torch.broadcast_tensors(loc, scale);
                this.loc = locScale[0].DetachFromDisposeScope();
                this.scale = locScale[1].DetachFromDisposeScope();
                this._mean = torch.full(ExtendedShape(), double.NaN, dtype: loc.dtype, device: loc.device).DetachFromDisposeScope();
                this._variance = torch.full(ExtendedShape(), double.PositiveInfinity, dtype: loc.dtype, device: loc.device).DetachFromDisposeScope();
                this.batch_shape = this.loc.size();
            }

            private Tensor loc;
            private Tensor scale;
            private Tensor _mean, _variance;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    loc?.Dispose();
                    scale?.Dispose();
                    _mean?.Dispose();
                    _variance?.Dispose();
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
                var shape = ExtendedShape(sample_shape);
                var eps = loc.new_empty(shape).cauchy_(generator: generator);
                return loc + eps * scale;
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value) =>
                WrappedTensorDisposeScope(() => -Math.Log(Math.PI) - scale.log() - (((value - loc) / scale).pow(2)).log1p());            

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            /// <returns></returns>
            public override Tensor entropy() =>
                WrappedTensorDisposeScope(() => Math.Log(Math.PI * 4) + scale.log());

            /// <summary>
            /// Returns the cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor cdf(Tensor value) =>
                WrappedTensorDisposeScope(() => torch.atan((value - loc) / scale) / Math.PI + 0.5);

            /// <summary>
            /// Returns the inverse cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor icdf(Tensor value) =>
                WrappedTensorDisposeScope(() => torch.tan(Math.PI * (value - 0.5)) * scale + loc);


            /// <summary>
            /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
            /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
            /// memory for the expanded distribution instance.
            /// </summary>
            /// <param name="batch_shape">Tthe desired expanded size.</param>
            /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Cauchy))
                    throw new ArgumentException("expand(): 'instance' must be a Cauchy distribution");

                var newDistribution = ((instance == null) ? new Cauchy(loc.expand(batch_shape), scale.expand(batch_shape), generator) : instance) as Cauchy;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.loc = loc.expand(batch_shape);
                    newDistribution.scale = scale.expand(batch_shape);
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
            /// Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of
            /// independent normally distributed random variables with means `0` follows a Cauchy distribution.
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Half width at half maximum.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Cauchy Cauchy(Tensor loc, Tensor scale, torch.Generator generator = null)
            {
                return new Cauchy(loc, scale, generator);
            }
        }
    }
}
