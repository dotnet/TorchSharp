// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A Normal (Gaussian) distribution.
        /// </summary>
        public class Normal : distributions.Distribution
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => loc;

            /// <summary>
            /// The mode of the distribution.
            /// </summary>
            public override Tensor mode => loc;

            /// <summary>
            /// The standard deviation of the distribution
            /// </summary>
            public override Tensor stddev => scale;

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance => scale.pow(2);

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Standard deviation.</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Normal(Tensor loc, Tensor scale, Generator generator = null) : base(generator)
            {
                var locScale = broadcast_tensors(loc, scale);
                this.loc = locScale[0].DetachFromDisposeScope();
                this.scale = locScale[1].DetachFromDisposeScope();
                this.batch_shape = this.loc.size();
            }

            private Tensor loc;
            private Tensor scale;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    loc?.Dispose();
                    scale?.Dispose();
                }
                base.Dispose(disposing);
            }

            /// <summary>
            /// Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.
            /// </summary>
            /// <param name="sample_shape"></param>
            public override Tensor sample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                using (no_grad()) {
                    return normal(loc.expand(shape), scale.expand(shape), generator);
                }
            }


            /// <summary>
            ///  Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples
            ///  if the distribution parameters are batched.
            /// </summary>
            /// <param name="sample_shape">The sample shape.</param>
            public override Tensor rsample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                var eps = empty(shape, dtype: loc.dtype, device: loc.device).normal_(generator: generator);
                return loc + eps * scale;
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                using var _ = NewDisposeScope();
                var v = scale.pow(2);
                var log_scale = scale.log();
                return (-((value - loc).pow(2)) / (2 * v) - log_scale - Math.Log(Math.Sqrt(2 * Math.PI))).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy()
            {
                return WrappedTensorDisposeScope(() =>
                    0.5 + 0.5 * Math.Log(2 * Math.PI) + log(scale)
                );
            }

            /// <summary>
            /// Returns the cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor cdf(Tensor value)
            {
                return WrappedTensorDisposeScope(() =>
                    0.5 * (1 + special.erf((value - loc) * scale.reciprocal() / Math.Sqrt(2)))
                );
            }

            /// <summary>
            /// Returns the inverse cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor icdf(Tensor value)
            {
                return WrappedTensorDisposeScope(() =>
                    loc + scale * special.erfinv(2 * value - 1) * Math.Sqrt(2)
                );
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
                if (instance != null && !(instance is Normal))
                    throw new ArgumentException("expand(): 'instance' must be a Normal distribution");

                var newDistribution = ((instance == null) ? new Normal(loc.expand(batch_shape), scale.expand(batch_shape), generator) : instance) as Normal;

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
            /// Samples from a Normal (Gaussian) distribution. The distribution of the ratio of
            /// independent normally distributed random variables with means `0` follows a Normal distribution.
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Standard deviation.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Normal Normal(Tensor loc, Tensor scale, Generator generator = null)
            {
                return new Normal(loc, scale, generator);
            }

            /// <summary>
            /// Samples from a Normal (Gaussian) distribution. The distribution of the ratio of
            /// independent normally distributed random variables with means `0` follows a Normal distribution.
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Standard deviation.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Normal Normal(float loc, float scale = 1.0f, Generator generator = null)
            {
                return new Normal(tensor(loc), tensor(scale), generator);
            }


            /// <summary>
            /// Samples from a Normal (Gaussian) distribution. The distribution of the ratio of
            /// independent normally distributed random variables with means `0` follows a Normal distribution.
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Standard deviation.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Normal Normal(double loc, double scale = 1.0, Generator generator = null)
            {
                return new Normal(tensor(loc), tensor(scale), generator);
            }
        }
    }
}
