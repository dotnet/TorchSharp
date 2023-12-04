// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A Laplace distribution.
        /// </summary>
        public class Laplace : torch.distributions.Distribution
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
            public override Tensor stddev => scale * Math.Sqrt(2.0);

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance => 2 * scale.pow(2);


            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Standard deviation.</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Laplace(Tensor loc, Tensor scale, torch.Generator generator = null) : base(generator)
            {
                this.batch_shape = loc.size();
                var locScale = torch.broadcast_tensors(loc, scale);
                this.loc = locScale[0].DetachFromDisposeScope();
                this.scale = locScale[1].DetachFromDisposeScope();
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
            ///  Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples
            ///  if the distribution parameters are batched.
            /// </summary>
            /// <param name="sample_shape">The sample shape.</param>
            public override Tensor rsample(params long[] sample_shape)
            {
                using var _ = NewDisposeScope();
                var shape = ExtendedShape(sample_shape);
                var finfo = torch.finfo(loc.dtype);
                var u = loc.new_empty(shape).uniform_(finfo.eps - 1, 1);
                var eps = torch.empty(shape, dtype: loc.dtype, device: loc.device).normal_(generator: generator);
                return (loc - scale * u.sign() * torch.log1p(-u.abs())).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                return torch.WrappedTensorDisposeScope(() =>
                    -torch.log(2 * scale) - torch.abs(value - loc) / scale
                );
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy()
            {
                return torch.WrappedTensorDisposeScope(() =>
                    1 + torch.log(2 * scale)
                );
            }

            /// <summary>
            /// Returns the cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor cdf(Tensor value)
            {
                return torch.WrappedTensorDisposeScope(() =>
                    0.5 - 0.5 * (value - loc).sign() * torch.expm1(-(value - loc).abs() / scale)
                );
            }

            /// <summary>
            /// Returns the inverse cumulative density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor icdf(Tensor value)
            {
                using var _ = NewDisposeScope();
                var term = value - 0.5;
                return (loc - scale * (term).sign() * torch.log1p(-2 * term.abs())).MoveToOuterDisposeScope();
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
                if (instance != null && !(instance is Laplace))
                    throw new ArgumentException("expand(): 'instance' must be a Normal distribution");

                var newDistribution = ((instance == null) ? new Laplace(loc.expand(batch_shape), scale.expand(batch_shape), generator) : instance) as Laplace;

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
            /// Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Standard deviation.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Laplace Laplace(Tensor loc, Tensor scale, torch.Generator generator = null)
            {
                return new Laplace(loc, scale, generator);
            }

            /// <summary>
            /// Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Standard deviation.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Laplace Laplace(float loc, float scale = 1.0f, torch.Generator generator = null)
            {
                return new Laplace(torch.tensor(loc), torch.tensor(scale), generator);
            }


            /// <summary>
            /// Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Standard deviation.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Laplace Laplace(double loc, double scale = 1.0, torch.Generator generator = null)
            {
                return new Laplace(torch.tensor(loc), torch.tensor(scale), generator);
            }
        }
    }
}
