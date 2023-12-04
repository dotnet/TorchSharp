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
        /// A Beta distribution parameterized by concentration1 and concentration0.
        /// </summary>
        public class Beta : torch.distributions.ExponentialFamily
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => concentration1 / (concentration1 + concentration0);

            public override Tensor mode => dirichlet.mode[TensorIndex.Ellipsis, 0];

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance {
                get {
                    using var _ = NewDisposeScope();
                    var total = concentration0 + concentration1;
                    return (concentration1 * concentration0 / (total.pow(2) * (total + 1))).MoveToOuterDisposeScope();
                }
            }

            // Note that the order of the arguments is not a mistake -- the original source has them
            // ordered this way.

            public Beta(Tensor concentration1, Tensor concentration0, torch.Generator generator = null) : base(generator)
            {
                var bcast = torch.broadcast_tensors(concentration1, concentration0);
                this.concentration1 = bcast[0].DetachFromDisposeScope();
                this.concentration0 = bcast[1].DetachFromDisposeScope();
                this.dirichlet = new Dirichlet(torch.stack(bcast, -1), generator);
                this.batch_shape = this.dirichlet.batch_shape;
            }


            protected Dirichlet dirichlet;
            private Tensor concentration1;
            private Tensor concentration0;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    dirichlet?.Dispose();
                    concentration0?.Dispose();
                    concentration1?.Dispose();
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
                return dirichlet.rsample(sample_shape).select(-1, 0);
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                var heads_tails = torch.stack(new Tensor[] { value, 1.0 - value }, -1);
                return dirichlet.log_prob(heads_tails);
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy()
            {
                return dirichlet.entropy();
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
                if (instance != null && !(instance is Beta))
                    throw new ArgumentException("expand(): 'instance' must be a Beta distribution");

                var c0 = concentration0.expand(batch_shape);
                var c1 = concentration1.expand(batch_shape);

                var newDistribution = ((instance == null) ? new Beta(c1, c0, generator) : instance) as Beta;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.dirichlet = new Dirichlet(torch.stack(new Tensor[] { c1, c0 }, -1));
                    newDistribution.concentration1 = c1;
                    newDistribution.concentration0 = c0;
                }
                return newDistribution;
            }

            protected override IList<Tensor> NaturalParams => new Tensor[] { concentration1, concentration0 };

            protected override Tensor MeanCarrierMeasure => new Tensor(IntPtr.Zero);

            protected override Tensor LogNormalizer(params Tensor[] parameters)
            {
                var x = parameters[0];
                var y = parameters[1];

                return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y);
            }
        }

    }

    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a Beta distribution parameterized by concentration1 and concentration0.
            /// </summary>
            /// <param name="concentration1">1st concentration parameter of the distribution (often referred to as 'α')</param>
            /// <param name="concentration0">2nd concentration parameter of the distribution (often referred to as 'β')</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            /// <remarks>The order of the arguments is not a mistake -- the original source has them ordered this way.
            /// </remarks>
            public static Beta Beta(Tensor concentration1, Tensor concentration0, torch.Generator generator = null)
            {
                return new Beta(concentration1, concentration0, generator);
            }
        }
    }
}
