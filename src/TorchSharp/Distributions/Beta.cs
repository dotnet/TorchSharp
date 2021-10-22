// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Text;

using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class Beta : torch.distributions.ExponentialFamily
        {

            public override Tensor mean => concentration1 / (concentration1 + concentration0);

            public override Tensor variance {
                get {
                    var total = concentration0 + concentration1;
                    return concentration1 * concentration0 / (total.pow(2) * (total + 1));
                }
            }

            // Note that the order of the arguments is not a mistake -- the original source has them
            // ordered this way.

            public Beta(Tensor concentration1, Tensor concentration0, torch.Generator generator = null) : base(generator)
            {
                var bcast = torch.broadcast_tensors(concentration1, concentration0);
                this.concentration1 = bcast[0];
                this.concentration0 = bcast[1];
                this.dirichlet = new Dirichlet(torch.stack(bcast, -1), generator);
                this.batch_shape = this.dirichlet.batch_shape;
            }


            protected Dirichlet dirichlet;
            private Tensor concentration1;
            private Tensor concentration0;

            public override Tensor rsample(params long[] sample_shape)
            {
                return dirichlet.rsample(sample_shape).select(-1, 0);
            }

            public override Tensor log_prob(Tensor value)
            {
                var heads_tails = torch.stack(new Tensor[] { value, 1.0 - value }, -1);
                return dirichlet.log_prob(heads_tails);
            }

            public override Tensor entropy()
            {
                return dirichlet.entropy();
            }

            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Beta))
                    throw new ArgumentException("expand(): 'instance' must be a Beta distribution");

                var c0 = concentration0.expand(batch_shape);
                var c1 = concentration1.expand(batch_shape);

                var newDistribution = ((instance == null) ? new Beta(c1, c0) : instance) as Beta;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.dirichlet = new Dirichlet(torch.stack(new Tensor[] { c1, c0 }, -1));
                    newDistribution.concentration1 = c1;
                    newDistribution.concentration0 = c0;
                }
                return newDistribution;
            }

            protected override IList<Tensor> NaturalParams => new Tensor[] { concentration1, concentration0 };

            protected override Tensor MeanCarrierMeasure => throw new NotImplementedException();

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
