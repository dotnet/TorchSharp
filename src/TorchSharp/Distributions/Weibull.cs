// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using static TorchSharp.torch;

namespace TorchSharp
{
    using System.Linq;
    using System.Net.WebSockets;
    using Modules;
    using TorchSharp.torchvision;
    using static torch.distributions;

    namespace Modules
    {
        /// <summary>
        /// Samples from a two-parameter Weibull distribution.
        /// </summary>
        public class Weibull : TransformedDistribution
        {
            internal Weibull(Tensor scale, Tensor concentration, Tensor concentration_reciprocal, Distribution base_distribution, torch.distributions.transforms.Transform[] transforms, torch.Generator generator = null) :
                base(base_distribution, transforms, generator)
            {
                this.scale = scale;
                this.concentration = concentration;
                this.concentration_reciprocal = concentration_reciprocal;
            }

            private Tensor scale;
            private Tensor concentration;
            private Tensor concentration_reciprocal;

            public override Tensor mean => scale * torch.exp(torch.lgamma(1 + concentration_reciprocal));

            public override Tensor mode => scale * ((concentration - 1) / concentration).pow(concentration_reciprocal);

            public override Tensor variance =>
                torch.WrappedTensorDisposeScope(() =>
                    scale.pow(2) * (torch.exp(torch.lgamma(1 + 2 * concentration_reciprocal)) - torch.exp(2 * torch.lgamma(1 + concentration_reciprocal)))
                );

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy() =>
                torch.WrappedTensorDisposeScope(() => euler_constant * (1 - concentration_reciprocal) + torch.log(scale * concentration_reciprocal) + 1);

            /// <summary>
            /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
            /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
            /// memory for the expanded distribution instance.
            /// </summary>
            /// <param name="batch_shape">Tthe desired expanded size.</param>
            /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
            public override Distribution expand(Size batch_shape, Distribution instance = null)
            {
                var cExp = concentration.expand(batch_shape);
                var cExpR = cExp.reciprocal();
                var nScale = scale.expand(batch_shape);

                var transforms = new torch.distributions.transforms.Transform[] {
                    new distributions.transforms.PowerTransform(cExpR),
                    new torch.distributions.transforms.AffineTransform(0, nScale)
                };

                var newDistribution = ((instance == null)
                    ? new Weibull(nScale, cExp, cExpR, base_distribution.expand(batch_shape), transforms, generator)
                    : instance) as Weibull;
                return newDistribution;
            }
        }
    }


    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Samples from a two-parameter Weibull distribution.
            /// </summary>
            /// <param name="concentration">Concentration parameter of distribution (k/shape).</param>
            /// <param name="scale">Scale parameter of the distribution.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Weibull Weibull(Tensor scale, Tensor concentration, torch.Generator generator = null)
            {
                var locScale = torch.broadcast_tensors(scale, concentration);
                scale = locScale[0];
                concentration = locScale[1];
                var concentration_reciprocal = concentration.reciprocal();

                var base_dist = Exponential(torch.ones_like(scale), generator);
                var transforms = new torch.distributions.transforms.Transform[] {
                    new torch.distributions.transforms.PowerTransform(exponent: concentration_reciprocal),
                    new torch.distributions.transforms.AffineTransform(0, -torch.ones_like(scale))
                };
                return new Weibull(scale, concentration, concentration_reciprocal, base_dist, transforms, generator);
            }
        }
    }
}
