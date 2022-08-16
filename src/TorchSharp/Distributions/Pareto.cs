// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using static TorchSharp.torch;

namespace TorchSharp
{
    using System.Linq;
    using Modules;
    using TorchSharp.torchvision;
    using static torch.distributions;

    namespace Modules
    {
        public class Pareto : TransformedDistribution
        {
            internal Pareto(Tensor scale, Tensor alpha, Distribution base_distribution, torch.distributions.transforms.Transform[] transforms, Generator generator = null) :
                base(base_distribution, transforms, generator)
            {
                this.scale = scale;
                this.alpha = alpha;
            }

            public Tensor scale { get; private set; }

            public Tensor alpha { get; private set; }

            public override Tensor mean {
                get {
                    using var _ = torch.NewDisposeScope();
                    var a = alpha.clamp(min: 2);
                    return (a * scale / (a - 1)).MoveToOuterDisposeScope();
                }
            }

            public override Tensor mode => scale;

            public override Tensor variance {
                get {
                    using var _ = torch.NewDisposeScope();
                    var a = alpha.clamp(min: 2);
                    return (scale.pow(2) * a / ((a - 1).pow(2) * (a - 2))).MoveToOuterDisposeScope();
                }
            }

            public override Tensor log_prob(Tensor value)
            {
                using var _ = torch.NewDisposeScope();
                var lp = base_distribution.log_prob(value) + Math.Log(2);
                lp[value.expand(lp.shape) < 0] = Double.NegativeInfinity;
                return lp.MoveToOuterDisposeScope();
            }

            public override Distribution expand(Size batch_shape, Distribution instance = null)
            {
                var newDistribution = ((instance == null)
                    ? torch.distributions.Pareto(scale.expand(batch_shape), alpha.expand(batch_shape), generator)
                    : instance) as Pareto;
                return base.expand(batch_shape, newDistribution);
            }

            public override Tensor entropy() => torch.WrappedTensorDisposeScope(() => (scale / alpha).log() + 1 + alpha.reciprocal());
        }
    }


    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Samples from a Pareto Type 1 distribution.
            /// </summary>
            /// <param name="scale">Scale parameter of the distribution.</param>
            /// <param name="alpha">Shape parameter of the distribution</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Pareto Pareto(Tensor scale, Tensor alpha, torch.Generator generator = null)
            {
                var scaleAlpha = torch.broadcast_tensors(scale, alpha);
                scale = scaleAlpha[0];
                alpha = scaleAlpha[1];
                var base_dist = Exponential(alpha, generator);
                var transforms = new torch.distributions.transforms.Transform[] {
                    new torch.distributions.transforms.ExpTransform(),
                    new torch.distributions.transforms.AffineTransform(loc:0, scale:scale)
                };
                return new Pareto(scale, alpha, base_dist, transforms, generator);
            }
        }
    }
}
