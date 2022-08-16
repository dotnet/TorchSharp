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
        /// <summary>
        /// Samples from a Gumbel Distribution.
        /// </summary>
        public class Gumbel : TransformedDistribution
        {
            internal Gumbel(Tensor loc, Tensor scale, Distribution base_distribution, torch.distributions.transforms.Transform[] transforms, torch.Generator generator = null) :
                base(base_distribution, transforms, generator)
            {
                this.batch_shape = loc.size();
                var locScale = torch.broadcast_tensors(loc, scale);
                this.loc = locScale[0];
                this.scale = locScale[1];
            }

            private Tensor loc;
            private Tensor scale;

            public override Tensor mean => loc + scale * euler_constant;

            public override Tensor mode => loc;

            public override Tensor variance => stddev.pow(2);

            public override Tensor stddev => pioversqrtsix * scale;

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy() => scale.log() + (1 + euler_constant);

            private readonly double pioversqrtsix = 1.282549830161864095544036359671; // Math.PI / Math.Sqrt(6);
        }
    }


    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Samples from a Gumbel Distribution.
            /// </summary>
            /// <param name="loc">Location parameter of the distribution.</param>
            /// <param name="scale">Scale parameter of the distribution.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Gumbel Gumbel(Tensor loc, Tensor scale, torch.Generator generator = null)
            {
                var locScale = torch.broadcast_tensors(loc, scale);
                loc = locScale[0];
                scale = locScale[1];

                var finfo = torch.finfo(loc.dtype);

                var base_dist = Uniform(torch.full_like(loc, finfo.tiny), torch.full_like(loc, 1 - finfo.eps), generator);
                var transforms = new torch.distributions.transforms.Transform[] {
                    new torch.distributions.transforms.ExpTransform().inv,
                    new torch.distributions.transforms.AffineTransform(0, -torch.ones_like(scale)),
                    new torch.distributions.transforms.ExpTransform().inv,
                    new torch.distributions.transforms.AffineTransform(0, -scale)
                };
                return new Gumbel(loc, scale, base_dist, transforms, generator);
            }
        }
    }
}
