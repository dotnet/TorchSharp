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
        public class HalfNormal : TransformedDistribution
        {
            internal HalfNormal(Tensor scale, torch.Generator generator = null) :
                base(Normal(0, scale), new torch.distributions.transforms.Transform[] { new torch.distributions.transforms.AbsTransform() }, generator)
            {
                this.scale = scale;
            }

            public Tensor scale { get; private set; }

            public override Tensor mean => scale * Math.Sqrt(2 / Math.PI);

            public override Tensor mode => torch.zeros_like(scale);

            public override Tensor variance => scale.pow(2) * (1 - 2 / Math.PI);

            public override Tensor log_prob(Tensor value)
            {
                var lp = base_distribution.log_prob(value) + Math.Log(2);
                lp[value.expand(lp.shape) < 0] = Double.NegativeInfinity;
                return lp;
            }

            public override Distribution expand(long[] batch_shape, Distribution instance = null)
            {
                var newDistribution = ((instance == null)
                    ? new HalfNormal(scale.expand(batch_shape))
                    : instance) as HalfNormal;
                return base.expand(batch_shape, newDistribution);
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
            /// <param name="scale">Scale parameter of the distribution.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static HalfNormal HalfNormal(Tensor scale, torch.Generator generator = null)
            {
                
                return new HalfNormal(scale, generator);
            }
        }
    }
}
