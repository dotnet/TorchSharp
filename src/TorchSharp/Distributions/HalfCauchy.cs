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
        public class HalfCauchy : TransformedDistribution
        {
            internal HalfCauchy(Tensor scale, torch.Generator generator = null) :
                base(Cauchy(torch.tensor(0).to(scale.dtype), scale, generator), new torch.distributions.transforms.AbsTransform(), generator)
            {
                this.scale = scale;
            }

            public Tensor scale { get; private set; }

            public override Tensor mean => scale * Math.Sqrt(2 / Math.PI);

            public override Tensor mode => torch.zeros_like(scale);

            public override Tensor variance => base_distribution.variance;

            public override Tensor log_prob(Tensor value)
            {
                value = torch.as_tensor(value, scale.dtype, scale.device);
                var lp = base_distribution.log_prob(value) + Math.Log(2);
                lp[value.expand(lp.shape) < 0] = Double.NegativeInfinity;
                return lp;
            }

            public override Tensor cdf(Tensor value)
            {
                return 2 * base_distribution.cdf(value) - 1;
            }

            public override Tensor icdf(Tensor value)
            {
                return base_distribution.icdf((value + 1) / 2);
            }

            public override Tensor entropy()
            {
                return base_distribution.entropy() - Math.Log(2);
            }

            public override Distribution expand(Size batch_shape, Distribution instance = null)
            {
                var newDistribution = ((instance == null)
                    ? new HalfCauchy(scale.expand(batch_shape), generator)
                    : instance) as HalfCauchy;
                return base.expand(batch_shape, newDistribution);
            }
        }
    }


    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a half-normal distribution parameterized by `scale`
            /// </summary>
            /// <param name="scale">Scale parameter of the distribution.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static HalfCauchy HalfCauchy(Tensor scale, torch.Generator generator = null)
            {
                
                return new HalfCauchy(scale, generator);
            }
        }
    }
}
