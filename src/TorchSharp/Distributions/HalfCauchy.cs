// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;
    using static torch.distributions;

    namespace Modules
    {
        /// <summary>
        /// Creates a half-Cauchy distribution parameterized by `scale`
        /// </summary>
        public class HalfCauchy : TransformedDistribution
        {
            internal HalfCauchy(Tensor scale, torch.Generator generator = null) :
                base(Cauchy(torch.tensor(0).to(scale.dtype), scale, generator), new torch.distributions.transforms.AbsTransform(), generator)
            {
                this.scale = scale?.alias().DetachFromDisposeScope();
            }

            public Tensor scale { get; private set; }

            public override Tensor mean => torch.full(ExtendedShape(), double.PositiveInfinity, dtype: scale.dtype, device: scale.device);

            public override Tensor mode => torch.zeros_like(scale);

            public override Tensor variance => base_distribution.variance;

            public override Tensor log_prob(Tensor value)
            {
                using var _ = torch.NewDisposeScope();
                value = torch.as_tensor(value, scale.dtype, scale.device);
                var lp = base_distribution.log_prob(value) + Math.Log(2);
                lp = torch.where(value >= 0, lp, double.NegativeInfinity);
                return lp.MoveToOuterDisposeScope();
            }

            public override Tensor cdf(Tensor value)
            {
                return 2 * base_distribution.cdf(value) - 1;
            }

            public override Tensor icdf(Tensor value)
            {
                return base_distribution.icdf((value + 1) / 2);
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy()
            {
                return base_distribution.entropy() - Math.Log(2);
            }

            /// <summary>
            /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
            /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
            /// memory for the expanded distribution instance.
            /// </summary>
            /// <param name="batch_shape">Tthe desired expanded size.</param>
            /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
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
            /// Creates a half-Cauchy distribution parameterized by `scale`
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
