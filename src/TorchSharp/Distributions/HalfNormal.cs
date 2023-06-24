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
        /// Creates a half-normal distribution parameterized by `scale`
        /// </summary>
        public class HalfNormal : TransformedDistribution
        {
            internal HalfNormal(Tensor scale, torch.Generator generator = null) :
                base(Normal(torch.tensor(0).to(scale.dtype), scale, generator), new torch.distributions.transforms.Transform[] { new torch.distributions.transforms.AbsTransform() }, generator)
            {
                this.scale = scale?.alias().DetachFromDisposeScope();
            }

            public Tensor scale { get; private set; }

            public override Tensor mean => scale * Math.Sqrt(2 / Math.PI);

            public override Tensor mode => torch.zeros_like(scale);

            public override Tensor variance => scale.pow(2) * (1 - 2 / Math.PI);

            public override Tensor log_prob(Tensor value)
            {
                using var _ = torch.NewDisposeScope();
                var lp = base_distribution.log_prob(value) + Math.Log(2);
                lp = torch.where(value >= 0, lp, double.NegativeInfinity);
                return lp.MoveToOuterDisposeScope();
            }

            public override Tensor cdf(Tensor value) => WrappedTensorDisposeScope(() => 2 * base_distribution.cdf(value) - 1);

            public override Tensor icdf(Tensor prob) => WrappedTensorDisposeScope(() => base_distribution.icdf((prob + 1) / 2));

            public override Tensor entropy() => WrappedTensorDisposeScope(() => base_distribution.entropy() - Math.Log(2));

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
                    ? new HalfNormal(scale.expand(batch_shape), generator)
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
            /// Creates a half-normal distribution parameterized by `scale`
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
