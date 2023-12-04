// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;
    using static torch.distributions;

    namespace Modules
    {
        public class LogNormal : TransformedDistribution
        {
            internal LogNormal(Tensor loc, Tensor scale, torch.Generator generator = null) :
                base(Normal(loc, scale, generator), new torch.distributions.transforms.Transform[] { new torch.distributions.transforms.ExpTransform() }, generator)
            {
                this.loc = loc.alias().DetachFromDisposeScope();
                this.scale = scale.alias().DetachFromDisposeScope();
            }

            public Tensor loc { get; private set; }

            public Tensor scale { get; private set; }

            public override Tensor mean => torch.WrappedTensorDisposeScope(() => (loc + scale.pow(2) / 2).exp());

            public override Tensor mode => torch.WrappedTensorDisposeScope(() => (loc - scale.square()).exp());

            public override Tensor variance => torch.WrappedTensorDisposeScope(() => (scale.pow(2).exp() - 1) * (2 * loc + scale.pow(2)).exp());

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    loc?.Dispose();
                    scale?.Dispose();
                }
                base.Dispose(disposing);
            }

            public override Tensor entropy()
            {
                return base_distribution.entropy() + loc;
            }

            public override Distribution expand(Size batch_shape, Distribution instance = null)
            {
                var newDistribution = ((instance == null)
                    ? new LogNormal(loc.expand(batch_shape), scale.expand(batch_shape), generator)
                    : instance) as LogNormal;
                return base.expand(batch_shape, newDistribution);
            }
        }
    }


    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a log-normal distribution parameterized by `loc` and `scale`
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Standard deviation.</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static LogNormal LogNormal(Tensor loc, Tensor scale, torch.Generator generator = null)
            {
                
                return new LogNormal(loc, scale, generator);
            }
        }
    }
}
