// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;
    using static torch.distributions;

    namespace Modules
    {
        public class Kumaraswamy : TransformedDistribution
        {
            internal Kumaraswamy(Tensor concentration1, Tensor concentration0, torch.Generator generator = null) : base(generator)
            {
                var c1c0 = broadcast_tensors(concentration1, concentration0);
                this.concentration1 = c1c0[0].DetachFromDisposeScope();
                this.concentration0 = c1c0[1].DetachFromDisposeScope();

                _init(Uniform(torch.full_like(this.concentration0, 0), torch.full_like(this.concentration0, 1)),
                      new distributions.transforms.Transform[] {
                          new torch.distributions.transforms.PowerTransform(exponent: this.concentration0.reciprocal()),
                          new torch.distributions.transforms.AffineTransform(loc:1.0, scale:-1.0),
                          new torch.distributions.transforms.PowerTransform(exponent: this.concentration1.reciprocal())
                      });
            }

            public Tensor concentration1 { get; private set; }

            public Tensor concentration0 { get; private set; }

            public override Tensor mean => moments(concentration1, concentration0, 1);

            public override Tensor mode {
                get {
                    using var _ = torch.NewDisposeScope();
                    var log_mode = concentration0.reciprocal() * (-concentration0).log1p() - (-concentration0 * concentration1).log1p();
                    log_mode[(concentration0 < 1) | (concentration1 < 1)] = double.NaN;
                    return log_mode.exp().MoveToOuterDisposeScope();
                }
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    concentration0?.Dispose();
                    concentration1?.Dispose();
                }
                base.Dispose(disposing);
            }

            public override Tensor variance => moments(concentration1, concentration0, 2) - torch.pow(mean, 2);

            public override Tensor entropy()
            {
                using var _ = torch.NewDisposeScope();
                var t1 = (1 - concentration1.reciprocal());
                var t0 = (1 - concentration0.reciprocal());
                var H0 = torch.digamma(concentration0 + 1) + euler_constant;
                return (t0 + t1 * H0 - torch.log(concentration1) - torch.log(concentration0)).MoveToOuterDisposeScope();
            }

            public override Distribution expand(Size batch_shape, Distribution instance = null)
            {
                var newDistribution = ((instance == null)
                    ? new Kumaraswamy(concentration1.expand(batch_shape), concentration0.expand(batch_shape), generator)
                    : instance) as Kumaraswamy;

                if (newDistribution == instance) {
                    newDistribution.concentration0 = concentration0.expand(batch_shape);
                    newDistribution.concentration1 = concentration1.expand(batch_shape);
                }
                return base.expand(batch_shape, newDistribution);
            }

            private Tensor moments(Tensor a, Tensor b, int n)
            {
                using var _ = torch.NewDisposeScope();
                var arg1 = 1 + n / a;
                var log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b);
                return (b * torch.exp(log_value)).MoveToOuterDisposeScope();
            }
        }
    }


    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Samples from a Kumaraswamy distribution.
            /// </summary>
            /// <param name="concentration1">1st concentration parameter of the distribution</param>
            /// <param name="concentration0">2nd concentration parameter of the distribution</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Kumaraswamy Kumaraswamy(Tensor concentration1, Tensor concentration0, torch.Generator generator = null)
            {
                
                return new Kumaraswamy(concentration1, concentration0, generator);
            }
        }
    }
}
