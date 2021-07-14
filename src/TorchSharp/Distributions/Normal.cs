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
        public class Normal : torch.distributions.Distribution
        {

            public override Tensor mean => loc;

            public override Tensor stddev => scale;

            public override Tensor variance => scale.pow(2);


            public Normal(Tensor loc, Tensor scale)
            {
                this.batch_shape = loc.size();
                var locScale = torch.broadcast_tensors(loc, scale);
                this.loc = locScale[0];
                this.scale = locScale[1];
            }

            private Tensor loc;
            private Tensor scale;

            public override Tensor sample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                using (torch.no_grad()) {
                    return torch.normal(loc.expand(shape), scale.expand(shape));
                }
            }

            public override Tensor rsample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                var eps = torch.empty(shape, dtype: loc.dtype, device: loc.device).normal_();
                return loc + eps * scale;
            }

            public override Tensor log_prob(Tensor value)
            {
                var v = scale.pow(2);
                var log_scale = scale.log();
                return -((value - loc).pow(2)) / (2 * v) - log_scale - Math.Log(Math.Sqrt(2 * Math.PI));
            }

            public override Tensor entropy()
            {
                return 0.5 + 0.5 * Math.Log(2 * Math.PI) + torch.log(scale);
            }

            public override Tensor cdf(Tensor value)
            {
                return 0.5 * (1 + torch.special.erf((value - loc) * scale.reciprocal() / Math.Sqrt(2)));
            }

            public override Tensor icdf(Tensor value)
            {
                return loc + scale * torch.special.erfinv(2 * value - 1) * Math.Sqrt(2);
            }
            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Normal))
                    throw new ArgumentException("expand(): 'instance' must be a Normal distribution");

                var newDistribution = ((instance == null) ? new Normal(loc.expand(batch_shape), scale.expand(batch_shape)) : instance) as Normal;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.loc = loc.expand(batch_shape);
                    newDistribution.scale = scale.expand(batch_shape);
                }
                return newDistribution;
            }
        }

    }
    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Samples from a Normal (Lorentz) distribution. The distribution of the ratio of
            /// independent normally distributed random variables with means `0` follows a Normal distribution.
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Half width at half maximum.</param>
            /// <returns></returns>
            public static Normal Normal(Tensor loc, Tensor scale)
            {
                return new Normal(loc, scale);
            }
        }
    }
}
