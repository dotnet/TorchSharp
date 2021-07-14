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
        public class Cauchy : torch.distributions.Distribution
        {

            public override Tensor mean => torch.full(ExtendedShape(), double.NaN, dtype: loc.dtype, device: loc.device);

            public override Tensor variance => torch.full(ExtendedShape(), double.NaN, dtype: loc.dtype, device: loc.device);


            public Cauchy(Tensor loc, Tensor scale) 
            {
                this.batch_shape = loc.size();
                var locScale = torch.broadcast_tensors(loc, scale);
                this.loc = locScale[0];
                this.scale = locScale[1];
            }

            private Tensor loc;
            private Tensor scale;

            public override Tensor rsample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                var eps = loc.new_empty(shape).cauchy_();
                return loc + eps * scale;
            }

            public override Tensor log_prob(Tensor value)
            {
                return Math.Log(Math.PI) - scale.log() - (1 + ((value - loc) / scale).pow(2)).log();
            }

            public override Tensor entropy()
            {
                return Math.Log(Math.PI * 4) + scale.log();
            }
            public override Tensor cdf(Tensor value)
            {
                return torch.atan((value - loc) / scale) / Math.PI + 0.5;
            }

            public override Tensor icdf(Tensor value)
            {
                return torch.tan(Math.PI * (value - 0.5)) * scale + loc;
            }
            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Cauchy))
                    throw new ArgumentException("expand(): 'instance' must be a Cauchy distribution");

                var newDistribution = ((instance == null) ? new Cauchy(loc.expand(batch_shape), scale.expand(batch_shape)) : instance) as Cauchy;

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
            /// Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of
            /// independent normally distributed random variables with means `0` follows a Cauchy distribution.
            /// </summary>
            /// <param name="loc">Mode or median of the distribution.</param>
            /// <param name="scale">Half width at half maximum.</param>
            /// <returns></returns>
            public static Cauchy Cauchy(Tensor loc, Tensor scale)
            {
                return new Cauchy(loc, scale);
            }
        }
    }
}
