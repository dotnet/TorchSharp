// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
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
        public class FisherSnedecor : torch.distributions.Distribution
        {

            public override Tensor mean {
                get {
                    var df2 = this.df2.clone();
                    df2[df2 <= 2] = torch.tensor(float.NaN);
                    return df2 / (df2 - 2);
                }
            }

            public override Tensor variance {
                get {
                    var df2 = this.df2.clone();
                    df2[df2 <= 4] = torch.tensor(float.NaN);
                    return 2 * df2.pow(2) * (this.df1 + df2 - 2) / (this.df1 * (df2 - 2).pow(2) * (df2 - 4));
                }
            }

            public FisherSnedecor(Tensor df1, Tensor df2) 
            {
                var bcast = torch.broadcast_tensors(df1, df2);
                this.df1 = bcast[0];
                this.df2 = bcast[1];
                this.gamma1 = new Gamma(this.df1 * 0.5, this.df1);
                this.gamma2 = new Gamma(this.df2 * 0.5, this.df2);
                this.batch_shape = this.df1.size();
            }

            private Tensor df1;
            private Tensor df2;
            private Gamma gamma1;
            private Gamma gamma2;

            public override Tensor rsample(params long[] sample_shape)
            {
                var shape = ExtendedShape(sample_shape);
                var X1 = gamma1.rsample(sample_shape).view(shape);
                var X2 = gamma2.rsample(sample_shape).view(shape);

                var tiny = torch.finfo(X2.dtype).tiny;
                X2.clamp_(min: tiny);
                var Y = X1 / X2;
                
                return Y.clamp_(min: tiny);
            }

            public override Tensor log_prob(Tensor value)
            {
                var ct1 = this.df1 * 0.5;
                var ct2 = this.df2 * 0.5;
                var ct3 = this.df1 / this.df2;
                var t1 = (ct1 + ct2).lgamma() - ct1.lgamma() - ct2.lgamma();
                var t2 = ct1 * ct3.log() + (ct1 - 1) * torch.log(value);
                var t3 = (ct1 + ct2) * torch.log1p(ct3 * value);
                return t1 + t2 - t3;
            }

            public override distributions.Distribution expand(long[] batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is FisherSnedecor))
                    throw new ArgumentException("expand(): 'instance' must be a FisherSnedecor distribution");

                var df1 = this.df1.expand(batch_shape);
                var df2 = this.df2.expand(batch_shape);

                var newDistribution = ((instance == null) ? new FisherSnedecor(df1, df2) : instance) as FisherSnedecor;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.df1 = df1;
                    newDistribution.df2 = df2;
                    newDistribution.gamma1 = this.gamma1.expand(batch_shape) as Gamma;
                    newDistribution.gamma2 = this.gamma2.expand(batch_shape) as Gamma;

                }
                return newDistribution;
            }

            public override Tensor entropy()
            {
                throw new NotImplementedException();
            }
        }

    }
    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a Fisher-Snedecor distribution parameterized by `df1` and `df2`.
            /// </summary>
            /// <param name="df1">Degrees of freedom parameter 1</param>
            /// <param name="df2">Degrees of freedom parameter 2</param>
            /// <returns></returns>
            public static FisherSnedecor FisherSnedecor(Tensor df1, Tensor df2)
            {
                return new FisherSnedecor(df1, df2);
            }
        }
    }
}
