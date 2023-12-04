// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A Fisher-Snedecor distribution parameterized by `df1` and `df2`.
        /// </summary>
        public class FisherSnedecor : torch.distributions.Distribution
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean {
                get {
                    var df2 = this.df2.clone();
                    df2[df2 <= 2] = torch.tensor(float.NaN);
                    return df2 / (df2 - 2);
                }
            }

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance {
                get {
                    using var _ = torch.NewDisposeScope();
                    var df2 = this.df2.clone();
                    df2[df2 <= 4] = torch.tensor(float.NaN);
                    return (2 * df2.pow(2) * (this.df1 + df2 - 2) / (this.df1 * (df2 - 2).pow(2) * (df2 - 4))).MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="df1">Degrees of freedom parameter 1</param>
            /// <param name="df2">Degrees of freedom parameter 2</param>
            /// <param name="generator">An optional random number generator object.</param>
            public FisherSnedecor(Tensor df1, Tensor df2, torch.Generator generator = null) : base(generator)
            {
                var bcast = torch.broadcast_tensors(df1, df2);
                this.df1 = bcast[0].DetachFromDisposeScope();
                this.df2 = bcast[1].DetachFromDisposeScope();
                this.gamma1 = new Gamma(this.df1 * 0.5, this.df1, generator);
                this.gamma2 = new Gamma(this.df2 * 0.5, this.df2, generator);
                this.batch_shape = this.df1.size();
            }

            private Tensor df1;
            private Tensor df2;
            private Gamma gamma1;
            private Gamma gamma2;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    df1?.Dispose();
                    df2?.Dispose();
                    gamma1?.Dispose();
                    gamma2?.Dispose();
                }
                base.Dispose(disposing);
            }

            /// <summary>
            ///  Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples
            ///  if the distribution parameters are batched.
            /// </summary>
            /// <param name="sample_shape">The sample shape.</param>
            public override Tensor rsample(params long[] sample_shape)
            {
                using var _ = torch.NewDisposeScope();
                var shape = ExtendedShape(sample_shape);
                var X1 = gamma1.rsample(sample_shape).view(shape);
                var X2 = gamma2.rsample(sample_shape).view(shape);

                var tiny = torch.finfo(X2.dtype).tiny;
                X2.clamp_(min: tiny);
                var Y = X1 / X2;
                
                return Y.clamp_(min: tiny).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                using var _ = torch.NewDisposeScope();
                var ct1 = this.df1 * 0.5;
                var ct2 = this.df2 * 0.5;
                var ct3 = this.df1 / this.df2;
                var t1 = (ct1 + ct2).lgamma() - ct1.lgamma() - ct2.lgamma();
                var t2 = ct1 * ct3.log() + (ct1 - 1) * torch.log(value);
                var t3 = (ct1 + ct2) * torch.log1p(ct3 * value);
                return (t1 + t2 - t3).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
            /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
            /// memory for the expanded distribution instance.
            /// </summary>
            /// <param name="batch_shape">Tthe desired expanded size.</param>
            /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is FisherSnedecor))
                    throw new ArgumentException("expand(): 'instance' must be a FisherSnedecor distribution");

                var df1 = this.df1.expand(batch_shape);
                var df2 = this.df2.expand(batch_shape);

                var newDistribution = ((instance == null) ? new FisherSnedecor(df1, df2, generator) : instance) as FisherSnedecor;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.df1 = df1;
                    newDistribution.df2 = df2;
                    newDistribution.gamma1 = this.gamma1.expand(batch_shape) as Gamma;
                    newDistribution.gamma2 = this.gamma2.expand(batch_shape) as Gamma;

                }
                return newDistribution;
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
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
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static FisherSnedecor FisherSnedecor(Tensor df1, Tensor df2, torch.Generator generator = null)
            {
                return new FisherSnedecor(df1, df2, generator);
            }
        }
    }
}
