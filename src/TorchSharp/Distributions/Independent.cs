// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    using System.Linq;
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// Reinterprets some of the batch dims of a distribution as event dims.
        /// </summary>
        public class Independent : torch.distributions.Distribution
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => base_dist.mean;

            public override Tensor mode => base_dist.mode;

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance => base_dist.variance;


            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="base_distribution">A base distribution.</param>
            /// <param name="reinterpreted_batch_ndims">the number of batch dims to reinterpret as event dims</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Independent(torch.distributions.Distribution base_distribution, int reinterpreted_batch_ndims, torch.Generator generator = null) : base(generator)
            {
                var shape = base_distribution.batch_shape.Concat(base_distribution.event_shape).ToArray();
                var event_dim = reinterpreted_batch_ndims + base_distribution.event_shape.Length;
                var batch_shape = shape.Take(shape.Length - event_dim).ToArray();
                var event_shape = shape.Skip(shape.Length - event_dim).ToArray();

                this.base_dist = base_distribution;
                this.reinterpreted_batch_ndims = reinterpreted_batch_ndims;

                _init(batch_shape, event_shape);
            }

            private distributions.Distribution base_dist;
            private int reinterpreted_batch_ndims;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    base_dist?.Dispose();
                }
                base.Dispose(disposing);
            }

            /// <summary>
            /// Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched.
            /// </summary>
            /// <param name="sample_shape"></param>
            public override Tensor sample(params long[] sample_shape)
            {
                return base_dist.sample(sample_shape);
            }


            /// <summary>
            ///  Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of reparameterized samples
            ///  if the distribution parameters are batched.
            /// </summary>
            /// <param name="sample_shape">The sample shape.</param>
            public override Tensor rsample(params long[] sample_shape)
            {
                return base_dist.rsample(sample_shape);
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                var lp = base_dist.log_prob(value);
                return distributions.transforms.Transform._sum_rightmost(lp, reinterpreted_batch_ndims);
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            public override Tensor entropy()
            {
                var ent = base_dist.entropy();
                return distributions.transforms.Transform._sum_rightmost(ent, reinterpreted_batch_ndims);
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
                if (instance != null && !(instance is Independent))
                    throw new ArgumentException("expand(): 'instance' must be a Independent distribution");

                var newDistribution = ((instance == null) ? new Independent(
                    base_dist.expand(batch_shape + event_shape.Slice(0, reinterpreted_batch_ndims)),
                    reinterpreted_batch_ndims,
                    generator) : instance) as Independent;

                if (newDistribution == instance) {
                    newDistribution._init(batch_shape, event_shape);
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
            ///  Reinterprets some of the batch dims of a distribution as event dims.
            ///  This is mainly useful for changing the shape of the result of `log_prob`.
            /// </summary>
            /// <param name="base_distribution">A base distribution.</param>
            /// <param name="reinterpreted_batch_dims">the number of batch dims to reinterpret as event dims</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static Independent Independent(Distribution base_distribution, int reinterpreted_batch_dims, torch.Generator generator = null)
            {
                return new Independent(base_distribution, reinterpreted_batch_dims, generator);
            }
        }
    }
}
