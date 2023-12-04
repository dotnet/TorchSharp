// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace TorchSharp
{
    using System.Reflection;
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A Dirichlet distribution parameterized by shape `concentration` and `rate`.
        /// </summary>
        public class Dirichlet : torch.distributions.ExponentialFamily
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => WrappedTensorDisposeScope(() => concentration / concentration.sum(-1, true));

            public override Tensor mode
            {
                get {
                    using var _ = NewDisposeScope();
                    var concentrationm1 = (concentration - 1).clamp(min: 0.0);
                    var mode = concentrationm1 / concentrationm1.sum(-1, true);
                    var mask = (concentration < 1).all(dim: -1);
                    mode[mask] = torch.nn.functional.one_hot(mode[mask].argmax(dim: -1), concentrationm1.shape[concentrationm1.ndim-1]).to(mode);
                    return mode.MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance {
                get {
                    using var _ = NewDisposeScope();
                    var con0 = concentration.sum(-1, true);
                    return (concentration * (con0 - concentration) / (con0.pow(2) * (con0 + 1))).MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="concentration">Shape parameter of the distribution (often referred to as 'α')</param>
            /// <param name="generator">An optional random number generator object.</param>
            public Dirichlet(Tensor concentration, torch.Generator generator = null) : base(generator)
            {
                var cshape = concentration.shape;
                this.batch_shape = cshape.Take(cshape.Length - 1).ToArray();
                this.event_shape = new long[] { cshape[cshape.Length - 1] };
                this.concentration = concentration.alias().DetachFromDisposeScope();
            }

            internal Tensor concentration;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    concentration?.Dispose();
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
                var shape = ExtendedShape(sample_shape);
                var con = concentration.expand(shape);
                return torch._sample_dirichlet(con, generator);
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                return (concentration - 1).xlogy(value).sum(-1) + torch.lgamma(concentration.sum(-1)) - torch.lgamma(concentration).sum(-1);
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            /// <returns></returns>
            public override Tensor entropy()
            {
                var k = concentration.size(-1);
                var a0 = concentration.sum(-1);

                return torch.lgamma(concentration).sum(-1) - torch.lgamma(a0) - (k - a0) * torch.digamma(a0) - ((concentration - 1.0) * torch.digamma(concentration)).sum(-1);
            }

            /// <summary>
            /// Returns a new distribution instance (or populates an existing instance provided by a derived class) with batch dimensions expanded to
            /// `batch_shape`. This method calls `torch.Tensor.expand()` on the distribution's parameters. As such, this does not allocate new
            /// memory for the expanded distribution instance.
            /// </summary>
            /// <param name="batch_shape">Tthe desired expanded size.</param>
            /// <param name="instance">new instance provided by subclasses that need to override `.expand`.</param>
            /// <returns></returns>
            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is Dirichlet))
                    throw new ArgumentException("expand(): 'instance' must be a Dirichlet distribution");

                var shape = new List<long>();
                shape.AddRange(batch_shape);
                shape.AddRange(event_shape);

                var c = concentration.expand(shape.ToArray());

                var newDistribution = ((instance == null) ? new Dirichlet(c, generator) : instance) as Dirichlet;

                newDistribution.batch_shape = batch_shape;
                if (newDistribution == instance) {
                    newDistribution.concentration = c;
                }
                return newDistribution;
            }

            protected override IList<Tensor> NaturalParams => new Tensor[] { concentration - 1 };

            protected override Tensor MeanCarrierMeasure => new Tensor(IntPtr.Zero);

            protected override Tensor LogNormalizer(params Tensor[] parameters)
            {
                var x = parameters[0];

                return x.lgamma().sum(-1) - x.sum(-1).lgamma();
            }
        }

    }

    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a Dirichlet distribution parameterized by shape `concentration` and `rate`.
            /// </summary>
            /// <param name="concentration">Shape parameter of the distribution (often referred to as 'α')</param>
            /// <param name="generator">An optional random number generator object.</param>
            public static Dirichlet Dirichlet(Tensor concentration, torch.Generator generator = null)
            {
                return new Dirichlet(concentration, generator);
            }
        }
    }
}
