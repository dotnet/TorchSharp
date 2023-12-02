// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using static TorchSharp.torch;

namespace TorchSharp
{
    using System.Linq;
    using static torch.distributions;

    namespace Modules
    {
        /// <summary>
        /// Extension of the Distribution class, which applies a sequence of Transforms to a base distribution.
        /// </summary>
        public class TransformedDistribution : Distribution
        {
            protected Distribution base_distribution;
            private torch.distributions.transforms.Transform[] transforms;
            private torch.distributions.transforms.Transform[] reverse_transforms;

            public TransformedDistribution(torch.Generator generator = null) : base(generator)
            {
            }

            public TransformedDistribution(Distribution base_distribution, torch.distributions.transforms.Transform transform, torch.Generator generator = null) :
                this(base_distribution, new torch.distributions.transforms.Transform[] { transform}, generator)
            {
                _init(base_distribution, transforms);
            }

            public TransformedDistribution(Distribution base_distribution, IEnumerable<torch.distributions.transforms.Transform> transforms, torch.Generator generator = null) : base(generator)
            {
                _init(base_distribution, transforms.ToArray());
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    base_distribution?.Dispose();
                }
                base.Dispose(disposing);
            }

            protected void _init(Distribution base_distribution, torch.distributions.transforms.Transform[] transforms)
            {
                this.transforms = transforms;
                this.reverse_transforms = transforms.Reverse().ToArray();

                var base_shape = base_distribution.batch_shape.Concat(base_distribution.event_shape).ToArray();
                var base_event_dim = base_distribution.event_shape.Length;
                var transform = new distributions.transforms.ComposeTransform(transforms.ToArray());
                var domain_event_dim = transform.domain.event_dim;
                var shape = transform.forward_shape(base_shape);
                var expanded_base_shape = transform.inverse_shape(shape);

                if (base_shape != expanded_base_shape) {
                    var base_batch_shape = expanded_base_shape.Take(expanded_base_shape.Length - base_event_dim).ToArray();
                    base_distribution = base_distribution.expand(base_batch_shape);
                }

                var reinterpreted_batch_ndims = domain_event_dim - base_event_dim;
                if (reinterpreted_batch_ndims > 0) {
                    //base_distribution = new distributions.Inde
                }

                this.base_distribution = base_distribution;

                var event_dim = transform.codomain.event_dim;
                event_dim += (base_event_dim - domain_event_dim > 0) ? base_event_dim - domain_event_dim : 0;

                var cut = shape.Length - event_dim;
                this.batch_shape = shape.Take(cut).ToArray();
                this.event_shape = shape.Skip(cut).ToArray();
            }

            public override Tensor mean => new Tensor(IntPtr.Zero);

            public override Tensor mode => base.mode;

            public override Tensor variance => new Tensor(IntPtr.Zero);

            public override Tensor stddev => base.stddev;

            public override Tensor cdf(Tensor value)
            {
                using var scope = torch.NewDisposeScope();
                foreach (var transform in reverse_transforms) {
                    value = transform._inverse(value);
                }
                value = base_distribution.cdf(value);
                value = monotonize_cdf(value);
                return value.MoveToOuterDisposeScope();
            }

            public override Tensor entropy()
            {
                throw new NotImplementedException();
            }

            public override distributions.Distribution expand(Size batch_shape, distributions.Distribution instance = null)
            {
                if (instance != null && !(instance is TransformedDistribution))
                    throw new ArgumentException("expand(): 'instance' must be a TransformedDistribution distribution");

                var shape = batch_shape.Concat(event_shape).ToArray();
                foreach (var t in reverse_transforms) {
                    shape = t.inverse_shape(shape);
                }
                int baseLen = shape.Length - base_distribution.event_shape.Length;
                var base_batch_shape = new long[baseLen];
                for (var i = 0; i < baseLen; i++) base_batch_shape[i] = shape[i];

                var newDistribution = ((instance == null) ?
                    new TransformedDistribution(base_distribution.expand(base_batch_shape), transforms) :
                    instance) as TransformedDistribution;

                newDistribution.batch_shape = batch_shape;
                newDistribution.event_shape = event_shape;
                if (newDistribution == instance) {
                    newDistribution.transforms = transforms;
                    newDistribution.base_distribution = base_distribution.expand(base_batch_shape);
                }
                return newDistribution;
            }

            public override Tensor icdf(Tensor value)
            {
                using var scope = torch.NewDisposeScope();
                value = monotonize_cdf(value);
                value = base_distribution.icdf(value);
                foreach (var transform in transforms) {
                    value = transform.forward(value);
                }
                return value.MoveToOuterDisposeScope();
            }

            public override Tensor log_prob(Tensor value)
            {
                using var scope = torch.NewDisposeScope();
                var event_dim = event_shape.Length;
                Tensor lp = 0.0;
                var y = value;
                foreach (var t in reverse_transforms) {
                    var x = t._inverse(y);
                    event_dim += t.domain.event_dim - t.codomain.event_dim;
                    lp = lp - distributions.transforms.Transform._sum_rightmost(t.log_abs_det_jacobian(x, y), event_dim - t.domain.event_dim);
                    y = x;
                }
                lp = lp + distributions.transforms.Transform._sum_rightmost(base_distribution.log_prob(y), event_dim - base_distribution.event_shape.Length);
                return lp.MoveToOuterDisposeScope();
            }

            public override Tensor rsample(params long[] sample_shape)
            {
                using var scope = torch.NewDisposeScope();
                using var _ = torch.no_grad();
                var x = base_distribution.rsample(sample_shape);
                foreach (var t in transforms) {
                    x = t.forward(x);
                }
                return x.MoveToOuterDisposeScope();
            }

            public override Tensor sample(params long[] sample_shape)
            {
                using var scope = torch.NewDisposeScope();
                using var _ = torch.no_grad();
                var x = base_distribution.sample(sample_shape);
                foreach (var t in transforms) {
                    x = t.forward(x);
                }
                return x.MoveToOuterDisposeScope();
            }

            private Tensor monotonize_cdf(Tensor value)
            {
                Tensor sign = 1;
                foreach (var transform in transforms) {
                    sign = sign * transform.sign;
                }
                return sign * (value - 0.5) + 0.5;
            }
        }
    }
}
