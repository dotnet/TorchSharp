// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;

#nullable enable
namespace TorchSharp
{

    public static partial class torch
    {
        public static partial class distributions
        {
            public static partial class transforms
            {

                /// <summary>
                /// Abstract class for invertable transformations with computable log det jacobians.
                ///
                /// They are primarily used in torch.distributions.TransformedDistribution.
                /// </summary>
                /// <remarks>
                /// Derived classes should implement one or both of 'forward()' or 'inverse()'.
                /// Derived classes that set `bijective=true` should also implement 'log_abs_det_jacobian()'
                /// </remarks>
                public abstract class Transform
                {
                    protected bool _bijective = false;

                    protected constraints.Constraint _domain = null!;

                    protected constraints.Constraint _codomain = null!;

                    protected Transform? _inv = null;

                    public virtual int event_dim {
                        get {
                            if (_domain.event_dim == codomain.event_dim)
                                return _domain.event_dim;
                            throw new InvalidOperationException("Please use either .domain.event_dim or .codomain.event_dim");
                        }
                    }

                    public virtual Transform inv {
                        get {
                            Transform? result = null;
                            if (this._inv != null)
                                result = _inv;
                            if (result == null) {
                                result = new _InverseTransform(this);
                                _inv = result;
                            }
                            return result;
                        }
                    }

                    public virtual constraints.Constraint domain {
                        get {
                            return _domain;
                        }
                    }

                    public virtual constraints.Constraint codomain {
                        get {
                            return _codomain;
                        }
                    }

                    public virtual bool bijective {
                        get {
                            return _bijective;
                        }
                    }

                    public Tensor sign {
                        get {
                            return _sign();
                        }
                    }

                    protected internal virtual Tensor _sign()
                    {
                        throw new NotImplementedException();
                    }

                    protected internal abstract Tensor _call(Tensor x);

                    protected internal abstract Tensor _inverse(Tensor y);

                    protected internal virtual Tensor log_abs_det_jacobian(Tensor x, Tensor y)
                    {
                        throw new NotImplementedException();
                    }

                    public Tensor forward(Tensor x) => this._call(x);

                    public virtual long[] forward_shape(long[] shape) => shape;

                    public virtual long[] inverse_shape(long[] shape) => shape;

                    protected internal static Tensor _sum_rightmost(Tensor value, int dim)
                    {
                        if (dim == 0) return value;
                        var required_shape = new long[value.shape.Length - dim + 1];
                        var i = 0;
                        for (; i < value.shape.Length - dim; i++) required_shape[i] = value.shape[i];
                        required_shape[i] = -1;
                        return value.reshape(required_shape).sum(-1);
                    }
                }


                internal class _InverseTransform : Transform
                {
                    public _InverseTransform(torch.distributions.transforms.Transform transform)
                    {
                        this._inv = transform;
                    }

                    public override constraints.Constraint domain {
                        get {
                            return _inv!.domain;
                        }
                    }

                    public override constraints.Constraint codomain {
                        get {
                            return _inv!.codomain;
                        }
                    }

                    public override bool bijective {
                        get {
                            return _inv!.bijective;
                        }
                    }

                    public override int event_dim => base.event_dim;

                    public override Transform inv => base.inv;

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y)
                    {
                        return -_inv!.log_abs_det_jacobian(y, x);
                    }

                    protected internal override Tensor _call(Tensor x)
                    {
                        return _inv!._inverse(x);
                    }

                    protected internal override Tensor _inverse(Tensor y)
                    {
                        return _inv!._call(y);
                    }

                    protected internal override Tensor _sign()
                    {
                        return _inv!.sign;
                    }

                    public override long[] forward_shape(long[] shape)
                    {
                        return _inv!.forward_shape(shape);
                    }

                    public override long[] inverse_shape(long[] shape)
                    {
                        return _inv!.inverse_shape(shape);
                    }
                }

                public class ComposeTransform : Transform
                {
                    public ComposeTransform(IEnumerable<Transform> parts, int cache_size = 0)
                    {
                        _parts = parts.ToArray();
                        _reverse_parts = parts.Reverse().ToArray();
                    }

                    private Transform[] _parts;
                    private Transform[] _reverse_parts;

                    public override int event_dim => base.event_dim;

                    public override Transform inv {
                        get {
                            Transform? i = _inv;

                            if (i == null) {
                                i = new ComposeTransform(_reverse_parts.Select(p => p.inv));
                                _inv = i;
                            }
                            return _inv!;
                        }
                    }

                    public override constraints.Constraint domain {
                        get {
                            if (_parts == null) return constraints.real;

                            var cnt = _parts.Length;
                            var d = _parts[0].domain;

                            var ed = _parts[cnt - 1].codomain.event_dim;
                            foreach (var part in _reverse_parts) {
                                ed += part.domain.event_dim - part.codomain.event_dim;
                                ed = ed < part.domain.event_dim ? part.domain.event_dim : ed;
                            }
                            if (ed > d.event_dim) {
                                d = constraints.independent(domain, ed - domain.event_dim);
                            }
                            return d;
                        }
                    }

                    public override constraints.Constraint codomain {
                        get {
                            if (_parts == null) return constraints.real;

                            var cnt = _parts.Length;
                            var cod = _parts[cnt - 1].domain;

                            var ed = _parts[0].domain.event_dim;
                            foreach (var part in _parts) {
                                ed += part.codomain.event_dim - part.domain.event_dim;
                                ed = ed < part.codomain.event_dim ? part.codomain.event_dim : ed;
                            }
                            if (ed > cod.event_dim) {
                                cod = constraints.independent(codomain, ed - codomain.event_dim);
                            }
                            return cod;
                        }
                    }

                    public override bool bijective => _parts.All(p => p.bijective);

                    protected internal override Tensor _sign()
                    {
                        Tensor s = 1;
                        foreach (var p in _parts) s *= p.sign;
                        return s;
                    }

                    protected internal override Tensor _call(Tensor x)
                    {
                        using var _ = torch.NewDisposeScope();
                        foreach (var p in _parts) {
                            x = p._call(x);
                        }
                        return x.MoveToOuterDisposeScope();
                    }

                    protected internal override Tensor _inverse(Tensor y)
                    {
                        throw new NotImplementedException();
                    }

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y)
                    {
                        if (_parts == null) {
                            return torch.zeros_like(x);
                        }

                        using var _ = torch.NewDisposeScope();

                        var xs = new List<Tensor>();
                        xs.Add(x);
                        for (int i = 0; i < _parts.Length-1; i++) {
                            var part = _parts[i];
                            xs.Add(part._call(xs[i]));
                        }
                        xs.Add(y);

                        var terms = new List<Tensor>();
                        var event_dim = domain.event_dim;

                        for (int i = 0; i < _parts.Length - 1; i++) {
                            var part = _parts[i];
                            var x1 = xs[i];
                            var y1 = xs[i + 1];

                            terms.Add(_sum_rightmost(part.log_abs_det_jacobian(x1, y), event_dim - part.domain.event_dim));
                            event_dim += part.codomain.event_dim - part.domain.event_dim;
                        }

                        Tensor result = terms[0];
                        for (var i = 1; i < terms.Count; i++) {
                            result = result + terms[i];
                        }
                        return result.MoveToOuterDisposeScope();
                    }

                    public override long[] forward_shape(long[] shape)
                    {
                        foreach (var p in _parts) {
                            shape = p.forward_shape(shape);
                        }
                        return shape;
                    }

                    public override long[] inverse_shape(long[] shape)
                    {
                        foreach (var p in _reverse_parts) {
                            shape = p.forward_shape(shape);
                        }
                        return shape;
                    }
                }

                public class IndepdenentTransform : Transform
                {
                    private Transform base_transform;
                    private int reinterpreted_batch_dims;

                    public IndepdenentTransform(Transform base_transform, int reinterpreted_batch_dims)
                    {
                        this.base_transform = base_transform;
                        this.reinterpreted_batch_dims = reinterpreted_batch_dims;
                    }

                    public override int event_dim => base.event_dim;

                    public override Transform inv => base.inv;

                    public override constraints.Constraint domain => constraints.independent(base_transform.domain, reinterpreted_batch_dims);

                    public override constraints.Constraint codomain => constraints.independent(base_transform.codomain, reinterpreted_batch_dims);

                    public override bool bijective => base_transform.bijective;

                    public override long[] forward_shape(long[] shape)
                    {
                        return base_transform.forward_shape(shape);
                    }

                    public override long[] inverse_shape(long[] shape)
                    {
                        return base_transform.inverse_shape(shape);
                    }

                    protected internal override Tensor _sign() => base_transform._sign();

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y)
                    {
                        using var _ = torch.NewDisposeScope();
                        var result = base_transform.log_abs_det_jacobian(x, y);
                        result = _sum_rightmost(result, reinterpreted_batch_dims);
                        return result.MoveToOuterDisposeScope();
                    }

                    protected internal override Tensor _call(Tensor x)
                    {
                        if (x.dim() < domain.event_dim)
                            throw new ArgumentException("Too few dimensions on input");
                        return base_transform._call(x);
                    }

                    protected internal override Tensor _inverse(Tensor y)
                    {
                        if (y.dim() < codomain.event_dim)
                            throw new ArgumentException("Too few dimensions on input");
                        return base_transform._inverse(y);
                    }
                }

                public class ReshapeTransform : Transform
                {
                    private Size in_shape;
                    private Size out_shape;

                    public ReshapeTransform(long[] in_shape, long[] out_shape)
                    {
                        this.in_shape = in_shape;
                        this.out_shape = out_shape;
                        if (this.in_shape.numel() != this.out_shape.numel())
                            throw new ArgumentException("in_shape, out_shape have different numbers of elements");

                    }
                    public override bool bijective => true;

                    public override constraints.Constraint domain => constraints.independent(constraints.real, in_shape.Length);

                    public override constraints.Constraint codomain => constraints.independent(constraints.real, out_shape.Length);

                    public override long[] forward_shape(long[] shape)
                    {
                        return base.forward_shape(shape);
                    }

                    public override long[] inverse_shape(long[] shape)
                    {
                        return base.inverse_shape(shape);
                    }

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y)
                    {
                        long inLen = x.shape.Length - (x.dim() - in_shape.Length);
                        var batch_shape = new long[inLen];
                        for (var i = 0; i < inLen; i++) {
                            batch_shape[i] = x.shape[i];
                        }
                        return x.new_zeros(batch_shape);
                    }

                    protected internal override Tensor _call(Tensor x)
                    {
                        long inLen = x.shape.Length - (x.dim() - in_shape.Length);
                        long otLen = out_shape.Length;
                        var batch_shape = new long[inLen+otLen];
                        for (var i = 0; i < inLen; i++) {
                            batch_shape[i] = x.shape[i];
                        }
                        for (var i = 0; i < otLen; i++) {
                            batch_shape[i+inLen] = out_shape[i];
                        }
                        return x.reshape(batch_shape);
                    }

                    protected internal override Tensor _inverse(Tensor y)
                    {
                        long otLen = y.shape.Length - (y.dim() - out_shape.Length);
                        long inLen = in_shape.Length;
                        var batch_shape = new long[inLen + otLen];
                        for (var i = 0; i < otLen; i++) {
                            batch_shape[i] = y.shape[i];
                        }
                        for (var i = 0; i < inLen; i++) {
                            batch_shape[i + otLen] = in_shape[i];
                        }
                        return y.reshape(batch_shape);
                    }
                }

                public class ExpTransform : Transform
                {
                    public override constraints.Constraint domain => constraints.real;

                    public override constraints.Constraint codomain => constraints.positive;

                    public override bool bijective => true;

                    protected internal override Tensor _sign() => 1;

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y) => x;
                    protected internal override Tensor _call(Tensor x) => x.exp();

                    protected internal override Tensor _inverse(Tensor y) => y.log();
                }

                public class LogTransform : Transform
                {
                    public override constraints.Constraint domain => constraints.positive;

                    public override constraints.Constraint codomain => constraints.real;

                    public override bool bijective => true;

                    protected internal override Tensor _sign() => 1;

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y) => -x;

                    protected internal override Tensor _call(Tensor x) => x.log();

                    protected internal override Tensor _inverse(Tensor y) => y.exp();
                }

                public class PowerTransform : Transform
                {
                    private Tensor exponent;

                    public PowerTransform(Tensor exponent)
                    {
                        this.exponent = exponent;
                    }

                    public override constraints.Constraint domain => constraints.positive;

                    public override constraints.Constraint codomain => constraints.positive;

                    public override bool bijective => true;

                    protected internal override Tensor _sign() => 1;

                    public override long[] forward_shape(long[] shape)
                    {
                        return torch.broadcast_shapes(shape, exponent.shape).Shape;
                    }

                    public override long[] inverse_shape(long[] shape)
                    {
                        return torch.broadcast_shapes(shape, exponent.shape).Shape;
                    }

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y)
                    {
                        return torch.WrappedTensorDisposeScope(() => (exponent * y / x).abs().log());
                    }

                    protected internal override Tensor _call(Tensor x)
                    {
                        return x.pow(exponent);
                    }

                    protected internal override Tensor _inverse(Tensor y)
                    {
                        return y.pow(1 / exponent);
                    }
                }

                public class SigmoidTransform : Transform
                {
                    public override constraints.Constraint domain => constraints.real;

                    public override constraints.Constraint codomain => constraints.unit_interval;

                    public override bool bijective => true;

                    protected internal override Tensor _sign() => 1;

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y) => -nn.functional.softplus(-x) - nn.functional.softplus(x);

                    protected internal override Tensor _call(Tensor x)
                    {
                        var finfo = torch.finfo(x.dtype);
                        return torch.WrappedTensorDisposeScope(() => torch.clamp(torch.sigmoid(x), min: finfo.tiny, max: 1 - finfo.eps));
                    }

                    protected internal override Tensor _inverse(Tensor y)
                    {
                        using var _ = torch.NewDisposeScope();
                        var finfo = torch.finfo(y.dtype);
                        y = y.clamp(min: finfo.tiny, max: 1 - finfo.eps);
                        return (y.log() - (-y).log1p()).MoveToOuterDisposeScope();
                    }
                }

                public class SoftplusTransform : Transform
                {
                    public override constraints.Constraint domain => constraints.real;

                    public override constraints.Constraint codomain => constraints.positive;

                    public override bool bijective => true;

                    protected internal override Tensor _sign() => 1;

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y) => -nn.functional.softplus(-x);

                    protected internal override Tensor _call(Tensor x) => nn.functional.softplus(-x);

                    protected internal override Tensor _inverse(Tensor y) => (-y).expm1().neg().log() + y;
                }

                public class SoftmaxTransform : Transform
                {
                    public override constraints.Constraint domain => constraints.real_vector;

                    public override constraints.Constraint codomain => constraints.simplex;

                    public override bool bijective => true;

                    protected internal override Tensor _call(Tensor x)
                    {
                        var logprobs = x;
                        var probs = (logprobs - logprobs.max(-1, true).values).exp();
                        return probs / probs.sum(-1, true);
                    }

                    protected internal override Tensor _inverse(Tensor y)
                    {
                        return y.log();
                    }
                }


                public class TanhTransform : Transform
                {
                    public override constraints.Constraint domain => constraints.real;

                    public override constraints.Constraint codomain => constraints.interval(-1.0, 1.0);

                    public override bool bijective => true;

                    protected internal override Tensor _sign() => 1;

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y) => torch.WrappedTensorDisposeScope(() => 2.0 * (Math.Log(2.0) - x - -nn.functional.softplus(-2.0 * x)));

                    protected internal override Tensor _call(Tensor x) => x.tanh();

                    protected internal override Tensor _inverse(Tensor y) => y.atanh();
                }

                public class AbsTransform : Transform
                {
                    public override constraints.Constraint domain => constraints.real;

                    public override constraints.Constraint codomain => constraints.positive;

                    protected internal override Tensor _call(Tensor x) => x.abs();

                    protected internal override Tensor _inverse(Tensor y) => y;
                }

                public class AffineTransform : Transform
                {
                    private Tensor loc;
                    private Tensor scale;
                    private int _event_dim;

                    public AffineTransform(double loc, double scale, int event_dim = 0)
                    {
                        this._event_dim = event_dim;
                        this.loc = loc;
                        this.scale = scale;
                    }

                    public AffineTransform (Tensor loc, Tensor scale, int event_dim = 0)
                    {
                        this._event_dim = event_dim;
                        this.loc = loc;
                        this.scale = scale;
                    }

                    public override int event_dim => _event_dim;

                    public override bool bijective => true;

                    public override constraints.Constraint domain =>
                        _event_dim == 0 ? constraints.real : constraints.independent(constraints.real, event_dim);

                    public override constraints.Constraint codomain =>
                        _event_dim == 0 ? constraints.real : constraints.independent(constraints.real, event_dim);

                    public override long[] forward_shape(long[] shape)
                    {
                        return torch.broadcast_shapes(shape, loc.shape, scale.shape).Shape;
                    }

                    public override long[] inverse_shape(long[] shape)
                    {
                        return torch.broadcast_shapes(shape, loc.shape, scale.shape).Shape;
                    }

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y)
                    {
                        using var _ = torch.NewDisposeScope();
                        var shape = x.shape;
                        var scale = this.scale;
                        var result = torch.abs(scale).log();

                        if (_event_dim > 0) {
                            var result_size = new long[result.shape.Length - _event_dim + 1];
                            int i = 0;
                            for (; i < result.shape.Length - _event_dim; i++) result_size[i] = result.shape[i];
                            result_size[i] = -1;
                            result = result.view(result_size).sum(-1);
                            var nshape = new long[shape.Length];
                            for (; i < shape.Length - _event_dim; i++) nshape[i] = shape[i];
                            shape = nshape;
                        }
                        return result.expand(shape).MoveToOuterDisposeScope();
                    }

                    protected internal override Tensor _call(Tensor x) => torch.WrappedTensorDisposeScope(() => loc + scale * x);

                    protected internal override Tensor _inverse(Tensor y) => torch.WrappedTensorDisposeScope(() => (y - loc) / scale);

                    protected internal override Tensor _sign()
                    {
                        return scale.sign();
                    }
                }
            }
        }
    }
}
