// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;

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

                    protected constraints.Constraint _domain;

                    protected constraints.Constraint _codomain;

                    protected Transform _inv = null;

                    public virtual int event_dim {
                        get {
                            if (_domain.event_dim == codomain.event_dim)
                                return _domain.event_dim;
                            throw new ArgumentException("Please use either .domain.event_dim or .codomain.event_dim");
                        }
                    }

                    public virtual Transform inv {
                        get {
                            Transform result = null;
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

                    public int sign {
                        get {
                            return _sign();
                        }
                    }

                    protected abstract int _sign();
                    protected abstract Tensor _call(Tensor x);
                    protected internal abstract Tensor _inverse(Tensor y);

                    protected internal abstract Tensor log_abs_det_jacobian(Tensor x, Tensor y);

                    public Tensor forward(Tensor x) => this._call(x);

                    public virtual long[] forward_shape(long[] shape) => shape;

                    public virtual long[] inverse_shape(long[] shape) => shape;
                }


                internal class _InverseTransform : Transform
                {
                    private torch.distributions.transforms.Transform transform;

                    public _InverseTransform(torch.distributions.transforms.Transform transform)
                    {
                        this.transform = transform;
                    }

                    public override constraints.Constraint domain {
                        get {
                            return _inv.domain;
                        }
                    }

                    public override constraints.Constraint codomain {
                        get {
                            return _inv.codomain;
                        }
                    }

                    public override bool bijective {
                        get {
                            return _inv.bijective;
                        }
                    }

                    public override int event_dim => base.event_dim;

                    public override Transform inv => base.inv;

                    protected internal override Tensor log_abs_det_jacobian(Tensor x, Tensor y)
                    {
                        return -_inv.log_abs_det_jacobian(y, x);
                    }

                    protected override Tensor _call(Tensor x)
                    {
                        return _inv._inverse(x);
                    }

                    protected internal override Tensor _inverse(Tensor y)
                    {
                        throw new NotImplementedException();
                    }

                    protected override int _sign()
                    {
                        return _inv.sign;
                    }

                    public override long[] forward_shape(long[] shape)
                    {
                        return _inv.forward_shape(shape);
                    }

                    public override long[] inverse_shape(long[] shape)
                    {
                        return _inv.inverse_shape(shape);
                    }
                }
            }
        }
    }
}
