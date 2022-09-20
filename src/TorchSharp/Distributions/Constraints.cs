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
            public static partial class constraints
            {
                /// <summary>
                /// Abstract base class for constraints.
                /// 
                /// A constraint object represents a region over which a variable is valid, e.g. within which a variable can be optimized.
                /// </summary>
                /// <remarks>
                /// // It's not ideal to use a '_' first in a public .NET type name, but that's what Pytorch does for all contraint types.
                /// </remarks>
                public abstract class Constraint
                {
                    /// <summary>
                    /// Constructor
                    /// </summary>
                    /// <param name="is_discrete">Whether constrained space is discrete.</param>
                    /// <param name="event_dim">
                    /// Number of rightmost dimensions that together define an event.
                    /// The check() method will remove this many dimensions when computing validity.
                    /// </param>
                    protected Constraint(bool is_discrete, int event_dim)
                    {
                        this.is_discrete = is_discrete;
                        this.event_dim = event_dim;
                    }

                    /// <summary>
                    /// Constructor
                    /// </summary>
                    protected Constraint() : this(false, 0)
                    {
                    }

                    /// <summary>
                    /// Returns a byte tensor of sample_shape + batch_shape indicating  whether each event in value satisfies this constraint.
                    /// </summary>
                    /// <param name="value"></param>
                    /// <returns></returns>
                    public abstract Tensor check(Tensor value);

                    /// <summary>
                    /// Whether constrained space is discrete.
                    /// </summary>
                    public virtual bool is_discrete { get; protected set; }

                    /// <summary>
                    /// Number of rightmost dimensions that together define an event.
                    /// The check() method will remove this many dimensions when computing validity.
                    /// </summary>
                    public virtual int event_dim { get; protected set; }

                    public override string ToString()
                    {
                        return this.GetType().Name + "()";
                    }
                }

                /// <summary>
                /// Placeholder for variables whose support depends on other variables.
                /// These variables obey no simple coordinate-wise constraints.
                /// </summary>
                public class _Dependent : Constraint
                {
                    public _Dependent(bool is_discrete = false, int event_dim = 0) : base(is_discrete, event_dim) { }

                    public override Tensor check(Tensor value)
                    {
                        throw new ArgumentException("Cannot determine validity of dependent constraint");
                    }
                }

                /// <summary>
                /// Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
                /// dims in :meth:`check`, so that an event is valid only if all its
                /// independent entries are valid.
                /// </summary>
                public class _IndependentConstraint : Constraint
                {
                    public _IndependentConstraint(Constraint base_constraint, int reinterpreted_batch_ndims)
                    {

                    }

                    public override Tensor check(Tensor value)
                    {
                        throw new NotImplementedException();
                    }

                    public override string ToString()
                    {
                        return this.GetType().Name + "()";
                    }

                    public Constraint base_constraint { get; private set; }

                    /// <summary>
                    /// Whether constrained space is discrete.
                    /// </summary>
                    public override bool is_discrete { get => base_constraint.is_discrete; }

                    /// <summary>
                    /// Number of rightmost dimensions that together define an event.
                    /// The check() method will remove this many dimensions when computing validity.
                    /// </summary>
                    public override int event_dim { get => base_constraint.event_dim; }

                }

                /// <summary>
                /// Constrain to the two values {0, 1}.
                /// </summary>
                public class _Boolean : Constraint
                {
                    public _Boolean() : base(true, 0) { }

                    public override Tensor check(Tensor value) => (value == 0) | (value == 1);
                }

                /// <summary>
                /// Constrain to one-hot vectors.
                /// </summary>
                public class _OneHot : Constraint
                {
                    public _OneHot() : base(true, 1) { }

                    public override Tensor check(Tensor value)
                    {
                        var is_boolean = (value == 0) | (value == 1);
                        var is_normalized = value.sum(-1).eq(1);
                        return is_boolean.all(-1) & is_normalized;
                    }
                }

                /// <summary>
                /// Constrain to an integer interval [lower_bound, upper_bound].
                /// </summary>
                public class _IntegerInterval : Constraint
                {
                    public _IntegerInterval(long lower_bound, long upper_bound) : base(true, 0)
                    {
                        this.lower_bound = lower_bound;
                        this.upper_bound = upper_bound;
                    }

                    public long lower_bound { get; private set; }

                    public long upper_bound { get; private set; }

                    public override Tensor check(Tensor value)
                    {
                        return (value % 1 == 0) & (lower_bound <= value) & (value <= upper_bound);
                    }

                    public override string ToString()
                    {
                        return $"{GetType().Name}(lower_bound={lower_bound},upper_bound={upper_bound})";
                    }
                }

                /// <summary>
                /// Constrain to an integer interval [-inf, upper_bound].
                /// </summary>
                public class _IntegerLessThan : Constraint
                {
                    public _IntegerLessThan(long upper_bound) : base(true, 0)
                    {
                        this.upper_bound = upper_bound;
                    }

                    public long upper_bound { get; private set; }

                    public override Tensor check(Tensor value)
                    {
                        return (value % 1 == 0) & (value <= upper_bound);
                    }

                    public override string ToString()
                    {
                        return $"{GetType().Name}(upper_bound={upper_bound})";
                    }
                }

                /// <summary>
                /// Constrain to an integer interval [lower_bound, inf].
                /// </summary>
                public class _IntegerGreaterThan : Constraint
                {
                    public _IntegerGreaterThan(long lower_bound) : base(true, 0)
                    {
                        this.lower_bound = lower_bound;
                    }

                    public long lower_bound { get; private set; }

                    public override Tensor check(Tensor value)
                    {
                        return (value % 1 == 0) & (lower_bound <= value);
                    }

                    public override string ToString()
                    {
                        return $"{GetType().Name}(lower_bound={lower_bound})";
                    }
                }

                /// <summary>
                /// Trivially constrain to the extended real numbers [-inf, inf].
                /// </summary>
                public class _Real : Constraint
                {
                    public _Real() : base(false, 0) { }

                    public override Tensor check(Tensor value) => value.eq(value); // False only for NaN.
                }

                /// <summary>
                /// Constrain to an interval [lower_bound, upper_bound].
                /// </summary>
                public class _Interval : Constraint
                {
                    public _Interval(double lower_bound, double upper_bound) : base(true, 0)
                    {
                        this.lower_bound = lower_bound;
                        this.upper_bound = upper_bound;
                    }

                    public double lower_bound { get; private set; }

                    public double upper_bound { get; private set; }

                    public override Tensor check(Tensor value)
                    {
                        return (lower_bound <= value) & (value <= upper_bound);
                    }

                    public override string ToString()
                    {
                        return $"{GetType().Name}(lower_bound={lower_bound},upper_bound={upper_bound})";
                    }
                }

                /// <summary>
                /// Constrain to an interval [lower_bound, upper_bound).
                /// </summary>
                public class _HalfOpenInterval : Constraint
                {
                    public _HalfOpenInterval(double lower_bound, double upper_bound) : base(true, 0)
                    {
                        this.lower_bound = lower_bound;
                        this.upper_bound = upper_bound;
                    }

                    public double lower_bound { get; private set; }

                    public double upper_bound { get; private set; }

                    public override Tensor check(Tensor value)
                    {
                        return (lower_bound <= value) & (value < upper_bound);
                    }

                    public override string ToString()
                    {
                        return $"{GetType().Name}(lower_bound={lower_bound},upper_bound={upper_bound})";
                    }
                }

                /// <summary>
                /// Constrain to an interval (lower_bound, inf].
                /// </summary>
                public class _GreaterThan : Constraint
                {
                    public _GreaterThan(double lower_bound) : base(true, 0)
                    {
                        this.lower_bound = lower_bound;
                    }

                    public double lower_bound { get; private set; }

                    public override Tensor check(Tensor value)
                    {
                        return (lower_bound < value);
                    }

                    public override string ToString()
                    {
                        return $"{GetType().Name}(lower_bound={lower_bound})";
                    }
                }

                /// <summary>
                /// Constrain to an interval [lower_bound, inf].
                /// </summary>
                public class _GreaterThanEq : Constraint
                {
                    public _GreaterThanEq(double lower_bound) : base(true, 0)
                    {
                        this.lower_bound = lower_bound;
                    }

                    public double lower_bound { get; private set; }

                    public override Tensor check(Tensor value)
                    {
                        return (lower_bound <= value);
                    }

                    public override string ToString()
                    {
                        return $"{GetType().Name}(lower_bound={lower_bound})";
                    }
                }

                /// <summary>
                /// Constrain to an integer interval [-inf, upper_bound).
                /// </summary>
                public class _LessThan : Constraint
                {
                    public _LessThan(double upper_bound) : base(true, 0)
                    {
                        this.upper_bound = upper_bound;
                    }

                    public double upper_bound { get; private set; }

                    public override Tensor check(Tensor value)
                    {
                        return (value < upper_bound);
                    }

                    public override string ToString()
                    {
                        return $"{GetType().Name}(upper_bound={upper_bound})";
                    }
                }

                /// <summary>
                /// Constrain to the unit simplex in the innermost (rightmost) dimension.
                /// Specifically: `x >= 0` and `x.sum(-1) == 1`.
                /// </summary>
                public class _Simplex : Constraint
                {
                    public _Simplex() : base(false, 1) { }

                    public override Tensor check(Tensor value)
                    {
                        var is_positive = value >= 0;
                        return is_positive.all(dim: -1) & ((value.sum(-1) - 1).abs() < 1e-6);
                    }
                }

                /// <summary>
                /// Constrain to nonnegative integer values summing to at most an upper bound.
                /// </summary>
                public class _Multinomial : Constraint
                {
                    public _Multinomial(long upper_bound) : base(true, 1)
                    {
                        this.upper_bound = upper_bound;
                    }

                    public long upper_bound { get; private set; }

                    public override Tensor check(Tensor x)
                    {
                        return (x >= 0).all(dim: -1) & (x.sum(dim: -1) <= upper_bound);
                    }

                    public override string ToString()
                    {
                        return $"{GetType().Name}(upper_bound={upper_bound})";
                    }
                }

                /// <summary>
                /// Constrain to lower-triangular square matrices.
                /// </summary>
                public class _LowerTriangular : Constraint
                {
                    public _LowerTriangular() : base(false, 2) { }

                    public override Tensor check(Tensor value)
                    {
                        var tril = value.tril();
                        var newshape = new long[value.shape.Length - 1];
                        var i = 0;
                        for (; i < value.shape.Length - 2; i++) newshape[i] = value.shape[i];
                        newshape[i] = -1;
                        return (tril == value).view(newshape).min(-1).values;
                    }
                }

                /// <summary>
                /// Constrain to lower-triangular square matrices with positive diagonals.
                /// </summary>
                public class _LowerCholesky : Constraint
                {
                    public _LowerCholesky() : base(false, 2) { }

                    public override Tensor check(Tensor value)
                    {
                        var value_tril = value.tril();
                        var newshape = new long[value.shape.Length - 1];
                        var i = 0;
                        for (; i < value.shape.Length - 2; i++) newshape[i] = value.shape[i];
                        newshape[i] = -1;

                        var lower_triangular = (value_tril == value).view(newshape).min(-1).values;
                        var positive_diagonal = (value.diagonal(dim1: -2, dim2: -1) > 0).min(-1).values;
                        return lower_triangular & positive_diagonal;
                    }
                }

                /// <summary>
                /// Constrain to lower-triangular square matrices with positive diagonals and each
                /// row vector being of unit length.
                /// </summary>
                public class _CorrCholesky : Constraint
                {
                    public _CorrCholesky() : base(false, 2) { }

                    public override Tensor check(Tensor value)
                    {
                        var tol = torch.finfo(value.dtype).eps * value.size(-1) * 10;  // 10 is an adjustable fudge factor
                        var row_norm = torch.linalg.norm(value.detach(), dims: new[] { -1L });
                        var unit_row_norm = (row_norm - 1.0).abs().le(tol).all(dim: -1);
                        return lc.check(value) & unit_row_norm;
                    }

                    private _LowerCholesky lc = new _LowerCholesky();
                }

                /// <summary>
                /// Constrain to square matrices.
                /// </summary>
                public class _Square : Constraint
                {
                    public _Square() : base(false, 2) { }

                    public override Tensor check(Tensor value)
                    {
                        var newshape = new long[value.shape.Length - 2];
                        var i = 0;
                        for (; i < value.shape.Length - 2; i++) newshape[i] = value.shape[i];

                        return torch.full(
                            size: newshape,
                            value: (value.shape[value.shape.Length - 2] == value.shape[value.shape.Length - 1]),
                            dtype: torch.@bool,
                            device: value.device);
                    }
                }

                /// <summary>
                /// Constrain to symmetric square matrices.
                /// </summary>
                public class _Symmetric : _Square
                {

                    public override Tensor check(Tensor value)
                    {
                        var square_check = base.check(value);
                        if (!square_check.all().item<bool>())
                            return square_check;

                        return value.isclose(value.mT, atol: 1e-6).all(-2).all(-1);
                    }
                }

                /// <summary>
                /// Constrain to positive-semidefinite matrices.
                /// </summary>
                public class _PositiveSemiDefinite : _Symmetric
                {
                    public override Tensor check(Tensor value)
                    {
                        var sym_check = base.check(value);
                        if (!sym_check.all().item<bool>())
                            return sym_check;
                        return torch.linalg.eigvalsh(value).ge(0).all(-1);
                    }
                }

                /// <summary>
                /// Constrain to positive-definite matrices.
                /// </summary>
                public class _PositiveDefinite : _Symmetric
                {
                    public override Tensor check(Tensor value)
                    {
                        var sym_check = base.check(value);
                        if (!sym_check.all().item<bool>())
                            return sym_check;
                        return torch.linalg.cholesky_ex(value).info.eq(0);
                    }
                }

                /// <summary>
                /// Constraint functor that applies a sequence of constraints cseq at the submatrices at dimension dim,
                /// each of size lengths[dim], in a way compatible with torch.cat().
                /// </summary>
                public class _Cat : Constraint
                {
                    public _Cat(IList<Constraint> cseq, long dim = 0, IList<long> lengths = null)
                    {
                        this.cseq = cseq;
                        if (lengths == null) {
                            lengths = Enumerable.Repeat(1L, cseq.Count).ToList();
                        }
                        this.lengths = lengths;
                        this.dim = dim;
                    }

                    public override Tensor check(Tensor value)
                    {
                        var checks = new List<Tensor>();
                        long start = 0;
                        for (int i = 0; i < cseq.Count; i++) {
                            var length = lengths[i];
                            var constr = cseq[i];
                            var v = value.narrow(dim, start, length);
                            checks.Add(constr.check(v));
                            start = start + length;
                        }
                        return torch.cat(checks, dim);
                    }

                    public override bool is_discrete { get => cseq.Any(c => c.is_discrete); }

                    public override int event_dim { get => cseq.Select(c => c.event_dim).Max(); }

                    private IList<Constraint> cseq;
                    private IList<long> lengths;
                    private long dim;
                }
                /// <summary>
                /// Constraint functor that applies a sequence of constraints cseq at the submatrices at dimension dim,
                /// each of size lengths[dim], in a way compatible with torch.cat().
                /// </summary>
                public class _Stack : Constraint
                {
                    public _Stack(IList<Constraint> cseq, int dim = 0)
                    {
                        this.cseq = cseq;
                        this.dim = dim;
                    }

                    public override Tensor check(Tensor value)
                    {
                        var vs = Enumerable.Range(0, (int)value.size(dim)).Select(i => value.select(dim, i)).ToList();
                        return torch.stack(Enumerable.Range(0, vs.Count).Select(i => cseq[i].check(vs[i])));
                    }

                    public override bool is_discrete { get => cseq.Any(c => c.is_discrete); }

                    public override int event_dim {
                        get {
                            var dim = cseq.Select(c => c.event_dim).Max();

                            if (this.dim + dim < 0)
                                dim += 1;

                            return dim;
                        }
                    }

                    private IList<Constraint> cseq;
                    private int dim;
                }


                // Public interface

                public static _Dependent dependent = new _Dependent();

                public static _IndependentConstraint independent(Constraint base_constraint, int reinterpreted_batch_ndims) => new _IndependentConstraint(base_constraint, reinterpreted_batch_ndims);

                public static _Boolean boolean = new _Boolean();

                public static _OneHot one_hot = new _OneHot();

                public static _IntegerGreaterThan nonnegative_integer = new _IntegerGreaterThan(0);

                public static _IntegerGreaterThan positive_integer = new _IntegerGreaterThan(1);

                public static _IntegerInterval integer_interval(long lb, long ub) => new _IntegerInterval(lb, ub);

                public static _Real real = new _Real();

                public static _IndependentConstraint real_vector = independent(real, 1);

                public static _GreaterThan positive = new _GreaterThan(0.0);

                public static _GreaterThanEq nonnegative = new _GreaterThanEq(0.0);

                public static _GreaterThan greater_than(double lower_bound) => new _GreaterThan(lower_bound);

                public static _GreaterThanEq greater_than_eq(double lower_bound) => new _GreaterThanEq(lower_bound);

                public static _LessThan less_than(double upper_bound) => new _LessThan(upper_bound);

                public static _Multinomial multinomial(long upper_bound) => new _Multinomial(upper_bound);

                public static _Interval unit_interval = new _Interval(0.0, 1.0);

                public static _Interval interval(double lower_bound, double upper_bound) => new _Interval(lower_bound, upper_bound);

                public static _HalfOpenInterval half_open_interval(double lower_bound, double upper_bound) => new _HalfOpenInterval(lower_bound, upper_bound);

                public static _Simplex simplex = new _Simplex();

                public static _LowerTriangular lower_triangular = new _LowerTriangular();

                public static _LowerCholesky lower_cholesky = new _LowerCholesky();

                public static _CorrCholesky corr_cholesky = new _CorrCholesky();

                public static _Square square = new _Square();

                public static _Symmetric symmetric = new _Symmetric();

                public static _PositiveSemiDefinite positive_semidefinite = new _PositiveSemiDefinite();

                public static _PositiveDefinite positive_definite = new _PositiveDefinite();

                public static _Cat cat(IList<Constraint> cseq, long dim = 0, IList<long> lengths = null) => new _Cat(cseq, dim, lengths);

                public static _Stack stack(IList<Constraint> cseq, int dim = 0) => new _Stack(cseq, dim);
            }
        }
    }
}
