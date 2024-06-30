// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace TorchSharp
{
    using System.Numerics;
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// A multivariate normal (also called Gaussian) distribution parameterized by a mean vector and a covariance matrix.
        ///
        /// The multivariate normal distribution can be parameterized either in terms of a positive definite covariance matrix
        /// or a positive definite precision matrix or a lower-triangular matrix with positive-valued diagonal entries. This triangular matrix
        /// can be obtained via Cholesky decomposition of the covariance.
        /// </summary>
        public class MultivariateNormal : torch.distributions.Distribution
        {
            /// <summary>
            /// The mean of the distribution.
            /// </summary>
            public override Tensor mean => loc;

            /// <summary>
            /// The mode of the distribution.
            /// </summary>
            public override Tensor mode => loc;

            /// <summary>
            /// The variance of the distribution
            /// </summary>
            public override Tensor variance =>
                WrappedTensorDisposeScope(() => _unbroadcasted_scale_tril.pow(2).sum(-1).expand(batch_shape + event_shape));

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="loc"></param>
            /// <param name="covariance_matrix">Positive-definite covariance matrix</param>
            /// <param name="precision_matrix">Positive-definite precision matrix</param>
            /// <param name="scale_tril">The lower-triangular factor of covariance, with positive-valued diagonal</param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <remarks>
            /// Only one of `covariance_matrix` or `precision_matrix` or `scale_tril` may be specified.
            ///
            /// Using `scale_tril` will be more efficient: all computations internally are based on `scale_tril`.
            /// If `covariance_matrix` or `precision_matrix` is passed instead, it is only used to compute
            /// the corresponding lower triangular matrices using a Cholesky decomposition.
            /// </remarks>
            public MultivariateNormal(Tensor loc, Tensor covariance_matrix = null, Tensor precision_matrix = null, Tensor scale_tril = null, torch.Generator generator = null) : base(generator)
            {
                var argCount = 0;
                argCount += (covariance_matrix is null ? 0 : 1);
                argCount += (precision_matrix is null ? 0 : 1);
                argCount += (scale_tril is null ? 0 : 1);

                if (argCount != 1)
                    throw new ArgumentException("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.");

                using var _ = NewDisposeScope();

                if (scale_tril is not null) {
                    if (scale_tril.dim() < 2) {
                        throw new ArgumentException("scale_tril matrix must be at least two-dimensional, with optional leading batch dimensions");
                    }
                    batch_shape = torch.broadcast_shapes(TakeAllBut(scale_tril.shape, 2), TakeAllBut(loc.shape, 1));
                    this.scale_tril = scale_tril.expand(batch_shape + (-1, -1)).DetachFromDisposeScope();
                    _unbroadcasted_scale_tril = scale_tril;
                } else if (covariance_matrix is not null) {
                    if (covariance_matrix.dim() < 2) {
                        throw new ArgumentException("covariance_matrix matrix must be at least two-dimensional, with optional leading batch dimensions");
                    }
                    batch_shape = torch.broadcast_shapes(TakeAllBut(covariance_matrix.shape, 2).ToArray(), TakeAllBut(loc.shape, 1).ToArray());
                    this.covariance_matrix = covariance_matrix.expand(batch_shape + (-1, -1)).DetachFromDisposeScope();
                    _unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix).DetachFromDisposeScope();
                } else {
                    if (precision_matrix.dim() < 2) {
                        throw new ArgumentException("precision_matrix matrix must be at least two-dimensional, with optional leading batch dimensions");
                    }
                    batch_shape = torch.broadcast_shapes(TakeAllBut(precision_matrix.shape, 2).ToArray(), TakeAllBut(loc.shape, 1).ToArray());
                    this.precision_matrix = precision_matrix.expand(batch_shape + (-1, -1)).DetachFromDisposeScope();
                    _unbroadcasted_scale_tril = PrecisionToScaleTril(precision_matrix).DetachFromDisposeScope();
                }

                this.loc = loc.expand(batch_shape + -1).DetachFromDisposeScope();

                this.event_shape = loc.shape[loc.shape.Length-1];
            }

            private MultivariateNormal(torch.Generator generator = null) : base(generator) { }

            private Tensor loc;
            private Tensor scale_tril;
            private Tensor precision_matrix;
            private Tensor covariance_matrix;
            private Tensor _unbroadcasted_scale_tril;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    loc?.Dispose();
                    scale_tril?.Dispose();
                    precision_matrix?.Dispose();
                    covariance_matrix?.Dispose();
                    _unbroadcasted_scale_tril?.Dispose();
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
                using var _ = NewDisposeScope();

                var shape = ExtendedShape(sample_shape);
                var eps = torch.empty(shape, dtype: loc.dtype, device: loc.device).normal_(generator:generator);
                return (loc + BatchMV(_unbroadcasted_scale_tril, eps)).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Returns the log of the probability density/mass function evaluated at `value`.
            /// </summary>
            /// <param name="value"></param>
            public override Tensor log_prob(Tensor value)
            {
                using var _ = NewDisposeScope();

                var diff = value - loc;
                var M = BatchMahalanobis(_unbroadcasted_scale_tril, diff);
                var half_log_det = _unbroadcasted_scale_tril.diagonal(dim1: -2, dim2: -1).log().sum(-1);
                return (-0.5 * (event_shape[0] * Math.Log(2 * Math.PI) + M) - half_log_det).MoveToOuterDisposeScope();
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
                if (instance != null && !(instance is MultivariateNormal))
                    throw new ArgumentException("expand(): 'instance' must be a MultivariateNormal distribution");

                var newDistribution = ((instance == null) ? new MultivariateNormal(generator) : instance) as MultivariateNormal;

                var loc_shape = batch_shape + event_shape;
                var cov_shape = batch_shape + event_shape + event_shape;

                newDistribution.loc = loc.expand(loc_shape);
                newDistribution._unbroadcasted_scale_tril = _unbroadcasted_scale_tril;
                newDistribution.scale_tril = scale_tril?.expand(cov_shape);
                newDistribution.covariance_matrix = covariance_matrix?.expand(cov_shape).DetachFromDisposeScope();
                newDistribution.precision_matrix = precision_matrix?.expand(cov_shape).DetachFromDisposeScope();

                newDistribution.batch_shape = batch_shape;
                newDistribution.event_shape = event_shape;

                return newDistribution;
            }

            /// <summary>
            /// Returns entropy of distribution, batched over batch_shape.
            /// </summary>
            /// <returns></returns>
            public override Tensor entropy()
            {
                using var _ = NewDisposeScope();

                var half_log_det = _unbroadcasted_scale_tril.diagonal(dim1: -2, dim2: -1).log().sum(-1);
                var H = 0.5 * event_shape[0] * (1 + Math.Log(2 * Math.PI)) + half_log_det;

                return ((batch_shape.Length == 0) ? H : H.expand(batch_shape)).MoveToOuterDisposeScope();
            }

            private Tensor BatchMV(Tensor bmat, Tensor bvec)
            {
                using var _ = NewDisposeScope();
                return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1).MoveToOuterDisposeScope();
            }

            private IEnumerable<long> Range(long start, long end, long step = 1)
            {
                var result = new List<long>();
                for (long i = start; i < end; i += step) {
                    result.Add(i);
                }
                return result;
            }

            private long[] TakeAllBut(long[] input, int count)
            {
                var result = new long[input.Length - count];
                for (int i = 0; i < result.Length; i++)
                    result[i] = input[i];
                return result;
            }

            private Tensor BatchMahalanobis(Tensor bL, Tensor bx)
            {
                using var _ = NewDisposeScope();

                var n = bx.size(-1);
                var bx_batch_shape = TakeAllBut(bx.shape, 1);

                var bx_batch_dims = bx_batch_shape.Length;
                var bL_batch_dims = bL.dim() - 2;
                int outer_batch_dims = bx_batch_dims - (int)bL_batch_dims;
                var old_batch_dims = outer_batch_dims + bL_batch_dims;
                var new_batch_dims = outer_batch_dims + 2 * bL_batch_dims;

                var bx_new_shape = bx.shape.Take(outer_batch_dims).ToList();

                for (int i = 0; i < bL.ndim - 2; i++) {
                    var sL = bL.shape[i];
                    var sx = bx.shape[outer_batch_dims + i];
                    bx_new_shape.Add(sx / sL);
                    bx_new_shape.Add(sL);
                }
                bx_new_shape.Add(n);
                bx = bx.reshape(bx_new_shape.ToArray());

                List<long> permute_dims = new List<long>();
                permute_dims.AddRange(Range(0, outer_batch_dims));
                permute_dims.AddRange(Range(outer_batch_dims, new_batch_dims, 2));
                permute_dims.AddRange(Range(outer_batch_dims + 1, new_batch_dims, 2));
                permute_dims.Add(new_batch_dims);

                bx = bx.permute(permute_dims.ToArray());

                var flat_L = bL.reshape(-1, n, n);
                var flat_x = bx.reshape(-1, flat_L.size(0), n);
                var flat_x_swap = flat_x.permute(1, 2, 0);

                var M_swap = torch.linalg.solve_triangular(flat_L, flat_x_swap, upper: false).pow(2).sum(-2);
                var M = M_swap.t();

                var permuted_M = M.reshape(TakeAllBut(bx.shape, 1));
                var permuted_inv_dims = Range(0, outer_batch_dims).ToList();
                for (int i = 0; i < bL_batch_dims; i++) {
                    permuted_inv_dims.Add(outer_batch_dims + i);
                    permuted_inv_dims.Add(old_batch_dims + i);
                }

                var reshaped_M = permuted_M.permute(permuted_inv_dims);

                return reshaped_M.reshape(bx_batch_shape).MoveToOuterDisposeScope();
            }

            private Tensor PrecisionToScaleTril(Tensor P)
            {
                using var _ = NewDisposeScope();
                var Lf = torch.linalg.cholesky(torch.flip(P, -2, -1));
                var L_inv = torch.transpose(torch.flip(Lf, -2, -1), -2, -1);
                var Id = torch.eye(P.shape[P.shape.Length - 1], dtype: P.dtype, device: P.device);
                var L = torch.linalg.solve_triangular(L_inv, Id, upper: false);
                return L.MoveToOuterDisposeScope();
            }
        }

    }
    public static partial class torch
    {
        public static partial class distributions
        {
            /// <summary>
            /// Creates a MultivariateNormal distribution parameterized by `probs` or `logits` (but not both).
            /// `total_count` must be broadcastable with `probs`/`logits`.
            /// </summary>
            /// <param name="loc"></param>
            /// <param name="covariance_matrix"></param>
            /// <param name="precision_matrix"></param>
            /// <param name="scale_tril"></param>
            /// <param name="generator">An optional random number generator object.</param>
            /// <returns></returns>
            public static MultivariateNormal MultivariateNormal(Tensor loc, Tensor covariance_matrix = null, Tensor precision_matrix = null, Tensor scale_tril = null, torch.Generator generator = null)
            {
                return new MultivariateNormal(loc, covariance_matrix, precision_matrix, scale_tril, generator);
            }
        }
    }
}
