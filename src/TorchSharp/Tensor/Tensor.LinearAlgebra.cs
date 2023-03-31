// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    public static partial class torch
    {
        public partial class Tensor
        {
            // https://pytorch.org/docs/stable/generated/torch.tensordot
            /// <summary>
            /// Returns a contraction of a and b over multiple dimensions.
            /// tensordot implements a generalized matrix product.
            /// </summary>
            public Tensor tensordot(Tensor b, long[] dims1, long[] dims2)
            {
                IntPtr res;
                unsafe {
                    fixed (long* pdims1 = dims1, pdims2 = dims2) {
                        res = THSLinalg_tensordot(Handle, b.Handle,(IntPtr)pdims1, dims1.Length,(IntPtr)pdims2, dims2.Length);
                    }
                }
                if (res == IntPtr.Zero) {
                    CheckForErrors();
                }
                return new Tensor(res);
            }

            // https://pytorch.org/docs/stable/generated/torch.tensordot
            /// <summary>
            /// Returns a contraction of this tensor and <paramref name="b"/> over multiple dimensions.
            /// tensordot implements a generalized matrix product.
            /// </summary>
            /// <param name="b">Right tensor to contract</param>
            /// <param name="dims">dimensions to contract for this tensor and <paramref name="b"/> respectively</param>
            /// <returns>contraction</returns>
            public Tensor tensordot(Tensor b, (long, long)[] dims)
                => tensordot(b, dims.Select(t => t.Item1).ToArray(), dims.Select(t => t.Item2).ToArray());

            // https://pytorch.org/docs/stable/generated/torch.tensordot
            /// <summary>
            /// Returns a contraction of this tensor and <paramref name="b"/> over multiple dimensions.
            /// tensordot implements a generalized matrix product.
            /// </summary>
            /// <param name="b">Right tensor to contract</param>
            /// <param name="dims">number of dimensions to contract for this tensor and <paramref name="b"/></param>
            /// <returns>contraction</returns>
            public Tensor tensordot(Tensor b, long dims = 2)
            {
                if (dims < 0) throw new ArgumentOutOfRangeException(nameof(dims), dims, "must be >= 0");

                var list = new long[dims];
                for (long i = 0; i <= dims; i++) {
                    list[i] = i;
                }

                return tensordot(b, list, list);
            }

            /// <summary>
            /// Computes the Cholesky decomposition of a symmetric positive-definite matrix AA or for batches of symmetric positive-definite matrices.
            /// </summary>
            /// <param name="upper">If upper is true, the returned matrix U is upper-triangular. If upper is false, the returned matrix L is lower-triangular</param>
            /// <returns></returns>
            public Tensor cholesky(bool upper = false)
            {
                var res = THSTensor_cholesky(Handle, upper);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the inverse of a symmetric positive-definite matrix AA using its Cholesky factor uu : returns matrix inv
            /// </summary>
            /// <param name="upper"></param>
            /// <returns></returns>
            public Tensor cholesky_inverse(bool upper = false)
            {
                var res = THSTensor_cholesky_inverse(Handle, upper);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix u.
            /// </summary>
            /// <param name="input2"></param>
            /// <param name="upper"></param>
            /// <returns></returns>
            public Tensor cholesky_solve(Tensor input2, bool upper = false)
            {
                var res = THSTensor_cholesky_solve(Handle, input2.Handle, upper);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the cross product of two 3-dimensional vectors.
            /// </summary>
            /// <remarks>
            /// Supports input of float, double, cfloat and cdouble dtypes. Also supports batches of vectors,
            /// for which it computes the product along the dimension dim. In this case, the output has the
            /// same batch dimensions as the inputs broadcast to a common shape.
            /// </remarks>
            public Tensor cross(Scalar other, long dim)
            {
                var res = THSTensor_cross(Handle, other.Handle, dim);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the determinant of a square matrix.
            /// </summary>
            public Tensor det()
            {
                return linalg.det(this);
            }

            /// <summary>
            /// Calculates log determinant of a square matrix or batches of square matrices.
            /// </summary>
            /// <returns></returns>
            public Tensor logdet()
            {
                var shape = this.shape;
                var len = shape.Length;
                if (shape[len - 1] != shape[len - 2]) throw new ArgumentException("The input tensor is not square");

                var res = THSTensor_logdet(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }


            /// <summary>
            /// This is a low-level function for calling LAPACK’s geqrf directly.
            /// This function returns a namedtuple (a, tau) as defined in LAPACK documentation for geqrf.
            /// </summary>
            /// <remarks>
            /// Computes a QR decomposition of input. Both Q and R matrices are stored in the same output tensor a.
            /// The elements of R are stored on and above the diagonal. Elementary reflectors (or Householder vectors)
            /// implicitly defining matrix Q are stored below the diagonal. The results of this function can be used
            /// together with torch.linalg.householder_product() to obtain the Q matrix or with torch.ormqr(), which
            /// uses an implicit representation of the Q matrix, for an efficient matrix-matrix multiplication.
            /// </remarks>
            public (Tensor a, Tensor tau) geqrf()
            {
                var res = THSTensor_geqrf(Handle, out var tau);
                if (res == IntPtr.Zero || tau == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(res), new Tensor(tau));
            }

            /// <summary>
            /// Matrix product of two tensors.
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            /// <remarks>
            /// The behavior depends on the dimensionality of the tensors as follows:
            /// 1. If both tensors are 1-dimensional, the dot product (scalar) is returned
            /// 2. If both arguments are 2-dimensional, the matrix-matrix product is returned.
            /// 3. If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
            /// 4. If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
            /// 5. If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiply is returned.
            /// </remarks>
            public Tensor matmul(Tensor target)
            {
                var res = THSTensor_matmul(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Performs a matrix multiplication of the matrices input and mat2.
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor mm(Tensor target)
            {
                var res = THSTensor_mm(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Performs a matrix-vector product of the matrix input and the vector vec.
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor mv(Tensor target)
            {
                var res = THSTensor_mv(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the matrix exponential of a square matrix or of each square matrix in a batch.
            /// </summary>
            public Tensor matrix_exp()
            {
                var res = THSTensor_matrix_exp(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the n-th power of a square matrix for an integer n.
            /// </summary>
            /// <param name="n">The exponent</param>
            /// <returns></returns>
            /// <remarks>Input tensor must be of shape (*, m, m) where * is zero or more batch dimensions.</remarks>
            public Tensor matrix_power(int n)
            {
                var res = THSLinalg_matrix_power(Handle, n);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the dot product of two 1D tensors.
            /// </summary>
            /// <returns></returns>
            /// <remarks>
            /// The vdot(a, b) function handles complex numbers differently than dot(a, b).
            /// If the first argument is complex, the complex conjugate of the first argument is used for the calculation of the dot product.
            /// </remarks>
            public Tensor vdot(Tensor target)
            {
                if (shape.Length != 1 || target.shape.Length != 1 || shape[0] != target.shape[0]) throw new InvalidOperationException("vdot arguments must have the same shape.");
                var res = THSTensor_vdot(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the dot product of two 1D tensors.
            /// </summary>
            /// <returns></returns>
            public Tensor dot(Tensor target)
            {
                if (shape.Length != 1 || target.shape.Length != 1 || shape[0] != target.shape[0]) throw new InvalidOperationException("dot arguments must have the same shape.");
                var res = THSTensor_dot(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the pseudoinverse (Moore-Penrose inverse) of a matrix.
            /// </summary>
            /// <param name="rcond">The tolerance value to determine when is a singular value zero </param>
            /// <param name="hermitian">Indicates whether A is Hermitian if complex or symmetric if real. </param>
            /// <remarks>Input should be tensor of shape (*, m, n) where * is zero or more batch dimensions.</remarks>
            /// <returns></returns>
            public Tensor pinverse(double rcond = 1e-15, bool hermitian = false)
            {
                var res = THSLinalg_pinverse(Handle, rcond, hermitian);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix.
            /// </summary>
            /// <param name="tau">Tensor of shape (*, min(mn, k)) where * is zero or more batch dimensions.</param>
            /// <param name="other">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="left">Controls the order of multiplication.</param>
            /// <param name="transpose">Controls whether the matrix Q is conjugate transposed or not.</param>
            /// <returns></returns>
            public Tensor ormqr(Tensor tau, Tensor other, bool left = true, bool transpose = false)
            {
                var res = THSTensor_ormqr(Handle, tau.handle, other.Handle, left, transpose);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }
        }
    }
}