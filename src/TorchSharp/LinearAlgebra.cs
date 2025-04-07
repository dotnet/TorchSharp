// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        public static class linalg
        {
            /// <summary>
            /// Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <returns></returns>
            public static Tensor cholesky(Tensor input)
            {
                var res = THSLinalg_cholesky(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.
            /// This function skips the(slow) error checking and error message construction of torch.linalg.cholesky(),
            /// instead directly returning the LAPACK error codes as part of a named tuple(L, info).
            /// This makes this function a faster way to check if a matrix is positive-definite, and it provides an opportunity to handle
            /// decomposition errors more gracefully or performantly than torch.linalg.cholesky() does.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <param name="check_errors">Controls whether to check the content of infos.</param>
            /// <returns></returns>
            public static (Tensor L, Tensor info) cholesky_ex(Tensor input, bool check_errors = false)
            {
                var res = THSLinalg_cholesky_ex(input.Handle, check_errors, out var pInfo);
                if (res == IntPtr.Zero || pInfo == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(res), new Tensor(pInfo));
            }

            public static Tensor cond(Tensor input, int p)
            {
                var res = THSLinalg_cond_int(input.Handle, p);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the condition number of a matrix with respect to a matrix norm.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <param name="p">The type of the matrix norm to use in the computations</param>
            /// <returns></returns>
            public static Tensor cond(Tensor input, double p)
            {
                var res = THSLinalg_cond_float(input.Handle, p);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the condition number of a matrix with respect to a matrix norm.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <param name="p">The type of the matrix norm to use in the computations</param>
            public static Tensor cond(Tensor input, string p)
            {
                var res = THSLinalg_cond_str(input.Handle, p);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the condition number of a matrix with respect to a matrix norm.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            public static Tensor cond(Tensor input)
            {
                var res = THSLinalg_cond_none(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the cross product of vectors in dimension dim of input and other.
            /// input and other must have the same size, and the size of their dim dimension should be 3.
            /// </summary>
            public static Tensor cross(Tensor input, Tensor other, long dim = -1)
            {
                var res = THSLinalg_cross(input.Handle, other.Handle, dim);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the determinant of a square matrix.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            public static Tensor det(Tensor input)
            {
                var res = THSLinalg_det(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.
            /// For complex A, it returns the angle and the natural logarithm of the modulus of the determinant, that is, a logarithmic polar decomposition of the determinant.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <returns></returns>
            public static (Tensor, Tensor) slogdet(Tensor input)
            {
                var res = THSLinalg_slogdet(input.Handle, out var logabsdet);
                if (res == IntPtr.Zero || logabsdet == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(res), new Tensor(logabsdet));
            }

            /// <summary>
            /// Returns a partial view of input with the its diagonal elements with respect to dim1 and dim2 appended as a dimension at the end of the shape.
            /// The argument offset controls which diagonal to consider:
            ///
            ///     If offset == 0, it is the main diagonal.
            ///     If offset &gt; 0, it is above the main diagonal.
            ///     If offset &lt; 0, it is below the main diagonal.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="offset">Which diagonal to consider. Default: 0 (main diagonal).</param>
            /// <param name="dim1">First dimension with respect to which to take diagonal. Default: -2.</param>
            /// <param name="dim2">Second dimension with respect to which to take diagonal. Default: -1.</param>
            /// <remarks>
            /// Applying torch.diag_embed() to the output of this function with the same arguments yields a diagonal matrix with the diagonal entries of the input.
            /// However, torch.diag_embed() has different default dimensions, so those need to be explicitly specified.
            /// </remarks>
            public static Tensor diagonal(Tensor input, int offset = 0, int dim1 = -2, int dim2 = -1) => input.diagonal(offset, dim1, dim2);

            /// <summary>
            /// Computes the eigenvalue decomposition of a square matrix if it exists.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <returns></returns>
            public static (Tensor, Tensor) eig(Tensor input)
            {
                var res = THSLinalg_eig(input.Handle, out var vectors);
                if (res == IntPtr.Zero || vectors == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(res), new Tensor(vectors));
            }

            /// <summary>
            /// Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <param name="UPLO">Controls whether to use the upper or lower triangular part of A in the computations. </param>
            /// <returns></returns>
            public static (Tensor, Tensor) eigh(Tensor input, char UPLO = 'L')
            {
                var res = THSLinalg_eigh(input.Handle, (byte)UPLO, out var vectors);
                if (res == IntPtr.Zero || vectors == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(res), new Tensor(vectors));
            }

            /// <summary>
            /// Computes the eigenvalues of a square matrix.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <returns></returns>
            public static Tensor eigvals(Tensor input)
            {
                var res = THSLinalg_eigvals(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the eigenvalues of a complex Hermitian or real symmetric matrix.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <param name="UPLO">Controls whether to use the upper or lower triangular part of A in the computations. </param>
            /// <returns></returns>
            public static Tensor eigvalsh(Tensor input, char UPLO = 'L')
            {
                var res = THSLinalg_eigvalsh(input.Handle, (byte)UPLO);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the first n columns of a product of Householder matrices.
            /// </summary>
            /// <param name="A">tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="tau">tensor of shape (*, k) where * is zero or more batch dimensions.</param>
            public static Tensor householder_product(Tensor A, Tensor tau)
            {
                var res = THSLinalg_householder_product(A.Handle, tau.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the inverse of a square matrix if it exists.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <returns></returns>
            /// <remarks>Throws a RuntimeError if the matrix is not invertible.</remarks>
            public static Tensor inv(Tensor input)
            {
                var res = THSLinalg_inv(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the inverse of a square matrix if it is invertible.
            /// Returns a named tuple(inverse, info). inverse contains the result of inverting A and info stores the LAPACK error codes.
            /// If A is not an invertible matrix, or if it’s a batch of matrices and one or more of them is not an invertible matrix,
            /// then info stores a positive integer for the corresponding matrix.The positive integer indicates the diagonal element of
            /// the LU decomposition of the input matrix that is exactly zero. info filled with zeros indicates that the inversion was successful.
            /// If check_errors = True and info contains positive integers, then a RuntimeError is thrown.
            /// </summary>
            /// <param name="input">The input tensor.</param>
            /// <param name="check_errors">Controls whether to check the content of info. controls whether to check the content of info. </param>
            /// <returns></returns>
            /// <remarks>Throws a RuntimeError if the matrix is not invertible.</remarks>
            public static (Tensor L, Tensor info) inv_ex(Tensor input, bool check_errors = false)
            {
                var res = THSLinalg_cholesky_ex(input.Handle, check_errors, out var pInfo);
                if (res == IntPtr.Zero || pInfo == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(res), new Tensor(pInfo));
            }

            /// <summary>
            /// Computes a solution to the least squares problem of a system of linear equations.
            /// </summary>
            /// <param name="input">lhs tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="other">rhs tensor of shape (*, m, k) where * is zero or more batch dimensions.</param>
            /// <returns></returns>
            public static (Tensor Solution, Tensor Residuals, Tensor Rank, Tensor SingularValues) lstsq(Tensor input, Tensor other)
            {
                var solution = THSLinalg_lstsq_none(input.Handle, other.Handle, out var residuals, out var rank, out var singularValues);
                if (solution == IntPtr.Zero || residuals == IntPtr.Zero || rank == IntPtr.Zero || singularValues == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(solution), new Tensor(residuals), new Tensor(rank), new Tensor(singularValues));
            }

            /// <summary>
            /// Computes the LU decomposition with partial pivoting of a matrix.
            /// </summary>
            /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="pivot">Controls whether to compute the LU decomposition with partial pivoting or no pivoting</param>
            /// <returns></returns>
            public static (Tensor P, Tensor L, Tensor U) lu(Tensor input, bool pivot = true)
            {
                var solution = THSLinalg_lu(input.Handle, pivot, out var pL, out var pU);
                if (solution == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(solution), new Tensor(pL), new Tensor(pU));
            }

            /// <summary>
            /// Computes a compact representation of the LU factorization with partial pivoting of a matrix.
            /// </summary>
            /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="pivot">Controls whether to compute the LU decomposition with partial pivoting or no pivoting</param>
            /// <returns></returns>
            public static (Tensor LU, Tensor? Pivots) lu_factor(Tensor input, bool pivot = true)
            {
                var solution = THSLinalg_lu_factor(input.Handle, pivot, out var pivots);
                if (solution == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(solution), pivots == IntPtr.Zero ? null : new Tensor(pivots));
            }

            /// <summary>
            /// Computes a compact representation of the LU factorization with partial pivoting of a matrix.
            /// </summary>
            /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="hermitian">Controls whether to consider the input to be Hermitian or symmetric. For real-valued matrices, this switch has no effect.</param>
            /// <returns></returns>
            public static (Tensor LU, Tensor? Pivots) ldl_factor(Tensor input, bool hermitian = true)
            {
                var solution = THSLinalg_ldl_factor(input.Handle, hermitian, out var pivots);
                if (solution == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(solution), pivots == IntPtr.Zero ? null : new Tensor(pivots));
            }

            /// <summary>
            /// Computes a compact representation of the LU factorization with partial pivoting of a matrix.
            /// </summary>
            /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="hermitian">Controls whether to consider the input to be Hermitian or symmetric. For real-valued matrices, this switch has no effect.</param>
            /// <param name="check_errors">Controls whether to check the content of info and raise an error if it is non-zero.</param>
            /// <returns></returns>
            public static (Tensor LU, Tensor? Pivots, Tensor? Info) ldl_factor_ex(Tensor input, bool hermitian = true, bool check_errors = false)
            {
                var solution = THSLinalg_ldl_factor_ex(input.Handle, hermitian, check_errors, out var pivots, out var info);
                if (solution == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(solution), pivots == IntPtr.Zero ? null : new Tensor(pivots), info == IntPtr.Zero ? null : new Tensor(info));
            }

            /// <summary>
            /// Computes the solution of a system of linear equations using the LDL factorization.
            /// </summary>
            /// <param name="LD">the n times n matrix or the batch of such matrices of size (*, n, n) where * is one or more batch dimensions</param>
            /// <param name="pivots">the pivots corresponding to the LDL factorization of LD</param>
            /// <param name="B">Right-hand side tensor of shape (*, n, k)</param>
            /// <param name="hermitian">Whether to consider the decomposed matrix to be Hermitian or symmetric. For real-valued matrices, this switch has no effect</param>
            /// <returns></returns>
            public static Tensor ldl_solve(Tensor LD, Tensor pivots, Tensor B, bool hermitian = false)
            {
                var res = THSLinalg_ldl_solve(LD.Handle, pivots.Handle, B.Handle, hermitian);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes a solution to the least squares problem of a system of linear equations.
            /// </summary>
            /// <param name="input">lhs tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="other">rhs tensor of shape (*, m, k) where * is zero or more batch dimensions.</param>
            /// <param name="rcond">Used to determine the effective rank of A. If rcond= None, rcond is set to the machine precision of the dtype of A times max(m, n).</param>
            public static (Tensor Solution, Tensor Residuals, Tensor Rank, Tensor SingularValues) lstsq(Tensor input, Tensor other, double rcond)
            {
                var solution = THSLinalg_lstsq_rcond(input.Handle, other.Handle, rcond, out var residuals, out var rank, out var singularValues);
                if (solution == IntPtr.Zero || residuals == IntPtr.Zero || rank == IntPtr.Zero || singularValues == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(solution), new Tensor(residuals), new Tensor(rank), new Tensor(singularValues));
            }

            /// <summary>
            /// Computes the matrix exponential of a square matrix or of each square matrix in a batch.
            /// </summary>
            public static Tensor matrix_exp(Tensor input) => input.matrix_exp();

            /// <summary>
            /// Computes a matrix norm.
            /// </summary>
            /// <param name="input">tensor with two or more dimensions.
            /// By default its shape is interpreted as (*, m, n) where * is zero or more batch dimensions, but this behavior can be controlled using dims.</param>
            /// <param name="ord">Order of norm. Default: "fro"</param>
            /// <param name="dims">Dimensions over which to compute the norm.</param>
            /// <param name="keepdim">If set to true, the reduced dimensions are retained in the result as dimensions with size one. </param>
            /// <returns></returns>
            public static Tensor matrix_norm(Tensor input, string ord = "fro", long[]? dims = null, bool keepdim = false)
            {
                if (dims == null) dims = new long[] { -2, -1 };
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = THSLinalg_matrix_norm_fronuc(input.Handle, ord == "fro" ? (byte)0 : (byte)1, (IntPtr)pdims, dims.Length, keepdim);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes a matrix norm.
            /// </summary>
            /// <param name="input">tensor with two or more dimensions.
            /// By default its shape is interpreted as (*, m, n) where * is zero or more batch dimensions, but this behavior can be controlled using dims.</param>
            /// <param name="ord">Order of norm.</param>
            /// <param name="dims">Dimensions over which to compute the norm.</param>
            /// <param name="keepdim">If set to true, the reduced dimensions are retained in the result as dimensions with size one. </param>
            /// <returns></returns>
            public static Tensor matrix_norm(Tensor input, double ord, long[]? dims = null, bool keepdim = false)
            {
                if (dims == null) dims = new long[] { -2, -1 };
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = THSLinalg_matrix_norm(input.Handle, ord.ToScalar().Handle, (IntPtr)pdims, dims.Length, keepdim);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the numerical rank of a matrix.
            /// The matrix rank is computed as the number of singular values(or eigenvalues in absolute value when hermitian = True) that are greater than the specified tol threshold.
            /// </summary>
            /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="atol">The absolute tolerance value.</param>
            /// <param name="rtol">The relative tolerance value.</param>
            /// <param name="hermitian">Indicates whether A is Hermitian if complex or symmetric if real</param>
            /// <returns></returns>
            public static Tensor matrix_rank(Tensor input, double? atol = null, double? rtol = null, bool hermitian = false)
            {
                unsafe {
                    var res = THSLinalg_matrix_rank(input.Handle, atol ?? double.NegativeInfinity, atol.HasValue, rtol ?? double.NegativeInfinity, rtol.HasValue, hermitian);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Computes the numerical rank of a matrix.
            /// The matrix rank is computed as the number of singular values(or eigenvalues in absolute value when hermitian = True) that are greater than the specified tol threshold.
            /// </summary>
            /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="atol">The absolute tolerance value.</param>
            /// <param name="rtol">The relative tolerance value.</param>
            /// <param name="hermitian">Indicates whether A is Hermitian if complex or symmetric if real</param>
            /// <returns></returns>
            public static Tensor matrix_rank(Tensor input, Tensor atol, Tensor? rtol = null, bool hermitian = false)
            {
                unsafe {
                    var res = THSLinalg_matrix_rank_tensor(input.Handle, atol is null ? IntPtr.Zero : atol.Handle, rtol is null ? IntPtr.Zero : rtol.Handle, hermitian);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Efficiently multiplies two or more matrices by reordering the multiplications so that the fewest arithmetic operations are performed.
            /// </summary>
            /// <param name="tensors">Two or more tensors to multiply. The first and last tensors may be 1D or 2D. Every other tensor must be 2D.</param>
            /// <returns></returns>
            public static Tensor multi_dot(IList<Tensor> tensors)
            {
                if (tensors.Count == 0) {
                    throw new ArgumentException(nameof(tensors));
                }
                if (tensors.Count == 1) {
                    return tensors[0].alias();
                }

                using (var parray = new PinnedArray<IntPtr>()) {
                    IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());
                    var res = THSLinalg_multi_dot(tensorsRef, parray.Array.Length);
                    if (res == IntPtr.Zero)
                        torch.CheckForErrors();
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Computes a vector or matrix norm.
            /// If A is complex valued, it computes the norm of A.abs()
            /// </summary>
            /// <param name="input">Tensor of shape (*, n) or (*, m, n) where * is zero or more batch dimensions</param>
            /// <param name="ord">Order of norm. </param>
            /// <param name="dims">Dimensions over which to compute the vector or matrix norm.</param>
            /// <param name="keepdim">If set to true, the reduced dimensions are retained in the result as dimensions with size one.</param>
            /// <returns></returns>
            public static Tensor norm(Tensor input, string ord, long[]? dims = null, bool keepdim = false)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = THSLinalg_norm_str(input.Handle, ord, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes a vector or matrix norm.
            /// If A is complex valued, it computes the norm of A.abs()
            /// </summary>
            /// <param name="input">Tensor of shape (*, n) or (*, m, n) where * is zero or more batch dimensions</param>
            /// <param name="ord">Order of norm. </param>
            /// <param name="dims">Dimensions over which to compute the vector or matrix norm.</param>
            /// <param name="keepdim">If set to true, the reduced dimensions are retained in the result as dimensions with size one.</param>
            public static Tensor norm(Tensor input, double ord, long[]? dims = null, bool keepdim = false)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = THSLinalg_norm_float(input.Handle, ord, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes a vector or matrix norm.
            /// If A is complex valued, it computes the norm of A.abs()
            /// </summary>
            /// <param name="input">Tensor of shape (*, n) or (*, m, n) where * is zero or more batch dimensions</param>
            /// <param name="ord">Order of norm. </param>
            /// <param name="dims">Dimensions over which to compute the vector or matrix norm.</param>
            /// <param name="keepdim">If set to true, the reduced dimensions are retained in the result as dimensions with size one.</param>
            public static Tensor norm(Tensor input, int ord, long[]? dims = null, bool keepdim = false)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = THSLinalg_norm_int(input.Handle, ord, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes a vector or matrix norm.
            /// If A is complex valued, it computes the norm of A.abs()
            /// </summary>
            /// <param name="input">Tensor of shape (*, n) or (*, m, n) where * is zero or more batch dimensions</param>
            /// <param name="dims">Dimensions over which to compute the vector or matrix norm.</param>
            /// <param name="keepdim">If set to true, the reduced dimensions are retained in the result as dimensions with size one.</param>
            public static Tensor norm(Tensor input, long[]? dims = null, bool keepdim = false)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = THSLinalg_norm_opt(input.Handle, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes the numerical rank of a matrix.
            /// The matrix rank is computed as the number of singular values(or eigenvalues in absolute value when hermitian = True) that are greater than the specified tol threshold.
            /// </summary>
            /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="atol">The absolute tolerance value.</param>
            /// <param name="rtol">The relative tolerance value.</param>
            /// <param name="hermitian">Indicates whether A is Hermitian if complex or symmetric if real</param>
            /// <returns></returns>
            public static Tensor pinv(Tensor input, double? atol = null, double? rtol = null, bool hermitian = false)
            {
                unsafe {
                    var res = THSLinalg_pinv(input.Handle, atol ?? double.NegativeInfinity, atol.HasValue, rtol ?? double.NegativeInfinity, rtol.HasValue, hermitian);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            /// <summary>
            /// Computes the numerical rank of a matrix.
            /// The matrix rank is computed as the number of singular values(or eigenvalues in absolute value when hermitian = True) that are greater than the specified tol threshold.
            /// </summary>
            /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="atol">The absolute tolerance value.</param>
            /// <param name="rtol">The relative tolerance value.</param>
            /// <param name="hermitian">Indicates whether A is Hermitian if complex or symmetric if real</param>
            /// <returns></returns>
            public static Tensor pinv(Tensor input, Tensor atol, Tensor? rtol = null, bool hermitian = false)
            {
                unsafe {
                    var res = THSLinalg_pinv_tensor(input.Handle, atol is null ? IntPtr.Zero : atol.Handle, rtol is null ? IntPtr.Zero : rtol.Handle, hermitian);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }

            public enum QRMode
            {
                Reduced = 0,
                Complete = 1,
                R = 2
            }

            /// <summary>
            /// Computes the QR decomposition of a matrix.
            /// </summary>
            /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="mode">Controls the shape of the returned tensors. One of ‘Reduced’, ‘Complete’, ‘R’.</param>
            /// <returns></returns>
            public static (Tensor Q, Tensor R) qr(Tensor input, QRMode mode = QRMode.Reduced)
            {
                var Q = THSLinalg_qr(input.Handle, (byte)mode, out var R);
                if (Q == IntPtr.Zero || R == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(Q), new Tensor(R));
            }

            /// <summary>
            /// Computes the solution of a square system of linear equations with a unique solution.
            /// </summary>
            /// <param name="A">Tensor of shape (*, n, n) where * is zero or more batch dimensions.</param>
            /// <param name="B">Right-hand side tensor of shape (*, n) or (*, n, k) or (n,) or (n, k)</param>
            /// <param name="left">whether to solve the system AX = B or XA = B.</param>
            /// <returns></returns>
            public static Tensor solve(Tensor A, Tensor B, bool left = true)
            {
                var res = THSLinalg_solve(A.Handle, B.Handle, left);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the solution of a square system of linear equations with a unique solution.
            /// </summary>
            /// <param name="A">Ttensor of shape (*, n, n) where * is zero or more batch dimensions.</param>
            /// <param name="B">Right-hand side tensor of shape (*, n) or (*, n, k) or (n,) or (n, k)</param>
            /// <param name="left">whether to solve the system AX = B or XA = B.</param>
            /// <param name="check_errors">controls whether to check the content of infos and raise an error if it is non-zero</param>
            /// <returns></returns>
            public static (Tensor result, Tensor info) solve_ex(Tensor A, Tensor B, bool left = true, bool check_errors = false)
            {
                var res = THSLinalg_solve_ex(A.Handle, B.Handle, left, check_errors, out var infos);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(res), new Tensor(infos));
            }

            /// <summary>
            /// Computes the solution of a square system of linear equations with a unique solution.
            /// </summary>
            /// <param name="A">Tensor of shape (*, n, n) where * is zero or more batch dimensions.</param>
            /// <param name="B">Right-hand side tensor of shape (*, n) or (*, n, k) or (n,) or (n, k)</param>
            /// <param name="upper">Whether A is an upper or lower triangular matrix</param>
            /// <param name="left">Whether to solve the system AX = B or XA = B.</param>
            /// <param name="unitriangular">If true, the diagonal elements of A are assumed to be all equal to 1.</param>
            /// <param name="out">Output tensor. B may be passed as out and the result is computed in-place on B.</param>
            /// <returns></returns>
            public static Tensor solve_triangular(Tensor A, Tensor B, bool upper, bool left = true, bool unitriangular = false, Tensor? @out = null)
            {
                var res = (@out is null)
                    ? THSLinalg_solve_triangular(A.Handle, B.Handle, upper, left, unitriangular)
                    : THSLinalg_solve_triangular_out(A.Handle, B.Handle, upper, left, unitriangular, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the singular value decomposition (SVD) of a matrix.
            /// </summary>
            /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
            /// <param name="fullMatrices">Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned tensors U and Vh.</param>
            /// <returns></returns>
            public static (Tensor U, Tensor S, Tensor Vh) svd(Tensor input, bool fullMatrices = true)
            {
                var U = THSLinalg_svd(input.Handle, fullMatrices, out var S, out var Vh);
                if (U == IntPtr.Zero || S == IntPtr.Zero || Vh == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(U), new Tensor(S), new Tensor(Vh));
            }

            /// <summary>
            /// Computes the singular values of a matrix.
            /// </summary>
            /// <param name="input">The input matrix</param>
            /// <returns></returns>
            public static Tensor svdvals(Tensor input)
            {
                var res = THSLinalg_svdvals(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the multiplicative inverse of torch.tensordot().
            /// </summary>
            /// <param name="input">Tensor to invert. </param>
            /// <param name="ind">Index at which to compute the inverse of torch.tensordot()</param>
            /// <returns></returns>
            public static Tensor tensorinv(Tensor input, long ind)
            {
                var res = THSLinalg_tensorinv(input.Handle, ind);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the solution X to the system torch.tensordot(A, X) = B.
            /// </summary>
            /// <param name="A">Tensor to solve for. </param>
            /// <param name="B">Another tensor, of shape a.shape[B.dim].</param>
            /// <param name="dims">Dimensions of A to be moved. If None, no dimensions are moved.</param>
            /// <returns></returns>
            public static Tensor tensorsolve(Tensor A, Tensor B, long[] dims)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = THSLinalg_tensorsolve(A.Handle, B.Handle, (IntPtr)pdims, dims.Length);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Computes a vector norm.
            /// </summary>
            /// <param name="input">Tensor, flattened by default, but this behavior can be controlled using dims.</param>
            /// <param name="ord">Order of norm. Default: 2</param>
            /// <param name="dims">Dimensions over which to compute the norm.</param>
            /// <param name="keepdim">If set to true, the reduced dimensions are retained in the result as dimensions with size one. </param>
            /// <returns></returns>
            public static Tensor vector_norm(Tensor input, double ord = 2d, long[]? dims = null, bool keepdim = false)
            {
                unsafe {
                    fixed (long* pdims = dims) {
                        var res = THSLinalg_vector_norm(input.Handle, ord.ToScalar().Handle, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Generates a Vandermonde matrix.
            /// </summary>
            /// <param name="input">tensor of shape (*, n) where * is zero or more batch dimensions consisting of vectors.</param>
            /// <param name="N">Number of columns in the output. Default: x.size(-1)</param>
            public static Tensor vander(Tensor input, long? N = null)
            {
                if (!N.HasValue) {
                    N = input.shape[input.ndim - 1];
                }
                var res = THSLinalg_vander(input.Handle, N.Value);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the dot product of two batches of vectors along a dimension.
            /// </summary>
            /// <param name="x">First batch of vectors.</param>
            /// <param name="y">Second batch of vectors</param>
            /// <param name="dim">Dimension along which to compute the dot product.</param>
            /// <param name="out">Optional output tensor.</param>
            public static Tensor vecdot(Tensor x, Tensor y, long dim = -1, Tensor? @out = null)
            {
                var res = THSLinalg_vecdot(x.Handle, y.Handle, dim, @out is null ? IntPtr.Zero : @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the solution of a square system of linear equations with a unique solution given an LU decomposition.
            /// </summary>
            /// <param name="LU">Tensor of shape (*, n, n) (or (*, k, k) if left= True) where * is zero or more batch dimensions as returned by lu_factor().</param>
            /// <param name="pivots">Tensor of shape (*, n) (or (*, k) if left= True) where * is zero or more batch dimensions as returned by lu_factor().</param>
            /// <param name="B">Right-hand side tensor of shape (*, n, k).</param>
            /// <param name="left">Whether to solve the system AX=B or XA = B. Default: True.</param>
            /// <param name="adjoint">Whether to solve the adjoint system.</param>
            /// <param name="out">Optional output tensor.</param>
            /// <returns></returns>
            public static Tensor lu_solve(Tensor LU, Tensor pivots, Tensor B, bool left = true, bool adjoint = false, Tensor? @out = null)
            {
                var res = THSLinalg_lu_solve(B.Handle, LU.Handle, pivots.Handle, left, adjoint, @out is null ? IntPtr.Zero : @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }
        }
    }
}
