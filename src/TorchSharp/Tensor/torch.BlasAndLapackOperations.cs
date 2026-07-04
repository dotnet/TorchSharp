// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable

using System;
using System.Collections.Generic;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#blas-and-lapack-operations
    public static partial class torch
    {
        public enum LobpcgMethod
        {
            basic,
            ortho
        }

        // https://pytorch.org/docs/stable/generated/torch.addbmm
        /// <summary>
        /// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced
        /// add step (all matrix multiplications get accumulated along the first dimension).
        /// input is added to the final result.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="batch1">The first batch of matrices to be multiplied</param>
        /// <param name="batch2">The second batch of matrices to be multiplied</param>
        /// <param name="beta">Nultiplier for input (β)</param>
        /// <param name="alpha">Multiplier for batch1 @ batch2 (α)</param>
        public static Tensor addbmm(Tensor input, Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1)
            => input.addbmm(batch1, batch2, beta, alpha);

        /// <summary>
        /// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced
        /// add step (all matrix multiplications get accumulated along the first dimension).
        /// input is added to the final result.
        /// In-place version of addbmm.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="batch1">The first batch of matrices to be multiplied</param>
        /// <param name="batch2">The second batch of matrices to be multiplied</param>
        /// <param name="beta">Nultiplier for input (β)</param>
        /// <param name="alpha">Multiplier for batch1 @ batch2 (α)</param>
        public static Tensor addbmm_(Tensor input, Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1)
            => input.addbmm_(batch1, batch2, beta, alpha);

        // https://pytorch.org/docs/stable/generated/torch.addmm
        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="mat1">First matrix</param>
        /// <param name="mat2">Second matrix</param>
        /// <param name="beta">Input scale factor</param>
        /// <param name="alpha">Matrix multiplication scale factor</param>
        public static Tensor addmm(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmm(mat1, mat2, beta, alpha);

        // https://pytorch.org/docs/stable/generated/torch.addmm
        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="mat1">First matrix</param>
        /// <param name="mat2">Second matrix</param>
        /// <param name="beta">Input scale factor</param>
        /// <param name="alpha">Matrix multiplication scale factor</param>
        public static Tensor addmm_(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmm_(mat1, mat2, beta, alpha);

        // https://pytorch.org/docs/stable/generated/torch.addmv

        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="mat1">First matrix</param>
        /// <param name="mat2">Second matrix</param>
        /// <param name="beta">Input scale factor</param>
        /// <param name="alpha">Matrix multiplication scale factor</param>
        public static Tensor addmv(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmv(mat1, mat2, beta, alpha);

        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="mat1">First matrix</param>
        /// <param name="mat2">Second matrix</param>
        /// <param name="beta">Input scale factor</param>
        /// <param name="alpha">Matrix multiplication scale factor</param>
        public static Tensor addmv_(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmv_(mat1, mat2, beta, alpha);

        // https://pytorch.org/docs/stable/generated/torch.addr
        /// <summary>
        /// Performs the outer-product of vectors vec1 and vec2 and adds it to the input tensor.
        /// </summary>
        /// <returns></returns>
        public static Tensor addr(Tensor input, Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f) => input.addr(vec1, vec2, beta, alpha);

        // https://pytorch.org/docs/stable/generated/torch.addr
        /// <summary>
        /// Performs the outer-product of vectors vec1 and vec2 and adds it to the input tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="vec1">The first vector of the outer product</param>
        /// <param name="vec2">The second vector of the outer product</param>
        /// <param name="beta">Input scale factor</param>
        /// <param name="alpha">Outer-product scale factor</param>
        public static Tensor addr_(Tensor input, Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f) => input.addr_(vec1, vec2, beta, alpha);

        // https://pytorch.org/docs/stable/generated/torch.baddbmm
        /// <summary>
        /// Performs a batch matrix-matrix product of matrices in batch1 and batch2. input is added to the final result.
        /// batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
        /// </summary>
        /// <param name="input">The tensor to be added</param>
        /// <param name="batch1">The first batch of matrices to be multiplied</param>
        /// <param name="batch2">The second batch of matrices to be multiplied</param>
        /// <param name="beta">A multiplier for input</param>
        /// <param name="alpha">A multiplier for batch1 @ batch2</param>
        public static Tensor baddbmm(Tensor input, Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1) => input.baddbmm(batch1, batch2, beta, alpha);

        // https://pytorch.org/docs/stable/generated/torch.bmm
        /// <summary>
        /// Performs a batch matrix-matrix product of matrices stored in input and mat2.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        public static Tensor bmm(Tensor input, Tensor batch2) => input.bmm(batch2);

        // https://pytorch.org/docs/stable/generated/torch.chain_matmul
        public static Tensor chain_matmul(params Tensor[] matrices) => torch.linalg.multi_dot(matrices);

        // https://pytorch.org/docs/stable/generated/torch.cholesky

        [Obsolete("torch.cholesky is deprecated in favor of torch.linalg.cholesky and will be removed in a future release. Use torch.linalg.cholesky instead.", false)]
#pragma warning disable CS0618 // Obsolete
        public static Tensor cholesky(Tensor input) => input.cholesky();
#pragma warning restore CS0618

        // https://pytorch.org/docs/stable/generated/torch.cholesky_inverse
        /// <summary>
        /// Computes the inverse of a symmetric positive-definite matrix AA using its Cholesky factor uu : returns matrix inv
        /// </summary>
        /// <returns></returns>
        public static Tensor cholesky_inverse(Tensor input, bool upper = false)
            => input.cholesky_inverse(upper);

        // https://pytorch.org/docs/stable/generated/torch.cholesky_solve
        /// <summary>
        /// Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix u.
        /// </summary>
        /// <returns></returns>
        public static Tensor cholesky_solve(Tensor input, Tensor input2, bool upper = false)
            => input.cholesky_solve(input2, upper);

        // https://pytorch.org/docs/stable/generated/torch.dot
        /// <summary>
        /// Computes the dot product of two 1D tensors.
        /// </summary>
        public static Tensor dot(Tensor input, Tensor other) => input.dot(other);

        // https://pytorch.org/docs/stable/generated/torch.eig
        [Obsolete("Method removed in Pytorch. Please use the `torch.linalg.eig` function instead.", true)]
        public static (Tensor eigenvalues, Tensor eigenvectors) eig(Tensor input, bool eigenvectors = false) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.geqrf
        /// <summary>
        /// This is a low-level function for calling LAPACK’s geqrf directly.
        /// This function returns a namedtuple (a, tau) as defined in LAPACK documentation for geqrf.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <remarks>
        /// Computes a QR decomposition of input. Both Q and R matrices are stored in the same output tensor a.
        /// The elements of R are stored on and above the diagonal. Elementary reflectors (or Householder vectors)
        /// implicitly defining matrix Q are stored below the diagonal. The results of this function can be used
        /// together with torch.linalg.householder_product() to obtain the Q matrix or with torch.ormqr(), which
        /// uses an implicit representation of the Q matrix, for an efficient matrix-matrix multiplication.
        /// </remarks>
        public static (Tensor a, Tensor tau) geqrf(Tensor input) => input.geqrf();

        // https://pytorch.org/docs/stable/generated/torch.ger
        /// <summary>
        /// Outer product of input and vec2.
        /// </summary>
        /// <param name="input">The input vector.</param>
        /// <param name="vec2">1-D input vector.</param>
        /// <remarks>If input is a vector of size n and vec2 is a vector of size m, then out must be a matrix of size n×m.</remarks>
        public static Tensor ger(Tensor input, Tensor vec2) => input.ger(vec2);

        // https://pytorch.org/docs/stable/generated/torch.inner
        /// <summary>
        /// Computes the dot product for 1D tensors.
        /// For higher dimensions, sums the product of elements from input and other along their last dimension.
        /// </summary>
        public static Tensor inner(Tensor input, Tensor vec2) => input.inner(vec2);

        // https://pytorch.org/docs/stable/generated/torch.inverse
        /// <summary>
        /// Alias for torch.linalg.inv()
        /// </summary>
        public static Tensor inverse(Tensor input) => linalg.inv(input);

        // https://pytorch.org/docs/stable/generated/torch.det
        /// <summary>
        /// Computes the determinant of a square matrix.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor det(Tensor input) => input.det();

        // https://pytorch.org/docs/stable/generated/torch.logdet
        public static Tensor logdet(Tensor input) => input.logdet();

        // https://pytorch.org/docs/stable/generated/torch.slogdet
        public static (Tensor res, Tensor logabsdet) slogdet(Tensor A) => torch.linalg.slogdet(A);

        // https://pytorch.org/docs/stable/generated/torch.lstsq
        /// <summary>
        /// Computes the solution to the least squares and least norm problems for a full rank matrix A of size m×n and a matrix B of size m×k.
        /// </summary>
        /// <param name="A">the m by n matrix AA</param>
        /// <param name="B">the matrix BB</param>
        /// <returns></returns>
        public static (Tensor Solution, Tensor QR) lstsq(Tensor B, Tensor A)
        {
            var solution = THSTorch_lstsq(B.Handle, A.Handle, out var qr);
            if (solution == IntPtr.Zero || qr == IntPtr.Zero)
                CheckForErrors();
            return (new Tensor(solution), new Tensor(qr));
        }

        // https://pytorch.org/docs/stable/generated/torch.lu
        /// <summary>
        /// Computes the LU factorization of a matrix or batches of matrices A. Returns a tuple containing the LU factorization and pivots of A. Pivoting is done if pivot is set to true.
        /// </summary>
        /// <param name="A">The tensor to factor of size (∗,m,n)</param>
        /// <param name="pivot">Controls whether pivoting is done. Default: true</param>
        /// <param name="get_infos">If set to True, returns an info IntTensor. Default: false</param>
        /// <returns></returns>
        public static (Tensor A_LU, Tensor? pivots, Tensor? infos) lu(Tensor A, bool pivot = true, bool get_infos = false)
        {
            var solution = THSTensor_lu(A.Handle, pivot, get_infos, out var infos, out var pivots);
            if (solution == IntPtr.Zero)
                torch.CheckForErrors();
            return (new Tensor(solution), pivots == IntPtr.Zero ? null : new Tensor(pivots), infos == IntPtr.Zero ? null : new Tensor(infos));
        }

        // https://pytorch.org/docs/stable/generated/torch.lu_solve
        /// <summary>
        /// Returns the LU solve of the linear system Ax = b using the partially pivoted LU factorization of A from torch.lu().
        /// </summary>
        /// <param name="b">The RHS tensor of size (∗,m,k), where *∗ is zero or more batch dimensions.</param>
        /// <param name="LU_data">The pivoted LU factorization of A from torch.lu() of size (∗,m,m), where *∗ is zero or more batch dimensions.</param>
        /// <param name="LU_pivots">
        /// The pivots of the LU factorization from torch.lu() of size (∗,m), where *∗ is zero or more batch dimensions.
        /// The batch dimensions of LU_pivots must be equal to the batch dimensions of LU_data.</param>
        /// <returns></returns>
        [Obsolete("torch.lu_solve is deprecated in favor of torch.linalg.lu_solve and will be removed in a future release. Use torch.linalg.lu_solve(LU, pivots, B) instead.", false)]
        public static Tensor lu_solve(Tensor b, Tensor LU_data, Tensor LU_pivots)
        {
            var solution = THSTensor_lu_solve(b.Handle, LU_data.Handle, LU_pivots.Handle);
            if (solution == IntPtr.Zero)
                CheckForErrors();
            return new Tensor(solution);
        }

        // https://pytorch.org/docs/stable/generated/torch.lu_unpack
        /// <summary>
        /// Unpacks the data and pivots from a LU factorization of a tensor into tensors L and U and a permutation tensor P.
        /// </summary>
        /// <param name="LU_data">The packed LU factorization data</param>
        /// <param name="LU_pivots">The packed LU factorization pivots</param>
        /// <param name="unpack_data">A flag indicating if the data should be unpacked. If false, then the returned L and U are null. Default: true</param>
        /// <param name="unpack_pivots">A flag indicating if the pivots should be unpacked into a permutation matrix P. If false, then the returned P is null. Default: true</param>
        /// <returns>A tuple of three tensors to use for the outputs (P, L, U)</returns>
        public static (Tensor P, Tensor? L, Tensor? U) lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data = true, bool unpack_pivots = true)
        {
            var solution = THSTensor_lu_unpack(LU_data.Handle, LU_pivots.Handle, unpack_data, unpack_pivots, out var L, out var U);
            if (solution == IntPtr.Zero)
                CheckForErrors();
            return (new Tensor(solution), L == IntPtr.Zero ? null : new Tensor(L), U == IntPtr.Zero ? null : new Tensor(U));
        }

        // https://pytorch.org/docs/stable/generated/torch.matmul
        /// <summary>
        /// Matrix product of two tensors.
        /// </summary>
        /// <returns></returns>
        /// <remarks>
        /// The behavior depends on the dimensionality of the tensors as follows:
        /// 1. If both tensors are 1-dimensional, the dot product (scalar) is returned
        /// 2. If both arguments are 2-dimensional, the matrix-matrix product is returned.
        /// 3. If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
        /// 4. If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
        /// 5. If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiply is returned.
        /// </remarks>
        public static Tensor matmul(Tensor input, Tensor target) => input.matmul(target);

        // https://pytorch.org/docs/stable/generated/torch.matrix_power
        /// <summary>
        /// Computes the n-th power of a square matrix for an integer n.
        /// </summary>
        /// <param name="input">The input square matrix.</param>
        /// <param name="n">The exponent</param>
        /// <returns></returns>
        /// <remarks>Input tensor must be of shape (*, m, m) where * is zero or more batch dimensions.</remarks>
        public static Tensor matrix_power(Tensor input, int n) => input.matrix_power(n);

        // https://pytorch.org/docs/stable/generated/torch.matrix_rank
        [Obsolete("This function was deprecated since version 1.9 and is now removed. Please use the 'torch.linalg.matrix_rank' function instead.", true)]
        public static Tensor matrix_rank(Tensor input, float? tol = null, bool symmetric = false) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.matrix_exp
        /// <summary>
        /// Computes the matrix exponential of a square matrix or of each square matrix in a batch.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor matrix_exp(Tensor input) => input.matrix_exp();

        // https://pytorch.org/docs/stable/generated/torch.mm
        /// <summary>
        /// Performs a matrix multiplication of the matrices input and mat2.
        /// </summary>
        /// <returns></returns>
        public static Tensor mm(Tensor input, Tensor target) => input.mm(target);

        // https://pytorch.org/docs/stable/generated/torch.mv
        /// <summary>
        /// Performs a matrix-vector product of the matrix input and the vector vec.
        /// </summary>
        /// <returns></returns>
        public static Tensor mv(Tensor input, Tensor target) => input.mv(target);

        // https://pytorch.org/docs/stable/generated/torch.orgqr
        /// <summary>
        /// Computes the first n columns of a product of Householder matrices.
        /// </summary>
        /// <param name="input">tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
        /// <param name="tau">tensor of shape (*, k) where * is zero or more batch dimensions.</param>
        public static Tensor orgqr(Tensor input, Tensor tau) => linalg.householder_product(input, tau);

        // https://pytorch.org/docs/stable/generated/torch.ormqr
        /// <summary>
        /// Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix.
        /// </summary>
        /// <param name="input">Tensor of shape (*, mn, k) where * is zero or more batch dimensions and mn equals to m or n depending on the left.</param>
        /// <param name="tau">Tensor of shape (*, min(mn, k)) where * is zero or more batch dimensions.</param>
        /// <param name="other">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
        /// <param name="left">Controls the order of multiplication.</param>
        /// <param name="transpose">Controls whether the matrix Q is conjugate transposed or not.</param>
        public static Tensor ormqr(Tensor input, Tensor tau, Tensor other, bool left=true, bool transpose=false) => input.ormqr(tau, other, left, transpose);       

        // https://pytorch.org/docs/stable/generated/torch.outer
        /// <summary>
        /// Outer product of input and vec2.
        /// </summary>
        /// <param name="input">1-D input vector.</param>
        /// <param name="vec2">1-D input vector.</param>
        /// <remarks>If input is a vector of size n and vec2 is a vector of size m, then out must be a matrix of size n×m.</remarks>
        public static Tensor outer(Tensor input, Tensor vec2) => input.outer(vec2);

        // https://pytorch.org/docs/stable/generated/torch.pinverse
        /// <summary>
        /// Computes the pseudoinverse (Moore-Penrose inverse) of a matrix.
        /// </summary>
        /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
        /// <param name="rcond">The tolerance value to determine when is a singular value zero </param>
        /// <param name="hermitian">Indicates whether A is Hermitian if complex or symmetric if real. </param>
        /// <remarks>Input should be tensor of shape (*, m, n) where * is zero or more batch dimensions.</remarks>
        public static Tensor pinverse(Tensor input, double rcond = 1e-15, bool hermitian = false) => input.pinverse(rcond, hermitian);

        // https://pytorch.org/docs/stable/generated/torch.qr
        [Obsolete("torch.qr() is deprecated in favor of torch.linalg.qr() and will be removed in a future PyTorch release.", true)]
        public static Tensor qr(Tensor input, bool some = true) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.svd
        [Obsolete("torch.qr() is deprecated in favor of torch.linalg.svd() and will be removed in a future PyTorch release.", true)]
        public static Tensor svd(Tensor input, bool some=true, bool compute_uv=true) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.svd_lowrank
        // NOTE TO SELF: there's no native method for this. PyTorch implements it in Python.
        [Obsolete("not implemented", true)]
        public static Tensor svd_lowrank(Tensor A, int q=6, int niter=2,Tensor? M=null) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.pca_lowrank
        // NOTE TO SELF: there's no native method for this. PyTorch implements it in Python.
        [Obsolete("not implemented", true)]
        public static Tensor pca_lowrank(Tensor A, int q=6, bool center=true, int niter=2) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.symeig
        [Obsolete("torch.symeig() is deprecated in favor of torch.linalg.eigh() and will be removed in a future PyTorch release", true)]
        public static Tensor symeig(Tensor input, bool eigenvectors = false, bool upper = true) => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.lobpcg
        [Obsolete("not implemented", true)]
        static Tensor lobpcg(
                Tensor A,
                int k = 1,
                Tensor? B = null,
                Tensor? X = null,
                int? n = null,
                Tensor? iK = null,
                int? niter = null,
                float? tol=null,
                bool largest=true,
                LobpcgMethod method = LobpcgMethod.ortho,
                Action? tracker=null,
                IDictionary<string, string>? ortho_iparams=null,
                IDictionary<string, string>? ortho_fparams=null,
                IDictionary<string, string>? ortho_bparams=null)
            => throw new NotImplementedException();

        /// <summary>
        /// Computes the softmax function for the input tensor.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">A dimension along which softmax will be computed.</param>
        /// <param name="dtype">The desired data type of returned tensor.</param>
        public static Tensor softmax(Tensor input, int dim, ScalarType? dtype = null)
            => torch.special.softmax(input, dim, dtype);

        // https://pytorch.org/docs/stable/generated/torch.trapz
        /// <summary>
        /// Computes the trapezoidal rule along dim. By default the spacing between elements is assumed
        /// to be 1, but dx can be used to specify a different constant spacing, and x can be used to specify arbitrary spacing along dim.
        /// </summary>
        /// <param name="y">Values to use when computing the trapezoidal rule.</param>
        /// <param name="x">Defines spacing between values as specified above.</param>
        /// <param name="dim">The dimension along which to compute the trapezoidal rule. The last (inner-most) dimension by default.</param>
        public static Tensor trapz(Tensor y, Tensor x, long dim = -1) => trapezoid(y, x, dim);

        /// <summary>
        /// Computes the trapezoidal rule along dim. By default the spacing between elements is assumed
        /// to be 1, but dx can be used to specify a different constant spacing, and x can be used to specify arbitrary spacing along dim.
        /// </summary>
        /// <param name="y">Values to use when computing the trapezoidal rule.</param>
        /// <param name="dx">Constant spacing between values.</param>
        /// <param name="dim">The dimension along which to compute the trapezoidal rule. The last (inner-most) dimension by default.</param>
        public static Tensor trapz(Tensor y, double dx = 1, long dim = -1) => trapezoid(y, dx, dim);

        // https://pytorch.org/docs/stable/generated/torch.trapezoid
        /// <summary>
        /// Computes the trapezoidal rule along dim. By default the spacing between elements is assumed
        /// to be 1, but dx can be used to specify a different constant spacing, and x can be used to specify arbitrary spacing along dim.
        /// </summary>
        /// <param name="y">Values to use when computing the trapezoidal rule.</param>
        /// <param name="x">Defines spacing between values as specified above.</param>
        /// <param name="dim">The dimension along which to compute the trapezoidal rule. The last (inner-most) dimension by default.</param>
        public static Tensor trapezoid(Tensor y, Tensor x, long dim = -1) => y.trapezoid(x, dim);

        /// <summary>
        /// Computes the trapezoidal rule along dim. By default the spacing between elements is assumed
        /// to be 1, but dx can be used to specify a different constant spacing, and x can be used to specify arbitrary spacing along dim.
        /// </summary>
        /// <param name="y">Values to use when computing the trapezoidal rule.</param>
        /// <param name="dx">Constant spacing between values.</param>
        /// <param name="dim">The dimension along which to compute the trapezoidal rule. The last (inner-most) dimension by default.</param>
        public static Tensor trapezoid(Tensor y, double dx = 1, long dim = -1) => y.trapezoid(dx, dim);

        // https://pytorch.org/docs/stable/generated/torch.cumulative_trapezoid
        /// <summary>
        /// Cumulatively computes the trapezoidal rule along dim. By default the spacing between elements is assumed
        /// to be 1, but dx can be used to specify a different constant spacing, and x can be used to specify arbitrary spacing along dim.
        /// </summary>
        /// <param name="y">Values to use when computing the trapezoidal rule.</param>
        /// <param name="x">Defines spacing between values as specified above.</param>
        /// <param name="dim">The dimension along which to compute the trapezoidal rule. The last (inner-most) dimension by default.</param>
        public static Tensor cumulative_trapezoid(Tensor y, Tensor x, long dim = -1) => y.cumulative_trapezoid(x, dim);

        /// <summary>
        /// Cumulatively computes the trapezoidal rule along dim. By default the spacing between elements is assumed
        /// to be 1, but dx can be used to specify a different constant spacing, and x can be used to specify arbitrary spacing along dim.
        /// </summary>
        /// <param name="y">Values to use when computing the trapezoidal rule.</param>
        /// <param name="dx">Constant spacing between values.</param>
        /// <param name="dim">The dimension along which to compute the trapezoidal rule. The last (inner-most) dimension by default.</param>
        public static Tensor cumulative_trapezoid(Tensor y, double dx = 1, long dim = -1) => y.cumulative_trapezoid(dx, dim);

        // https://pytorch.org/docs/stable/generated/torch.triangular_solve
        [Obsolete("torch.triangular_solve() is deprecated in favor of torch.linalg.solve_triangular() and will be removed in a future PyTorch release.", true)]
        static Tensor triangular_solve(
                Tensor b,
                Tensor A,
                bool upper = true,
                bool transpose = false,
                bool unitriangular = false)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.vdot
        /// <summary>
        /// Computes the dot product of two 1D tensors.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        /// <remarks>
        /// The vdot(a, b) function handles complex numbers differently than dot(a, b).
        /// If the first argument is complex, the complex conjugate of the first argument is used for the calculation of the dot product.
        /// </remarks>
        public static Tensor vdot(Tensor input, Tensor target) => input.vdot(target);
    }
}