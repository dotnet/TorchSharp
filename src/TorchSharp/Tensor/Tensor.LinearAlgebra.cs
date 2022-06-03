// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using System.Transactions;
using ICSharpCode.SharpZipLib.BZip2;

namespace TorchSharp
{
    public static partial class torch
    {

        public partial class Tensor
        {
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cholesky(IntPtr input, bool upper);

            /// <summary>
            /// Computes the Cholesky decomposition of a symmetric positive-definite matrix AA or for batches of symmetric positive-definite matrices.
            /// </summary>
            /// <param name="upper">If upper is true, the returned matrix U is upper-triangular. If upper is false, the returned matrix L is lower-triangular</param>
            /// <returns></returns>
            public Tensor cholesky(bool upper = false)
            {
                var res = THSTensor_cholesky(Handle, upper);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cholesky_inverse(IntPtr input, bool upper);

            /// <summary>
            /// Computes the inverse of a symmetric positive-definite matrix AA using its Cholesky factor uu : returns matrix inv
            /// </summary>
            /// <param name="upper"></param>
            /// <returns></returns>
            public Tensor cholesky_inverse(bool upper = false)
            {
                var res = THSTensor_cholesky_inverse(Handle, upper);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cholesky_solve(IntPtr input, IntPtr input2, bool upper);

            /// <summary>
            /// Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix u.
            /// </summary>
            /// <param name="input2"></param>
            /// <param name="upper"></param>
            /// <returns></returns>
            public Tensor cholesky_solve(Tensor input2, bool upper = false)
            {
                var res = THSTensor_cholesky_solve(Handle, input2.Handle, upper);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cross(IntPtr input, IntPtr other, long dim);

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
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the determinant of a square matrix.
            /// </summary>
            public Tensor det()
            {
                return torch.linalg.det(this);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_eig(IntPtr tensor, bool eigenvectors, out IntPtr pEigenvectors);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_matmul(IntPtr tensor, IntPtr target);

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
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mm(IntPtr tensor, IntPtr target);

            /// <summary>
            /// Performs a matrix multiplication of the matrices input and mat2.
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor mm(Tensor target)
            {
                var res = THSTensor_mm(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mv(IntPtr tensor, IntPtr target);

            /// <summary>
            /// Performs a matrix-vector product of the matrix input and the vector vec.
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor mv(Tensor target)
            {
                var res = THSTensor_mv(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_matrix_exp(IntPtr input);

            /// <summary>
            /// Computes the matrix exponential of a square matrix or of each square matrix in a batch.
            /// </summary>
            public Tensor matrix_exp()
            {
                var res = THSTensor_matrix_exp(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSLinalg_matrix_power(IntPtr tensor, long n);

            /// <summary>
            /// Computes the n-th power of a square matrix for an integer n.
            /// </summary>
            /// <param name="n">The exponent</param>
            /// <returns></returns>
            /// <remarks>Input tensor must be of shape (*, m, m) where * is zero or more batch dimensions.</remarks>
            public Tensor matrix_power(int n)
            {
                var res = THSLinalg_matrix_power(Handle, n);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_vdot(IntPtr tensor, IntPtr target);

            /// <summary>
            /// Computes the dot product of two 1D tensors. 
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            /// <remarks>
            /// The vdot(a, b) function handles complex numbers differently than dot(a, b).
            /// If the first argument is complex, the complex conjugate of the first argument is used for the calculation of the dot product.
            /// </remarks>
            public Tensor vdot(Tensor target)
            {
                if (shape.Length != 1 || target.shape.Length != 1 || shape[0] != target.shape[0]) throw new InvalidOperationException("vdot arguments must have the same shape.");
                var res = THSTensor_vdot(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSLinalg_pinverse(IntPtr tensor, double rcond, bool hermitian);

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
                    torch.CheckForErrors();
                return new Tensor(res);
            }
        }


        /// <summary>
        /// Computes the pseudoinverse (Moore-Penrose inverse) of a matrix.
        /// </summary>
        /// <param name="input">Tensor of shape (*, m, n) where * is zero or more batch dimensions.</param>
        /// <param name="rcond">The tolerance value to determine when is a singular value zero </param>
        /// <param name="hermitian">Indicates whether A is Hermitian if complex or symmetric if real. </param>
        /// <remarks>Input should be tensor of shape (*, m, n) where * is zero or more batch dimensions.</remarks>
        public static Tensor pinverse(Tensor input, double rcond = 1e-15, bool hermitian = false) => input.pinverse(rcond, hermitian);


        /// <summary>
        /// Computes the Cholesky decomposition of a symmetric positive-definite matrix 'input' or for batches of symmetric positive-definite matrices.
        /// </summary>
        /// <param name="input">The input matrix</param>
        /// <param name="upper">If upper is true, the returned matrix U is upper-triangular. If upper is false, the returned matrix L is lower-triangular</param>
        /// <returns></returns>
        public static Tensor cholesky(Tensor input, bool upper = false)
        {
            return input.cholesky(upper);
        }

        /// <summary>
        /// Computes the inverse of a symmetric positive-definite matrix AA using its Cholesky factor uu : returns matrix inv
        /// </summary>
        /// <returns></returns>
        public static Tensor cholesky_inverse(Tensor input, bool upper = false)
        {
            return input.cholesky_inverse(upper);
        }

        /// <summary>
        /// Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix u.
        /// </summary>
        /// <returns></returns>
        public static Tensor cholesky_solve(Tensor input, Tensor input2, bool upper = false)
        {
            return input.cholesky_solve(input2, upper);
        }

        /// <summary>
        /// Returns the cross product of vectors in dimension dim of input and other.
        /// input and other must have the same size, and the size of their dim dimension should be 3.
        /// </summary>
        public static Tensor cross(Tensor input, Scalar other, long dim)
        {
            return input.cross(other, dim);
        }

        /// <summary>
        /// Computes the determinant of a square matrix.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor det(Tensor input)
        {
            return torch.linalg.det(input);
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
        /// <param name="dim1">First dimension with respect to which to take diagonal. Default: 0.</param>
        /// <param name="dim2">Second dimension with respect to which to take diagonal. Default: 1.</param>
        /// <remarks>
        /// Applying torch.diag_embed() to the output of this function with the same arguments yields a diagonal matrix with the diagonal entries of the input.
        /// However, torch.diag_embed() has different default dimensions, so those need to be explicitly specified.
        /// </remarks>
        public static Tensor diagonal(Tensor input, long offset = 0, long dim1 = 0, long dim2 = 0) => input.diagonal(offset, dim1, dim2);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lu(IntPtr tensor, bool pivot, bool get_infos, out IntPtr infos, out IntPtr pivots);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lu_solve(IntPtr tensor, IntPtr LU_data, IntPtr LU_pivots);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_lu_unpack(IntPtr LU_data, IntPtr LU_pivots, bool unpack_data, bool unpack_pivots, out IntPtr L, out IntPtr U);


        /// <summary>
        /// Computes the LU factorization of a matrix or batches of matrices A. Returns a tuple containing the LU factorization and pivots of A. Pivoting is done if pivot is set to true.
        /// </summary>
        /// <param name="A">The tensor to factor of size (∗,m,n)</param>
        /// <param name="pivot">Controls whether pivoting is done. Default: true</param>
        /// <param name="get_infos">If set to True, returns an info IntTensor. Default: false</param>
        /// <returns></returns>
        public static (Tensor A_LU, Tensor pivots, Tensor infos) lu(Tensor A, bool pivot = true, bool get_infos = false)
        {
            var solution = THSTensor_lu(A.Handle, pivot, get_infos, out var infos, out var pivots);
            if (solution == IntPtr.Zero)
                torch.CheckForErrors();
            return (new Tensor(solution), pivots == IntPtr.Zero ? null : new Tensor(pivots), infos == IntPtr.Zero ? null : new Tensor(infos));
        }

        /// <summary>
        /// Returns the LU solve of the linear system Ax = b using the partially pivoted LU factorization of A from torch.lu().
        /// </summary>
        /// <param name="b">The RHS tensor of size (∗,m,k), where *∗ is zero or more batch dimensions.</param>
        /// <param name="LU_data">The pivoted LU factorization of A from torch.lu() of size (∗,m,m), where *∗ is zero or more batch dimensions.</param>
        /// <param name="LU_pivots">
        /// The pivots of the LU factorization from torch.lu() of size (∗,m), where *∗ is zero or more batch dimensions.
        /// The batch dimensions of LU_pivots must be equal to the batch dimensions of LU_data.</param>
        /// <returns></returns>
        public static Tensor lu_solve(Tensor b, Tensor LU_data, Tensor LU_pivots)
        {
            var solution = THSTensor_lu_solve(b.Handle, LU_data.Handle, LU_pivots.Handle);
            if (solution == IntPtr.Zero)
                torch.CheckForErrors();
            return new Tensor(solution);
        }

        /// <summary>
        /// Unpacks the data and pivots from a LU factorization of a tensor into tensors L and U and a permutation tensor P.
        /// </summary>
        /// <param name="LU_data">The packed LU factorization data</param>
        /// <param name="LU_pivots">The packed LU factorization pivots</param>
        /// <param name="unpack_data">A flag indicating if the data should be unpacked. If false, then the returned L and U are null. Default: true</param>
        /// <param name="unpack_pivots">A flag indicating if the pivots should be unpacked into a permutation matrix P. If false, then the returned P is null. Default: true</param>
        /// <returns>A tuple of three tensors to use for the outputs (P, L, U)</returns>
        public static (Tensor, Tensor, Tensor) lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data = true, bool unpack_pivots = true)
        {
            var solution = THSTensor_lu_unpack(LU_data.Handle, LU_pivots.Handle, unpack_data, unpack_pivots, out var L, out var U);
            if (solution == IntPtr.Zero)
                torch.CheckForErrors();
            return (new Tensor(solution), L == IntPtr.Zero ? null : new Tensor(L), U == IntPtr.Zero ? null : new Tensor(U));
        }


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
        public static Tensor matmul(Tensor input, Tensor target)
        {
            return input.matmul(target);
        }

        /// <summary>
        /// Performs a matrix multiplication of the matrices input and mat2.
        /// </summary>
        /// <returns></returns>
        public static Tensor mm(Tensor input, Tensor target)
        {
            return input.mm(target);
        }

        /// <summary>
        /// Performs a matrix-vector product of the matrix input and the vector vec.
        /// </summary>
        /// <returns></returns>
        public static Tensor mv(Tensor input, Tensor target)
        {
            return input.mv(target);
        }

        /// <summary>
        /// Computes the n-th power of a square matrix for an integer n.
        /// </summary>
        /// <param name="input">The input square matrix.</param>
        /// <param name="n">The exponent</param>
        /// <returns></returns>
        /// <remarks>Input tensor must be of shape (*, m, m) where * is zero or more batch dimensions.</remarks>
        public static Tensor matrix_power(Tensor input, int n)
        {
            return input.matrix_power(n);
        }

        public static Tensor norm(Tensor input) => input.norm();

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
        public static Tensor vdot(Tensor input, Tensor target)
        {
            return input.vdot(target);
        }
    }
}