using System;
using System.Globalization;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp
{
    public static partial class torch
    {

        public sealed partial class Tensor
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
                var res = THSTensor_cholesky(handle, upper);
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
                var res = THSTensor_cholesky_inverse(handle, upper);
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
                var res = THSTensor_cholesky_solve(handle, input2.Handle, upper);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cross(IntPtr input, IntPtr other, long dim);

            /// <summary>
            /// Returns the cross product of vectors in dimension dim of input and other.
            /// input and other must have the same size, and the size of their dim dimension should be 3.
            /// </summary>
            public Tensor cross(Scalar other, long dim)
            {
                var res = THSTensor_cross(handle, other.Handle, dim);
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
                var res = THSTensor_matmul(handle, target.Handle);
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
                var res = THSTensor_mm(handle, target.Handle);
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
                var res = THSTensor_mv(handle, target.Handle);
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
                var res = THSTensor_matrix_exp(handle);
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
                var res = THSLinalg_matrix_power(handle, n);
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
                var res = THSTensor_vdot(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

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
        public static Tensor det(Tensor input)
        {
            return torch.linalg.det(input);
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