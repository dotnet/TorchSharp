using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
namespace TorchSharp.Tensor
{
    public sealed partial class TorchTensor
    {
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cholesky(IntPtr input, bool upper);

        /// <summary>
        /// Computes the Cholesky decomposition of a symmetric positive-definite matrix AA or for batches of symmetric positive-definite matrices.
        /// </summary>
        /// <param name="upper">If upper is true, the returned matrix U is upper-triangular. If upper is false, the returned matrix L is lower-triangular</param>
        /// <returns></returns>
        public TorchTensor cholesky(bool upper = false)
        {
            var res = THSTensor_cholesky(handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cholesky_inverse(IntPtr input, bool upper);

        /// <summary>
        /// Computes the inverse of a symmetric positive-definite matrix AA using its Cholesky factor uu : returns matrix inv
        /// </summary>
        /// <param name="upper"></param>
        /// <returns></returns>
        public TorchTensor cholesky_inverse(bool upper = false)
        {
            var res = THSTensor_cholesky_inverse(handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cholesky_solve(IntPtr input, IntPtr input2, bool upper);

        /// <summary>
        /// Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix u.
        /// </summary>
        /// <param name="input2"></param>
        /// <param name="upper"></param>
        /// <returns></returns>
        public TorchTensor cholesky_solve(TorchTensor input2, bool upper = false)
        {
            var res = THSTensor_cholesky_solve(handle, input2.Handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cross(IntPtr input, IntPtr other, long dim);

        /// <summary>
        /// Returns the cross product of vectors in dimension dim of input and other.
        /// input and other must have the same size, and the size of their dim dimension should be 3.
        /// </summary>
        public TorchTensor cross(TorchScalar other, long dim)
        {
            var res = THSTensor_cross(handle, other.Handle, dim);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
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
        public TorchTensor matmul(TorchTensor target)
        {
            var res = THSTensor_matmul(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mm(IntPtr tensor, IntPtr target);

        /// <summary>
        /// Performs a matrix multiplication of the matrices input and mat2.
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor mm(TorchTensor target)
        {
            var res = THSTensor_mm(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mv(IntPtr tensor, IntPtr target);

        /// <summary>
        /// Performs a matrix-vector product of the matrix input and the vector vec.
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor mv(TorchTensor target)
        {
            var res = THSTensor_mv(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
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
        public TorchTensor vdot(TorchTensor target)
        {
            if (shape.Length != 1 || target.shape.Length != 1 || shape[0] != target.shape[0]) throw new InvalidOperationException("vdot arguments must have the same shape.");
            var res = THSTensor_vdot(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


    }
}
