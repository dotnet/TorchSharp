using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using TorchSharp.Tensor;

#nullable enable
namespace TorchSharp
{
    public static class linalg
    {
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_cholesky(IntPtr tensor);

        public static TorchTensor cholesky(TorchTensor input)
        {
            var res = THSLinalg_cholesky(input.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_det(IntPtr tensor);

        public static TorchTensor det(TorchTensor input)
        {
            var res = THSLinalg_det(input.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_slogdet(IntPtr tensor, out IntPtr pLogabsdet);

        public static (TorchTensor, TorchTensor) slogdet(TorchTensor input)
        {
            var res = THSLinalg_slogdet(input.Handle, out var logabsdet);
            if (res == IntPtr.Zero || logabsdet == IntPtr.Zero)
                Torch.CheckForErrors();
            return (new TorchTensor(res), new TorchTensor(logabsdet));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_eig(IntPtr tensor, out IntPtr pEigenvectors);

        public static (TorchTensor, TorchTensor) eig(TorchTensor input)
        {
            var res = THSLinalg_eig(input.Handle, out var vectors);
            if (res == IntPtr.Zero || vectors == IntPtr.Zero)
                Torch.CheckForErrors();
            return (new TorchTensor(res), new TorchTensor(vectors));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_eigh(IntPtr tensor, byte UPLO, out IntPtr pEigenvectors);

        public static (TorchTensor,TorchTensor) eigh(TorchTensor input, char UPLO)
        {
            var res = THSLinalg_eigh(input.Handle, (byte)UPLO, out var vectors);
            if (res == IntPtr.Zero || vectors == IntPtr.Zero)
                Torch.CheckForErrors();
            return (new TorchTensor(res), new TorchTensor(vectors));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_eigvals(IntPtr tensor);

        public static TorchTensor eigvals(TorchTensor input)
        {
            var res = THSLinalg_eigvals(input.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_eigvalsh(IntPtr tensor, byte UPLO);

        public static TorchTensor eigvalsh(TorchTensor input, char UPLO = 'L')
        {
            var res = THSLinalg_eigvalsh(input.Handle, (byte)UPLO);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_inv(IntPtr tensor);

        public static TorchTensor inv(TorchTensor input)
        {
            var res = THSLinalg_inv(input.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_lstsq_none(IntPtr tensor, IntPtr other, out IntPtr pResiduals, out IntPtr pRank, out IntPtr pSingularValues);
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_lstsq_rcond(IntPtr tensor, IntPtr other, double rcond, out IntPtr pResiduals, out IntPtr pRank, out IntPtr pSingularValues);

        public static (TorchTensor Solution, TorchTensor Residuals, TorchTensor Rank, TorchTensor SingularValues) lstsq(TorchTensor input, TorchTensor other)
        {
            var solution = THSLinalg_lstsq_none(input.Handle, other.Handle, out var residuals, out var rank, out var singularValues);
            if (solution == IntPtr.Zero || residuals == IntPtr.Zero || rank == IntPtr.Zero || singularValues == IntPtr.Zero)
                Torch.CheckForErrors();
            return (new TorchTensor(solution), new TorchTensor(residuals), new TorchTensor(rank), new TorchTensor(singularValues));
        }

        public static (TorchTensor Solution, TorchTensor Residuals, TorchTensor Rank, TorchTensor SingularValues) lstsq(TorchTensor input, TorchTensor other, double rcond)
        {
            var solution = THSLinalg_lstsq_rcond(input.Handle, other.Handle, rcond, out var residuals, out var rank, out var singularValues);
            if (solution == IntPtr.Zero || residuals == IntPtr.Zero || rank == IntPtr.Zero || singularValues == IntPtr.Zero)
                Torch.CheckForErrors();
            return (new TorchTensor(solution), new TorchTensor(residuals), new TorchTensor(rank), new TorchTensor(singularValues));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_matrix_norm_fronuc(IntPtr tensor, byte fronuc, IntPtr dim, int dim_length, bool keepdim);

        /// <summary>
        /// Computes a matrix norm.
        /// </summary>
        /// <param name="input">tensor with two or more dimensions.
        /// By default its shape is interpreted as (*, m, n) where * is zero or more batch dimensions, but this behavior can be controlled using dims.</param>
        /// <param name="ord">Order of norm. Default: "fro"</param>
        /// <param name="dims">Dimensions over which to compute the norm.</param>
        /// <param name="keepdim">If set to true, the reduced dimensions are retained in the result as dimensions with size one. </param>
        /// <returns></returns>
        public static TorchTensor matrix_norm(TorchTensor input, string ord = "fro", long[]? dims = null, bool keepdim = false)
        {
            if (dims == null) dims = new long[] { -2, -1 };
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSLinalg_matrix_norm_fronuc(input.Handle, ord == "fro" ? (byte) 0 : (byte) 1, (IntPtr)pdims, dims.Length, keepdim);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_matrix_norm(IntPtr tensor, IntPtr ord, IntPtr dim, int dim_length, bool keepdim);

        /// <summary>
        /// Computes a matrix norm.
        /// </summary>
        /// <param name="input">tensor with two or more dimensions.
        /// By default its shape is interpreted as (*, m, n) where * is zero or more batch dimensions, but this behavior can be controlled using dims.</param>
        /// <param name="ord">Order of norm.</param>
        /// <param name="dims">Dimensions over which to compute the norm.</param>
        /// <param name="keepdim">If set to true, the reduced dimensions are retained in the result as dimensions with size one. </param>
        /// <returns></returns>
        public static TorchTensor matrix_norm(TorchTensor input, double ord, long[]? dims = null, bool keepdim = false)
        {
            if (dims == null) dims = new long[] { -2, -1 };
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSLinalg_matrix_norm(input.Handle, ord.ToScalar().Handle, (IntPtr)pdims, dims.Length, keepdim);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_matrix_rank(IntPtr tensor, double tol, bool has_tol, bool hermitian);

        public static TorchTensor matrix_rank(TorchTensor input, double? tol = null, bool hermitian = false)
        {
            unsafe {
                var res = THSLinalg_matrix_rank(input.Handle, tol ?? double.NegativeInfinity, tol.HasValue, hermitian);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSLinalg_multi_dot(IntPtr tensor, int len);

        /// <summary>
        /// Efficiently multiplies two or more matrices by reordering the multiplications so that the fewest arithmetic operations are performed.
        /// </summary>
        /// <param name="tensors">Two or more tensors to multiply. The first and last tensors may be 1D or 2D. Every other tensor must be 2D.</param>
        /// <returns></returns>
        public static TorchTensor multi_dot(IList<TorchTensor> tensors)
        {
            if (tensors.Count == 0) {
                throw new ArgumentException(nameof(tensors));
            }
            if (tensors.Count == 1) {
                return tensors[0];
            }

            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                return new TorchTensor(THSLinalg_multi_dot(tensorsRef, parray.Array.Length));
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_norm_str(IntPtr tensor, [MarshalAs(UnmanagedType.LPStr)] string p, IntPtr dim, int dim_length, bool keepdim);
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_norm_float(IntPtr tensor, double p, IntPtr dim, int dim_length, bool keepdim);
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_norm_int(IntPtr tensor, int p, IntPtr dim, int dim_length, bool keepdim);
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_norm_opt(IntPtr tensor, IntPtr dim, int dim_length, bool keepdim);


        public static TorchTensor norm(TorchTensor input, string ord, long[]? dims = null, bool keepdim = false)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSLinalg_norm_str(input.Handle, ord, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        public static TorchTensor norm(TorchTensor input, double ord, long[]? dims = null, bool keepdim = false)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSLinalg_norm_float(input.Handle, ord, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        public static TorchTensor norm(TorchTensor input, int ord, long[]? dims = null, bool keepdim = false)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSLinalg_norm_int(input.Handle, ord, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        public static TorchTensor norm(TorchTensor input, long[]? dims = null, bool keepdim = false)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSLinalg_norm_opt(input.Handle, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_pinv(IntPtr tensor, double rcond, bool hermitian);

        public static TorchTensor pinv(TorchTensor input, double rcond = 1e-15, bool hermitian = false)
        {
            var res = THSLinalg_pinv(input.Handle, rcond, hermitian);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        public enum QRMode
        {
            Reduced = 0,
            Complete = 1,
            R = 2
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_qr(IntPtr tensor, byte mode, out IntPtr pR);

        public static (TorchTensor Q, TorchTensor R) qr(TorchTensor input, QRMode mode = QRMode.Reduced)
        {
            var Q = THSLinalg_qr(input.Handle, (byte)mode, out var R);
            if (Q == IntPtr.Zero || R == IntPtr.Zero)
                Torch.CheckForErrors();
            return (new TorchTensor(Q), new TorchTensor(R));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_solve(IntPtr tensor, IntPtr other);

        public static TorchTensor solve(TorchTensor input, TorchTensor other)
        {
            var res = THSLinalg_solve(input.Handle, other.handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_svd(IntPtr tensor, bool fullMatrices, out IntPtr pS, out IntPtr pVh);

        public static (TorchTensor U, TorchTensor S, TorchTensor Vh) svd(TorchTensor input, bool fullMatrices = true)
        {
            var U = THSLinalg_svd(input.Handle, fullMatrices, out var S, out var Vh);
            if (U == IntPtr.Zero || S == IntPtr.Zero || Vh == IntPtr.Zero)
                Torch.CheckForErrors();
            return (new TorchTensor(U), new TorchTensor(S), new TorchTensor(Vh));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_svdvals(IntPtr tensor);

        /// <summary>
        /// Computes the singular values of a matrix.
        /// </summary>
        /// <param name="input">The input matrix</param>
        /// <returns></returns>
        public static TorchTensor svdvals(TorchTensor input)
        {
            var res = THSLinalg_svdvals(input.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_tensorinv(IntPtr tensor, long ind);

        public static TorchTensor tensorinv(TorchTensor input, long ind)
        {
            var res = THSLinalg_tensorinv(input.Handle, ind);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_tensorsolve(IntPtr tensor, IntPtr other, IntPtr dim, int dim_length);

        public static TorchTensor tensorsolve(TorchTensor input, TorchTensor other, long[] dims)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSLinalg_tensorsolve(input.Handle, other.Handle, (IntPtr)pdims, dims.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_vector_norm(IntPtr tensor, IntPtr ord, IntPtr dim, int dim_length, bool keepdim);

        /// <summary>
        /// Computes a vector norm.
        /// </summary>
        /// <param name="input">Tensor, flattened by default, but this behavior can be controlled using dims.</param>
        /// <param name="ord">Order of norm. Default: 2</param>
        /// <param name="dims">Dimensions over which to compute the norm.</param>
        /// <param name="keepdim">If set to true, the reduced dimensions are retained in the result as dimensions with size one. </param>
        /// <returns></returns>
        public static TorchTensor vector_norm(TorchTensor input, double ord, long[]? dims = null, bool keepdim = false)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSLinalg_vector_norm(input.Handle, ord.ToScalar().Handle, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }
    }
}
