using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

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
        static extern IntPtr THSLinalg_eigh(IntPtr tensor, byte UPLO, out IntPtr pEigenvectors);

        public static (TorchTensor,TorchTensor) eigh(TorchTensor input, char UPLO)
        {
            var res = THSLinalg_eigh(input.Handle, (byte)UPLO, out var vectors);
            if (res == IntPtr.Zero || vectors == IntPtr.Zero)
                Torch.CheckForErrors();
            return (new TorchTensor(res), new TorchTensor(vectors));
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
        static extern IntPtr THSLinalg_norm_str(IntPtr tensor, [MarshalAs(UnmanagedType.LPStr)] string p, IntPtr dim, int dim_length, bool keepdim);
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_norm_float(IntPtr tensor, double p, IntPtr dim, int dim_length, bool keepdim);
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_norm_int(IntPtr tensor, int p, IntPtr dim, int dim_length, bool keepdim);
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSLinalg_norm_opt(IntPtr tensor, IntPtr dim, int dim_length, bool keepdim);


        public static TorchTensor norm(TorchTensor input, string ord, long[]? dims, bool keepdim)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSLinalg_norm_str(input.Handle, ord, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        public static TorchTensor norm(TorchTensor input, double ord, long[]? dims, bool keepdim)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSLinalg_norm_float(input.Handle, ord, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        public static TorchTensor norm(TorchTensor input, int ord, long[]? dims, bool keepdim)
        {
            unsafe {
                fixed (long* pdims = dims) {
                    var res = THSLinalg_norm_int(input.Handle, ord, (IntPtr)pdims, dims is null ? 0 : dims.Length, keepdim);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        public static TorchTensor norm(TorchTensor input, long[]? dims, bool keepdim)
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
    }
}
