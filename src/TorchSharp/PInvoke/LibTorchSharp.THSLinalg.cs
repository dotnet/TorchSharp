// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp.PInvoke
{
    internal static partial class LibTorchSharp
    {
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_cholesky(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_cholesky_ex(IntPtr tensor, bool check_errors, out IntPtr pInfo);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_cond_int(IntPtr tensor, int p);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_cond_float(IntPtr tensor, double p);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_cond_str(IntPtr tensor, [MarshalAs(UnmanagedType.LPStr)] string p);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_cond_none(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_cross(IntPtr input, IntPtr other, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_det(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_slogdet(IntPtr tensor, out IntPtr pLogabsdet);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_eig(IntPtr tensor, out IntPtr pEigenvectors);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_eigh(IntPtr tensor, byte UPLO, out IntPtr pEigenvectors);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_eigvals(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_eigvalsh(IntPtr tensor, byte UPLO);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_inv(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_inv_ex(IntPtr tensor, bool check_errors, out IntPtr pInfo);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_lstsq_none(IntPtr tensor, IntPtr other, out IntPtr pResiduals, out IntPtr pRank, out IntPtr pSingularValues);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_lstsq_rcond(IntPtr tensor, IntPtr other, double rcond, out IntPtr pResiduals, out IntPtr pRank, out IntPtr pSingularValues);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_lu_factor(IntPtr tensor, bool pivot, out IntPtr pPivots);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_matrix_norm_fronuc(IntPtr tensor, byte fronuc, IntPtr dim, int dim_length, bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_matrix_norm(IntPtr tensor, IntPtr ord, IntPtr dim, int dim_length, bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_matrix_rank(IntPtr tensor, double atol, bool has_atol, double rtol, bool has_rtol, bool hermitian);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_matrix_rank_tensor(IntPtr tensor, IntPtr atol, IntPtr rtol, bool hermitian);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_multi_dot(IntPtr tensor, int len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_norm_str(IntPtr tensor, [MarshalAs(UnmanagedType.LPStr)] string p, IntPtr dim, int dim_length, bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_norm_float(IntPtr tensor, double p, IntPtr dim, int dim_length, bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_norm_int(IntPtr tensor, int p, IntPtr dim, int dim_length, bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_norm_opt(IntPtr tensor, IntPtr dim, int dim_length, bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_pinv(IntPtr tensor, double atol, bool has_atol, double rtol, bool has_rtol, bool hermitian);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_pinv_tensor(IntPtr tensor, IntPtr atol, IntPtr rtol, bool hermitian);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_qr(IntPtr tensor, byte mode, out IntPtr pR);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_solve(IntPtr tensor, IntPtr other, bool left);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_solve_ex(IntPtr tensor, IntPtr other, bool left, bool check_errors, out IntPtr infos);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_svd(IntPtr tensor, bool fullMatrices, out IntPtr pS, out IntPtr pVh);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_svdvals(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_tensorinv(IntPtr tensor, long ind);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_tensorsolve(IntPtr tensor, IntPtr other, IntPtr dim, int dim_length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_vector_norm(IntPtr tensor, IntPtr ord, IntPtr dim, int dim_length, bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_matrix_power(IntPtr tensor, long n);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_pinverse(IntPtr tensor, double rcond, bool hermitian);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_vander(IntPtr tensor, long N);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_vecdot(IntPtr x, IntPtr y, long dim, IntPtr output);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_lu_solve(IntPtr B, IntPtr LU, IntPtr pivots, bool left, bool adjoint, IntPtr output);

    }
}
