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
        internal static extern IntPtr THSLinalg_cholesky_ex(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool check_errors, out IntPtr pInfo);

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
        internal static extern IntPtr THSTensor_geqrf(IntPtr tensor, out IntPtr tau);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_eigh(IntPtr tensor, byte UPLO, out IntPtr pEigenvectors);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_eigvals(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_eigvalsh(IntPtr tensor, byte UPLO);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_householder_product(IntPtr tensor, IntPtr tau);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_inv(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_inv_ex(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool check_errors, out IntPtr pInfo);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_lstsq_none(IntPtr tensor, IntPtr other, out IntPtr pResiduals, out IntPtr pRank, out IntPtr pSingularValues);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_lstsq_rcond(IntPtr tensor, IntPtr other, double rcond, out IntPtr pResiduals, out IntPtr pRank, out IntPtr pSingularValues);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_ldl_factor(IntPtr A, [MarshalAs(UnmanagedType.U1)] bool hermitian, out IntPtr pivots);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_ldl_factor_ex(IntPtr A, [MarshalAs(UnmanagedType.U1)] bool hermitian, [MarshalAs(UnmanagedType.U1)] bool check_errors, out IntPtr pivots, out IntPtr info);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_ldl_solve(IntPtr LD, IntPtr pivots, IntPtr B, [MarshalAs(UnmanagedType.U1)] bool hermitian);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_lu(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool pivot, out IntPtr pL, out IntPtr pU);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_lu_factor(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool pivot, out IntPtr pPivots);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_matrix_norm_fronuc(IntPtr tensor, byte fronuc, IntPtr dim, int dim_length, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_matrix_norm(IntPtr tensor, IntPtr ord, IntPtr dim, int dim_length, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_matrix_rank(IntPtr tensor, double atol, [MarshalAs(UnmanagedType.U1)] bool has_atol, double rtol, [MarshalAs(UnmanagedType.U1)] bool has_rtol, [MarshalAs(UnmanagedType.U1)] bool hermitian);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_matrix_rank_tensor(IntPtr tensor, IntPtr atol, IntPtr rtol, [MarshalAs(UnmanagedType.U1)] bool hermitian);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_dot(IntPtr tensor, int len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_multi_dot(IntPtr tensor, int len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_norm_str(IntPtr tensor, [MarshalAs(UnmanagedType.LPStr)] string p, IntPtr dim, int dim_length, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_norm_float(IntPtr tensor, double p, IntPtr dim, int dim_length, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_norm_int(IntPtr tensor, int p, IntPtr dim, int dim_length, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_norm_opt(IntPtr tensor, IntPtr dim, int dim_length, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_pinv(IntPtr tensor, double atol, [MarshalAs(UnmanagedType.U1)] bool has_atol, double rtol, [MarshalAs(UnmanagedType.U1)] bool has_rtol, [MarshalAs(UnmanagedType.U1)] bool hermitian);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_pinv_tensor(IntPtr tensor, IntPtr atol, IntPtr rtol, [MarshalAs(UnmanagedType.U1)] bool hermitian);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_qr(IntPtr tensor, byte mode, out IntPtr pR);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_solve(IntPtr tensor, IntPtr other, [MarshalAs(UnmanagedType.U1)] bool left);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_solve_ex(IntPtr tensor, IntPtr other, [MarshalAs(UnmanagedType.U1)] bool left, [MarshalAs(UnmanagedType.U1)] bool check_errors, out IntPtr infos);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_svd(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool fullMatrices, out IntPtr pS, out IntPtr pVh);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_svdvals(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_tensorinv(IntPtr tensor, long ind);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_tensorsolve(IntPtr tensor, IntPtr other, IntPtr dim, int dim_length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_vector_norm(IntPtr tensor, IntPtr ord, IntPtr dim, int dim_length, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_matrix_power(IntPtr tensor, long n);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_pinverse(IntPtr tensor, double rcond, [MarshalAs(UnmanagedType.U1)] bool hermitian);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_vander(IntPtr tensor, long N);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_vecdot(IntPtr x, IntPtr y, long dim, IntPtr output);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_lu_solve(IntPtr B, IntPtr LU, IntPtr pivots, [MarshalAs(UnmanagedType.U1)] bool left, [MarshalAs(UnmanagedType.U1)] bool adjoint, IntPtr output);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSLinalg_tensordot(IntPtr input1, IntPtr input2, IntPtr dims1, int dims1_length, IntPtr dims2, int dims2_length);
    }
}
