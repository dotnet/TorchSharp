// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class NativeMethods
    {
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_airy_ai(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_airy_ai_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_bessel_j0(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_bessel_j0_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_bessel_j1(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_bessel_j1_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_bessel_y0(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_bessel_y0_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_bessel_y1(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_bessel_y1_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_modified_bessel_i0(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_modified_bessel_i0_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_modified_bessel_i1(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_modified_bessel_i1_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_modified_bessel_k0(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_modified_bessel_k0_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_modified_bessel_k1(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_modified_bessel_k1_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_scaled_modified_bessel_k0(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_scaled_modified_bessel_k0_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_scaled_modified_bessel_k1(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_scaled_modified_bessel_k1_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_spherical_bessel_j0(IntPtr tensor);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_spherical_bessel_j0_out(IntPtr tensor, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_chebyshev_polynomial_t(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_chebyshev_polynomial_t_out(IntPtr x, IntPtr n, IntPtr @out);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_chebyshev_polynomial_u(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_chebyshev_polynomial_u_out(IntPtr x, IntPtr n, IntPtr @out);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_chebyshev_polynomial_v(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_chebyshev_polynomial_v_out(IntPtr x, IntPtr n, IntPtr @out);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_chebyshev_polynomial_w(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_chebyshev_polynomial_w_out(IntPtr x, IntPtr n, IntPtr @out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_shifted_chebyshev_polynomial_t(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_shifted_chebyshev_polynomial_t_out(IntPtr x, IntPtr n, IntPtr @out);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_shifted_chebyshev_polynomial_u(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_shifted_chebyshev_polynomial_u_out(IntPtr x, IntPtr n, IntPtr @out);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_shifted_chebyshev_polynomial_v(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_shifted_chebyshev_polynomial_v_out(IntPtr x, IntPtr n, IntPtr @out);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_shifted_chebyshev_polynomial_w(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_shifted_chebyshev_polynomial_w_out(IntPtr x, IntPtr n, IntPtr @out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_hermite_polynomial_h(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_hermite_polynomial_h_out(IntPtr x, IntPtr n, IntPtr @out);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_hermite_polynomial_he(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_hermite_polynomial_he_out(IntPtr x, IntPtr n, IntPtr @out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_laguerre_polynomial_l(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_laguerre_polynomial_l_out(IntPtr x, IntPtr n, IntPtr @out);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_legendre_polynomial_p(IntPtr x, IntPtr n);
        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_legendre_polynomial_p_out(IntPtr x, IntPtr n, IntPtr @out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_entr(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_erf(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_erfc(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_erfcx(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_erfinv(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_expit(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_expm1(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_exp2(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_gammaln(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_gammainc(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_gammaincc(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_polygamma(long n, IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_multigammaln(IntPtr tensor, long p);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_digamma(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_i0(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_i0e(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_i1(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_i1e(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_logit(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_log_softmax(IntPtr tensor, long dim, sbyte scalar_type);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_softmax(IntPtr tensor, long dim, sbyte scalar_type);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_ndtr(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_ndtri(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_sinc(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_xlog1py(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSSpecial_zeta(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern double THSSpecial_erf_scalar(double x);

        [DllImport("LibTorchSharp")]
        internal static extern double THSSpecial_erfc_scalar(double x);
    }
}
