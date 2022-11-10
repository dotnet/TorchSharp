// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TorchSharp.PInvoke
{
    internal static partial class LibTorchSharp
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
