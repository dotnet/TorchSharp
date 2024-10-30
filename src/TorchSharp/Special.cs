// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    public static partial class torch
    {
        public static class special
        {
            /// <summary>
            /// Airy function.
            /// </summary>
            /// <param name="input">Input tensor</param>
            /// <param name="out">Optional output tensor, will be modified if present.</param>
            /// <returns></returns>
            public static Tensor airy_ai(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_airy_ai(input.Handle) :
                    THSSpecial_airy_ai_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Bessel function of the first kind of order 0.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="out">An optional output tensor, which will be modified if present.</param>
            public static Tensor bessel_j0(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_bessel_j0(input.Handle) :
                    THSSpecial_bessel_j0_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Bessel function of the first kind of order 1.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="out">An optional output tensor, which will be modified if present.</param>
            public static Tensor bessel_j1(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_bessel_j1(input.Handle) :
                    THSSpecial_bessel_j1_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Bessel function of the second kind of order 1.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="out">An optional output tensor, which will be modified if present.</param>
            public static Tensor bessel_y0(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_bessel_y0(input.Handle) :
                    THSSpecial_bessel_y0_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Bessel function of the second kind of order 1.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="out">An optional output tensor, which will be modified if present.</param>
            public static Tensor bessel_y1(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_bessel_y1(input.Handle) :
                    THSSpecial_bessel_y1_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Modified Bessel function of the first kind of order 0.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="out">An optional output tensor, which will be modified if present.</param>
            public static Tensor modified_bessel_i0(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_modified_bessel_i0(input.Handle) :
                    THSSpecial_modified_bessel_i0_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Modified Bessel function of the first kind of order 1.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="out">An optional output tensor, which will be modified if present.</param>
            public static Tensor modified_bessel_i1(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_modified_bessel_i1(input.Handle) :
                    THSSpecial_modified_bessel_i1_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Modified Bessel function of the second kind of order 1.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="out">An optional output tensor, which will be modified if present.</param>
            public static Tensor modified_bessel_k0(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_modified_bessel_k0(input.Handle) :
                    THSSpecial_modified_bessel_k0_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Modified Bessel function of the second kind of order 1.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="out">An optional output tensor, which will be modified if present.</param>
            public static Tensor modified_bessel_k1(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_modified_bessel_k1(input.Handle) :
                    THSSpecial_modified_bessel_k1_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Scaled modified Bessel function of the second kind of order 0
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="out">An optional output tensor, which will be modified if present.</param>
            public static Tensor scaled_modified_bessel_k0(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_scaled_modified_bessel_k0(input.Handle) :
                    THSSpecial_scaled_modified_bessel_k0_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Scaled modified Bessel function of the second kind of order 1
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="out">An optional output tensor, which will be modified if present.</param>
            public static Tensor scaled_modified_bessel_k1(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_scaled_modified_bessel_k1(input.Handle) :
                    THSSpecial_scaled_modified_bessel_k1_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Bessel function of the first kind of order 0.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="out">An optional output tensor, which will be modified if present.</param>
            public static Tensor spherical_bessel_j0(Tensor input, Tensor? @out = null)
            {
                var res = @out is null ?
                    THSSpecial_spherical_bessel_j0(input.Handle) :
                    THSSpecial_spherical_bessel_j0_out(input.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the Chebyshev polynomial of the first kind.
            ///
            /// See: https://en.wikipedia.org/wiki/Chebyshev_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            public static Tensor chebyshev_polynomial_t(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_chebyshev_polynomial_t(x.Handle, n.Handle) :
                    THSSpecial_chebyshev_polynomial_t_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }


            /// <summary>
            /// Computes the Chebyshev polynomial of the second kind.
            ///
            /// See: https://en.wikipedia.org/wiki/Chebyshev_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            public static Tensor chebyshev_polynomial_u(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_chebyshev_polynomial_u(x.Handle, n.Handle) :
                    THSSpecial_chebyshev_polynomial_u_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the Chebyshev polynomial of the third kind.
            ///
            /// See: https://en.wikipedia.org/wiki/Chebyshev_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            public static Tensor chebyshev_polynomial_v(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_chebyshev_polynomial_v(x.Handle, n.Handle) :
                    THSSpecial_chebyshev_polynomial_v_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the Chebyshev polynomial of the fourth kind.
            ///
            /// See: https://en.wikipedia.org/wiki/Chebyshev_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            public static Tensor chebyshev_polynomial_w(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_chebyshev_polynomial_w(x.Handle, n.Handle) :
                    THSSpecial_chebyshev_polynomial_w_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the Chebyshev polynomial of the first kind.
            ///
            /// See: https://en.wikipedia.org/wiki/Chebyshev_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            public static Tensor shifted_chebyshev_polynomial_t(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_shifted_chebyshev_polynomial_t(x.Handle, n.Handle) :
                    THSSpecial_shifted_chebyshev_polynomial_t_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }


            /// <summary>
            /// Computes the Chebyshev polynomial of the second kind.
            ///
            /// See: https://en.wikipedia.org/wiki/Chebyshev_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            public static Tensor shifted_chebyshev_polynomial_u(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_shifted_chebyshev_polynomial_u(x.Handle, n.Handle) :
                    THSSpecial_shifted_chebyshev_polynomial_u_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the Chebyshev polynomial of the third kind.
            ///
            /// See: https://en.wikipedia.org/wiki/Chebyshev_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            public static Tensor shifted_chebyshev_polynomial_v(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_shifted_chebyshev_polynomial_v(x.Handle, n.Handle) :
                    THSSpecial_shifted_chebyshev_polynomial_v_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the Chebyshev polynomial of the fourth kind.
            ///
            /// See: https://en.wikipedia.org/wiki/Chebyshev_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            public static Tensor shifted_chebyshev_polynomial_w(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_shifted_chebyshev_polynomial_w(x.Handle, n.Handle) :
                    THSSpecial_shifted_chebyshev_polynomial_w_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Probabilist's Hermite polynomials.
            ///
            /// See: https://en.wikipedia.org/wiki/Hermite_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            public static Tensor hermite_polynomial_h(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_hermite_polynomial_h(x.Handle, n.Handle) :
                    THSSpecial_hermite_polynomial_h_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Physicists's Hermite polynomials.
            ///
            /// See: https://en.wikipedia.org/wiki/Hermite_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            public static Tensor hermite_polynomial_he(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_hermite_polynomial_he(x.Handle, n.Handle) :
                    THSSpecial_hermite_polynomial_he_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Laguerre polynomials
            ///
            /// See: https://en.wikipedia.org/wiki/Laguerre_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            /// <returns></returns>
            public static Tensor laguerre_polynomial_l(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_laguerre_polynomial_l(x.Handle, n.Handle) :
                    THSSpecial_laguerre_polynomial_l_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Legendre polynomials
            ///
            /// https://en.wikipedia.org/wiki/Legendre_polynomials
            /// </summary>
            /// <param name="x">The input tensor.</param>
            /// <param name="n">n</param>
            /// <param name="out">An optional output tensor.</param>
            public static Tensor legendre_polynomial_p(Tensor x, Tensor n, Tensor? @out =null)
            {
                var res = @out is null ?
                    THSSpecial_legendre_polynomial_p(x.Handle, n.Handle) :
                    THSSpecial_legendre_polynomial_p_out(x.Handle, n.Handle, @out.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the entropy on input, elementwise.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor entr(Tensor input)
            {
                var res = THSSpecial_entr(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the error function of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor erf(Tensor input)
            {
                var res = THSSpecial_erf(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the complementary error function of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor erfc(Tensor input)
            {
                var res = THSSpecial_erfc(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the scaled complementary error function of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor erfcx(Tensor input)
            {
                var res = THSSpecial_erfc(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the inverse error function of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor erfinv(Tensor input)
            {
                var res = THSSpecial_erfinv(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the expit (also known as the logistic sigmoid function) of the elements of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor expit(Tensor input)
            {
                var res = THSSpecial_expit(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the exponential of the elements minus 1 of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor expm1(Tensor input)
            {
                var res = THSSpecial_expm1(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the base two exponential function of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor exp2(Tensor input)
            {
                var res = THSSpecial_exp2(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the natural logarithm of the absolute value of the gamma function on input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor gammaln(Tensor input)
            {
                var res = THSSpecial_gammaln(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the regularized lower incomplete gamma function.
            /// </summary>
            /// <param name="input">The first non-negative input tensor</param>
            /// <param name="other">The second non-negative input tensor</param>
            /// <returns></returns>
            public static Tensor gammainc(Tensor input, Tensor other)
            {
                var res = THSSpecial_gammainc(input.Handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the regularized upper incomplete gamma function.
            /// </summary>
            /// <param name="input">The first non-negative input tensor</param>
            /// <param name="other">The second non-negative input tensor</param>
            /// <returns></returns>
            public static Tensor gammaincc(Tensor input, Tensor other)
            {
                var res = THSSpecial_gammaincc(input.Handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the n-th derivative of the digamma function on input.
            /// </summary>
            /// <param name="n">The order of the polygamma function</param>
            /// <param name="input">The second non-negative input tensor</param>
            /// <returns></returns>
            public static Tensor polygamma(long n, Tensor input)
            {
                var res = THSSpecial_polygamma(n, input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the multivariate log-gamma function with dimension pp element-wise.
            /// </summary>
            /// <param name="input">The tensor to compute the multivariate log-gamma function</param>
            /// <param name="p">The number of dimensions</param>
            /// <returns></returns>
            public static Tensor multigammaln(Tensor input, long p)
            {
                var res = THSSpecial_multigammaln(input.Handle, p);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the logarithmic derivative of the gamma function on input.
            /// </summary>
            /// <param name="input">The second non-negative input tensor</param>
            /// <returns></returns>
            public static Tensor digamma(Tensor input)
            {
                var res = THSSpecial_digamma(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the logarithmic derivative of the gamma function on input.
            /// </summary>
            /// <param name="input">The second non-negative input tensor</param>
            public static Tensor psi(Tensor input) => digamma(input);

            /// <summary>
            /// Computes the zeroth order modified Bessel function of the first kind for each element of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor i0(Tensor input)
            {
                var res = THSSpecial_i0(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the exponentially scaled zeroth order modified Bessel function of the first kind (as defined below) for each element of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor i0e(Tensor input)
            {
                var res = THSSpecial_i0e(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the first order modified Bessel function of the first kind for each element of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor i1(Tensor input)
            {
                var res = THSSpecial_i1(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the exponentially scaled first order modified Bessel function of the first kind (as defined below) for each element of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor i1e(Tensor input)
            {
                var res = THSSpecial_i1e(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with the logit of the elements of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor logit(Tensor input)
            {
                var res = THSSpecial_logit(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with the logit of the elements of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="dim">A dimension along which log_softmax will be computed.</param>
            /// <param name="dtype">The desired data type of returned tensor.</param>
            /// <returns></returns>
            public static Tensor log_softmax(Tensor input, long dim, ScalarType? dtype = null)
            {
                var dt = dtype.HasValue ? dtype.Value : input.dtype;
                var res = THSSpecial_log_softmax(input.Handle, dim, (sbyte)dt);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the area under the standard Gaussian probability density function, integrated from minus infinity to input, elementwise.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor ndtr(Tensor input)
            {
                var res = THSSpecial_ndtr(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the argument, x, for which the area under the Gaussian probability density function (integrated from minus infinity to x) is equal to input, elementwise.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor ndtri(Tensor input)
            {
                var res = THSSpecial_ndtri(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the normalized sinc of input.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <returns></returns>
            public static Tensor sinc(Tensor input)
            {
                var res = THSSpecial_sinc(input.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Alias for torch.special.expit()
            /// </summary>
            /// <returns></returns>
            public static Tensor sigmoid(Tensor input) => input.sigmoid();

            /// <summary>
            /// Alias for torch.special.expit()
            /// </summary>
            /// <returns></returns>
            public static Tensor sigmoid_(Tensor input) => input.sigmoid_();

            /// <summary>
            /// Computes the softmax function for the input tensor.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="dim">A dimension along which softmax will be computed.</param>
            /// <param name="dtype">The desired data type of returned tensor.</param>
            /// <returns></returns>
            public static Tensor softmax(Tensor input, long dim, ScalarType? dtype = null)
            {
                var dt = dtype.HasValue ? dtype.Value : input.dtype;
                var res = THSSpecial_softmax(input.Handle, dim, (sbyte)dt);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes input * log1p(other). Similar to SciPy’s scipy.special.xlog1py.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="other"></param>
            /// <returns></returns>
            public static Tensor xlog1py(Tensor input, Tensor other)
            {
                var res = THSSpecial_xlog1py(input.Handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the Hurwitz zeta function, elementwise.
            /// </summary>
            /// <param name="x">The input tensor corresponding to x</param>
            /// <param name="q">The input tensor corresponding to q</param>
            /// <remarks>The Riemann zeta function corresponds to the case when q = 1.</remarks>
            public static Tensor zeta(Tensor x, Tensor q)
            {
                var res = THSSpecial_zeta(x.Handle, q.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }
        }
    }
}
