// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    public static partial class torch
    {
        public static class special
        {
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
