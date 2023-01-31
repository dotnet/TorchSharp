// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics.Contracts;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    public static partial class torch
    {
        // This files contains trigonometric and hyperbolic functions.
        // All are element-wise operations on tensors.

        // Most operations are duplicated -- in the 'torch' namespace, and as methods on 'Tensor.'
        // This is done in order to mimic the Pytorch experience.
        public partial class Tensor
        {
            /// <summary>
            /// Computes the element-wise angle (in radians) of the given input tensor.
            /// </summary>
            /// <returns></returns>
            /// <remarks>
            /// Starting in Torch 1.8, angle returns pi for negative real numbers, zero for non-negative real numbers, and propagates NaNs.
            /// Previously the function would return zero for all real numbers and not propagate floating-point NaNs.
            /// </remarks>
            public Tensor angle()
            {
                var res = THSTensor_angle(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor asin()
            {
                var res = THSTensor_asin(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arcsin() => asin();

            /// <summary>
            /// Computes the arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor asin_()
            {
                var res = THSTensor_asin_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor arcsin_() => asin_();

            /// <summary>
            /// Computes the arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor acos()
            {
                var res = THSTensor_acos(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arccos() => acos();

            /// <summary>
            /// Computes the arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor acos_()
            {
                var res = THSTensor_acos_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arccos_() => acos_();

            /// <summary>
            /// Computes the arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor atan()
            {
                var res = THSTensor_atan(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            [Pure]public Tensor arctan() => atan();

            [Pure]public Tensor arctan(Tensor other) => atan2(other);
            public Tensor arctan_(Tensor other) => atan2_(other);

            /// <summary>
            /// Computes the arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor atan_()
            {
                var res = THSTensor_atan_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arctan_() => atan_();

            /// <summary>
            /// Element-wise arctangent of input / other with consideration of the quadrant.
            /// </summary>
            /// <param name="other">The second tensor</param>
            public Tensor atan2(Tensor other)
            {
                var res = THSTensor_atan2(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            public Tensor arctan2_(Tensor other) => atan2_(other);
            public Tensor arctan2(Tensor other) => atan2(other);

            /// <summary>
            /// Element-wise arctangent of input / other with consideration of the quadrant.
            /// </summary>
            /// <param name="other">The second tensor</param>
            /// <returns></returns>
            public Tensor atan2_(Tensor other)
            {
                var res = THSTensor_atan2_(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cos()
            {
                var res = THSTensor_cos(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cos_()
            {
                var res = THSTensor_cos_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sin()
            {
                var res = THSTensor_sin(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sin_()
            {
                var res = THSTensor_sin_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tan()
            {
                var res = THSTensor_tan(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tan_()
            {
                var res = THSTensor_tan_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the normalized sinc of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sinc()
            {
                var res = THSTensor_sinc(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the normalized sinc of input, in place.
            /// </summary>
            /// <returns></returns>
            public Tensor sinc_()
            {
                var res = THSTensor_sinc_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sinh()
            {
                var res = THSTensor_sinh(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sinh_()
            {
                var res = THSTensor_sinh_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cosh()
            {
                var res = THSTensor_cosh(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cosh_()
            {
                var res = THSTensor_cosh_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tanh()
            {
                var res = THSTensor_tanh(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tanh_()
            {
                var res = THSTensor_tanh_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arcsinh()
            {
                var res = THSTensor_arcsinh(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor asinh() => arcsinh();

            /// <summary>
            /// Computes the hyperbolic arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arcsinh_()
            {
                var res = THSTensor_arcsinh_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor asinh_() => arcsinh_();

            /// <summary>
            /// Computes the hyperbolic arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arccosh()
            {
                var res = THSTensor_arccosh(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor acosh() => arccosh();

            /// <summary>
            /// Computes the hyperbolic arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arccosh_()
            {
                var res = THSTensor_arccosh_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor acosh_() => arccosh_();

            /// <summary>
            /// Computes the hyperbolic arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arctanh()
            {
                var res = THSTensor_arctanh(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor atanh() => arctanh();

            /// <summary>
            /// Computes the hyperbolic arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arctanh_()
            {
                var res = THSTensor_arctanh_(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor atanh_() => arctanh_();
        }
    }
}
