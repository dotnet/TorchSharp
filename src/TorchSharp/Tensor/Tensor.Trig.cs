// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics.Contracts;
using System.Linq;
using TorchSharp.Amp;
using static TorchSharp.PInvoke.NativeMethods;

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
                return ReturnCheckForErrors(THSTensor_angle(Handle));
            }

            /// <summary>
            /// Computes the arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor asin()
            {
                return ReturnCheckForErrorsAutocast(THSTensor_asin(Handle), ScalarType.Float32);
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
                THSTensor_asin_(Handle);
                CheckForErrors();
                return this;
            }

            public Tensor arcsin_() => asin_();

            /// <summary>
            /// Computes the arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor acos()
            {
                return ReturnCheckForErrorsAutocast(THSTensor_acos(Handle), ScalarType.Float32);
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
                THSTensor_acos_(Handle);
                CheckForErrors();
                return this;
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
                return ReturnCheckForErrors(THSTensor_atan(Handle));
            }

            /// <summary>
            /// Computes the arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            [Pure] public Tensor arctan() => atan();

            [Pure] public Tensor arctan(Tensor other) => atan2(other);
            public Tensor arctan_(Tensor other) => atan2_(other);

            /// <summary>
            /// Computes the arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor atan_()
            {
                THSTensor_atan_(Handle);
                CheckForErrors();
                return this;
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
                if (AutocastMode.IsAutocastEnabled()) {
                    var sts = new[] { this.dtype, other.dtype };
                    if (sts.All(x => x == ScalarType.Float16))
                        (handle, other.handle) = AutocastMode.AutoCast(handle, other.handle, ScalarType.Float16);
                    if (sts.Any(x => x == ScalarType.Float32))
                        (handle, other.handle) = AutocastMode.AutoCast(handle, other.handle, ScalarType.Float32);
                }

                return ReturnCheckForErrors(THSTensor_atan2(Handle, other.Handle));
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
                THSTensor_atan2_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cos()
            {
                return ReturnCheckForErrors(THSTensor_cos(Handle));
            }

            /// <summary>
            /// Computes the cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cos_()
            {
                THSTensor_cos_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sin()
            {
                return ReturnCheckForErrors(THSTensor_sin(Handle));
            }

            /// <summary>
            /// Computes the sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sin_()
            {
                THSTensor_sin_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tan()
            {
                return ReturnCheckForErrorsAutocast(THSTensor_tan(Handle), ScalarType.Float32);
            }

            /// <summary>
            /// Computes the tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tan_()
            {
                THSTensor_tan_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the normalized sinc of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sinc()
            {
                return ReturnCheckForErrors(THSTensor_sinc(Handle));
            }

            /// <summary>
            /// Computes the normalized sinc of input, in place.
            /// </summary>
            /// <returns></returns>
            public Tensor sinc_()
            {
                THSTensor_sinc_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the hyperbolic sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sinh()
            {
                return ReturnCheckForErrorsAutocast(THSTensor_sinh(Handle), ScalarType.Float32);
            }

            /// <summary>
            /// Computes the hyperbolic sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sinh_()
            {
                THSTensor_sinh_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the hyperbolic cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cosh()
            {
                return ReturnCheckForErrorsAutocast(THSTensor_cosh(Handle), ScalarType.Float32);
            }

            /// <summary>
            /// Computes the hyperbolic cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cosh_()
            {
                THSTensor_cosh_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the hyperbolic tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tanh()
            {
                return ReturnCheckForErrors(THSTensor_tanh(Handle));
            }

            /// <summary>
            /// Computes the hyperbolic tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tanh_()
            {
                THSTensor_tanh_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the hyperbolic arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arcsinh()
            {
                return ReturnCheckForErrors(THSTensor_arcsinh(Handle));
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
                THSTensor_arcsinh_(Handle);
                CheckForErrors();
                return this;
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
                return ReturnCheckForErrors(THSTensor_arccosh(Handle));
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
                THSTensor_arccosh_(Handle);
                CheckForErrors();
                return this;
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
                return ReturnCheckForErrors(THSTensor_arctanh(Handle));
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
                THSTensor_arctanh_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the hyperbolic arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor atanh_() => arctanh_();
        }
    }
}
