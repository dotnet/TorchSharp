// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
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
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_angle(IntPtr tensor);

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
                var res = THSTensor_angle(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_asin(IntPtr tensor);

            /// <summary>
            /// Computes the arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor asin()
            {
                var res = THSTensor_asin(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arcsin() => asin();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_asin_(IntPtr tensor);

            /// <summary>
            /// Computes the arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor asin_()
            {
                var res = THSTensor_asin_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            public Tensor arcsin_() => asin_();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_acos(IntPtr tensor);

            /// <summary>
            /// Computes the arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor acos()
            {
                var res = THSTensor_acos(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arccos() => acos();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_acos_(IntPtr tensor);

            /// <summary>
            /// Computes the arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor acos_()
            {
                var res = THSTensor_acos_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arccos_() => acos_();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_atan(IntPtr tensor);

            /// <summary>
            /// Computes the arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor atan()
            {
                var res = THSTensor_atan(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arctan() => atan();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_atan_(IntPtr tensor);

            /// <summary>
            /// Computes the arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor atan_()
            {
                var res = THSTensor_atan_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arctan_() => atan_();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_atan2(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise arctangent of input / other with consideration of the quadrant.
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor atan2(Tensor other)
            {
                var res = THSTensor_atan2(handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_atan2_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise arctangent of input / other with consideration of the quadrant.
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor atan2_(Tensor other)
            {
                var res = THSTensor_atan2_(handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cos(IntPtr tensor);

            /// <summary>
            /// Computes the cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cos()
            {
                var res = THSTensor_cos(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cos_(IntPtr tensor);

            /// <summary>
            /// Computes the cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cos_()
            {
                var res = THSTensor_cos_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sin(IntPtr tensor);

            /// <summary>
            /// Computes the sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sin()
            {
                var res = THSTensor_sin(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sin_(IntPtr tensor);

            /// <summary>
            /// Computes the sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sin_()
            {
                var res = THSTensor_sin_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_tan(IntPtr tensor);

            /// <summary>
            /// Computes the tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tan()
            {
                var res = THSTensor_tan(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_tan_(IntPtr tensor);

            /// <summary>
            /// Computes the tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tan_()
            {
                var res = THSTensor_tan_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sinc(IntPtr tensor);

            /// <summary>
            /// Computes the normalized sinc of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sinc()
            {
                var res = THSTensor_sinc(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sinc_(IntPtr tensor);

            /// <summary>
            /// Computes the normalized sinc of input, in place.
            /// </summary>
            /// <returns></returns>
            public Tensor sinc_()
            {
                var res = THSTensor_sinc_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sinh(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sinh()
            {
                var res = THSTensor_sinh(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sinh_(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic sine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sinh_()
            {
                var res = THSTensor_sinh_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cosh(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cosh()
            {
                var res = THSTensor_cosh(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cosh_(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic cosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor cosh_()
            {
                var res = THSTensor_cosh_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_tanh(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tanh()
            {
                var res = THSTensor_tanh(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_tanh_(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic tangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor tanh_()
            {
                var res = THSTensor_tanh_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_arcsinh(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arcsinh()
            {
                var res = THSTensor_arcsinh(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor asinh() => arcsinh();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_arcsinh_(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arcsinh_()
            {
                var res = THSTensor_arcsinh_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arcsine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor asinh_() => arcsinh_();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_arccosh(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arccosh()
            {
                var res = THSTensor_arccosh(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor acosh() => arccosh();


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_arccosh_(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arccosh_()
            {
                var res = THSTensor_arccosh_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arccosine of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor acosh_() => arccosh_();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_arctanh(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arctanh()
            {
                var res = THSTensor_arctanh(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor atanh() => arctanh();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_arctanh_(IntPtr tensor);

            /// <summary>
            /// Computes the hyperbolic arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor arctanh_()
            {
                var res = THSTensor_arctanh_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the hyperbolic arctangent of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor atanh_() => arctanh_();
        }

        // Duplicated methods in the 'torch' namespace:

        /// <summary>
        /// Computes the element-wise angle (in radians) of the given input tensor.
        /// </summary>
        /// <returns></returns>
        /// <remarks>
        /// Starting in Torch 1.8, angle returns pi for negative real numbers, zero for non-negative real numbers, and propagates NaNs.
        /// Previously the function would return zero for all real numbers and not propagate floating-point NaNs.
        /// </remarks>
        public static Tensor angle(Tensor input)
        {
            return input.angle();
        }

        /// <summary>
        /// Computes the arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor asin(Tensor input)
        {
            return input.asin();
        }

        /// <summary>
        /// Computes the arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arcsin(Tensor input) => asin(input);

        /// <summary>
        /// Computes the arcsine of the elements of input, in place.
        /// </summary>
        /// <returns></returns>
        public static Tensor asin_(Tensor input)
        {
            return input.asinh_();
        }

        public static Tensor arcsin_(Tensor input) => asin_(input);

        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor acos(Tensor input)
        {
            return input.acos();
        }

        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arccos(Tensor input) => acos(input);

        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor acos_(Tensor input)
        {
            return input.acos_();
        }

        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arccos_(Tensor input) => acos_(input);

        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor atan(Tensor input)
        {
            return input.atan();
        }

        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arctan(Tensor input) => atan(input);

        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor atan_(Tensor input)
        {
            return input.atan_();
        }

        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arctan_(Tensor input) => atan_(input);



        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <returns></returns>
        public static Tensor atan2(Tensor input, Tensor other)
        {
            return input.atan2(other);
        }

        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <returns></returns>
        public static Tensor arctan(Tensor input, Tensor other) => atan2(input, other);

        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <returns></returns>
        public static Tensor atan2_(Tensor input, Tensor other)
        {
            return input.atan2_(other);
        }

        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <returns></returns>
        public static Tensor arctan2_(Tensor input, Tensor other) => atan2_(input, other);



        /// <summary>
        /// Computes the cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor cos(Tensor input)
        {
            return input.cos();
        }

        /// <summary>
        /// Computes the cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor cos_(Tensor input)
        {
            return input.cos_();
        }

        /// <summary>
        /// Computes the sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor sin(Tensor input)
        {
            return input.sin();
        }

        /// <summary>
        /// Computes the sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor sin_(Tensor input)
        {
            return input.sin_();
        }

        /// <summary>
        /// Computes the tangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor tan(Tensor input)
        {
            return input.tan();
        }

        /// <summary>
        /// Computes the tangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor tan_(Tensor input)
        {
            return input.tan_();
        }

        /// <summary>
        /// Computes the normalized sinc of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor sinc(Tensor input)
        {
            return input.sinc();
        }

        /// <summary>
        /// Computes the normalized sinc of input, in place.
        /// </summary>
        /// <returns></returns>
        public static Tensor sinc_(Tensor input)
        {
            return input.sinc_();
        }

        /// <summary>
        /// Computes the hyperbolic sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor sinh(Tensor input)
        {
            return input.sinh();
        }

        /// <summary>
        /// Computes the hyperbolic sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor sinh_(Tensor input)
        {
            return input.sinh_();
        }

        /// <summary>
        /// Computes the hyperbolic cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor cosh(Tensor input)
        {
            return input.cosh();
        }

        /// <summary>
        /// Computes the hyperbolic cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor cosh_(Tensor input)
        {
            return input.cosh_();
        }

        /// <summary>
        /// Computes the hyperbolic tangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor tanh(Tensor input)
        {
            return input.tanh();
        }

        /// <summary>
        /// Computes the hyperbolic tangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor tanh_(Tensor input)
        {
            return input.tanh_();
        }

        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arcsinh(Tensor input)
        {
            return input.arcsinh();
        }

        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor asinh(Tensor input) => arcsinh(input);

        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arcsinh_(Tensor input)
        {
            return input.arcsinh_();
        }

        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor asinh_(Tensor input) => arcsinh_(input);

        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arccosh(Tensor input)
        {
            return input.arccosh();
        }

        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor acosh(Tensor input) => arccosh(input);


        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arccosh_(Tensor input)
        {
            return input.arccosh_();
        }

        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor acosh_(Tensor input) => arccosh_(input);

        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arctanh(Tensor input)
        {
            return input.arctanh();
        }

        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor atanh(Tensor input) => arctanh(input);

        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arctanh_(Tensor input)
        {
            return input.arctanh_();
        }

        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor atanh_(Tensor input) => arctanh_(input);
    }
}
