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

        public sealed partial class Tensor
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
                return new Tensor(THSTensor_asin(handle));
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
    }
}
