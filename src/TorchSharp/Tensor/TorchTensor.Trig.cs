using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
namespace TorchSharp.Tensor
{
    // This files contains trigonometric and hyperbolic functions.
    // All are element-wise operations on tensors.

    public sealed partial class TorchTensor
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
        public TorchTensor angle()
        {
            return new TorchTensor(THSTensor_angle(handle));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_asin(IntPtr tensor);

        /// <summary>
        /// Computes the arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor asin()
        {
            return new TorchTensor(THSTensor_asin(handle));
        }

        /// <summary>
        /// Computes the arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor arcsin() => asin();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_asin_(IntPtr tensor);

        /// <summary>
        /// Computes the arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor asin_()
        {
            var res = THSTensor_asin_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        public TorchTensor arcsin_() => asin_();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_acos(IntPtr tensor);

        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor acos()
        {
            var res = THSTensor_acos(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor arccos() => acos();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_acos_(IntPtr tensor);

        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor acos_()
        {
            var res = THSTensor_acos_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor arccos_() => acos_();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan(IntPtr tensor);

        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor atan()
        {
            var res = THSTensor_atan(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor arctan() => atan();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan_(IntPtr tensor);

        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor atan_()
        {
            var res = THSTensor_atan_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor arctan_() => atan_();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan2(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor atan2(TorchTensor other)
        {
            var res = THSTensor_atan2(handle, other.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan2_(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor atan2_(TorchTensor other)
        {
            var res = THSTensor_atan2_(handle, other.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cos(IntPtr tensor);

        /// <summary>
        /// Computes the cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor cos()
        {
            var res = THSTensor_cos(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cos_(IntPtr tensor);

        /// <summary>
        /// Computes the cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor cos_()
        {
            var res = THSTensor_cos_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sin(IntPtr tensor);

        /// <summary>
        /// Computes the sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor sin()
        {
            var res = THSTensor_sin(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sin_(IntPtr tensor);

        /// <summary>
        /// Computes the sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor sin_()
        {
            var res = THSTensor_sin_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_tan(IntPtr tensor);

        /// <summary>
        /// Computes the tangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor tan()
        {
            var res = THSTensor_tan(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_tan_(IntPtr tensor);

        /// <summary>
        /// Computes the tangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor tan_()
        {
            var res = THSTensor_tan_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sinc(IntPtr tensor);

        /// <summary>
        /// Computes the normalized sinc of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor sinc()
        {
            var res = THSTensor_sinc(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sinc_(IntPtr tensor);

        /// <summary>
        /// Computes the normalized sinc of input, in place.
        /// </summary>
        /// <returns></returns>
        public TorchTensor sinc_()
        {
            var res = THSTensor_sinc_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sinh(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor sinh()
        {
            var res = THSTensor_sinh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sinh_(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor sinh_()
        {
            var res = THSTensor_sinh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cosh(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor cosh()
        {
            var res = THSTensor_cosh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cosh_(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor cosh_()
        {
            var res = THSTensor_cosh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_tanh(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic tangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor tanh()
        {
            var res = THSTensor_tanh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_tanh_(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic tangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor tanh_()
        {
            var res = THSTensor_tanh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arcsinh(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor arcsinh()
        {
            var res = THSTensor_arcsinh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor asinh() => arcsinh();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arcsinh_(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor arcsinh_()
        {
            var res = THSTensor_arcsinh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor asinh_() => arcsinh_();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arccosh(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor arccosh()
        {
            var res = THSTensor_arccosh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor acosh() => arccosh();


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arccosh_(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor arccosh_()
        {
            var res = THSTensor_arccosh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor acosh_() => arccosh_();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arctanh(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor arctanh()
        {
            var res = THSTensor_arctanh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor atanh() => arctanh();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arctanh_(IntPtr tensor);

        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor arctanh_()
        {
            var res = THSTensor_arctanh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor atanh_() => arctanh_();
    }
}
