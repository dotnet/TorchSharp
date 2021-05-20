using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
namespace TorchSharp.Tensor
{
    public sealed partial class TorchTensor
    {
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_angle(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor angle()
        {
            return new TorchTensor(THSTensor_angle(handle));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_asin(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor asin()
        {
            return new TorchTensor(THSTensor_asin(handle));
        }

        public TorchTensor arcsin() => asin();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_asin_(IntPtr tensor);

        /// <summary>
        /// 
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
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor acos()
        {
            var res = THSTensor_acos(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        public TorchTensor arccos() => acos();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_acos_(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor acos_()
        {
            var res = THSTensor_acos_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        public TorchTensor arccos_() => acos_();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan(IntPtr tensor);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public TorchTensor atan()
        {
            var res = THSTensor_atan(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        public TorchTensor arctan() => atan();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan_(IntPtr tensor);

        public TorchTensor atan_()
        {
            var res = THSTensor_atan_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        public TorchTensor arctan_() => atan_();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan2(IntPtr tensor, IntPtr other);

        public TorchTensor atan2(TorchTensor other)
        {
            var res = THSTensor_atan2(handle, other.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_atan2_(IntPtr tensor, IntPtr other);

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
        /// 
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
        /// 
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
        /// 
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
        /// 
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
        /// 
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
        /// 
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

        public TorchTensor sinc()
        {
            var res = THSTensor_sinc(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sinc_(IntPtr tensor);

        public TorchTensor sinc_()
        {
            var res = THSTensor_sinc_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sinh(IntPtr tensor);

        public TorchTensor sinh()
        {
            var res = THSTensor_sinh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sinh_(IntPtr tensor);

        public TorchTensor sinh_()
        {
            var res = THSTensor_sinh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cosh(IntPtr tensor);

        public TorchTensor cosh()
        {
            var res = THSTensor_cosh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_cosh_(IntPtr tensor);

        public TorchTensor cosh_()
        {
            var res = THSTensor_cosh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_tanh(IntPtr tensor);

        public TorchTensor tanh()
        {
            var res = THSTensor_tanh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_tanh_(IntPtr tensor);

        public TorchTensor tanh_()
        {
            var res = THSTensor_tanh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arcsinh(IntPtr tensor);

        public TorchTensor arcsinh()
        {
            var res = THSTensor_arcsinh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arcsinh_(IntPtr tensor);

        public TorchTensor arcsinh_()
        {
            var res = THSTensor_arcsinh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arccosh(IntPtr tensor);

        public TorchTensor arccosh()
        {
            var res = THSTensor_arccosh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arccosh_(IntPtr tensor);

        public TorchTensor arccosh_()
        {
            var res = THSTensor_arccosh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arctanh(IntPtr tensor);

        public TorchTensor arctanh()
        {
            var res = THSTensor_arctanh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_arctanh_(IntPtr tensor);

        public TorchTensor arctanh_()
        {
            var res = THSTensor_arctanh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        public TorchTensor asinh() => arcsinh();

        public TorchTensor asinh_() => arctanh_();

        public TorchTensor acosh() => arccosh();

        public TorchTensor acosh_() => arccosh_();

        public TorchTensor atanh() => arctanh();

        public TorchTensor atanh_() => arctanh_();

    }
}
