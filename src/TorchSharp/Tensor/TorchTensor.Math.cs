using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp.Tensor
{
    // This file contains the mathematical operators on TorchTensor

    public sealed partial class TorchTensor
    {
        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_abs(IntPtr tensor);

        public TorchTensor abs()
        {
            var res = THSTensor_abs(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        public TorchTensor absolute() => abs();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_abs_(IntPtr tensor);

        public TorchTensor abs_()
        {
            var res = THSTensor_abs_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        public TorchTensor absolute_() => abs_();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_add(IntPtr tensor, IntPtr trg, IntPtr alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor add(TorchTensor target)
        {
            return add(target, 1);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="target"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor add(TorchTensor target, TorchScalar alpha)
        {
            var res = THSTensor_add(handle, target.Handle, alpha.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_add_scalar(IntPtr tensor, IntPtr trg, IntPtr alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public TorchTensor add(TorchScalar scalar)
        {
            return add(scalar, 1);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="scalar"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor add(TorchScalar scalar, TorchScalar alpha)
        {
            return new TorchTensor(THSTensor_add_scalar(handle, scalar.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_add_(IntPtr tensor, IntPtr trg, IntPtr alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor add_(TorchTensor target)
        {
            return add_(target, 1);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="target"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor add_(TorchTensor target, TorchScalar alpha)
        {
            return new TorchTensor(THSTensor_add_(handle, target.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_add_scalar_(IntPtr tensor, IntPtr trg, IntPtr alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public TorchTensor add_(TorchScalar scalar)
        {
            return add_(scalar, 1);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="scalar"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor add_(TorchScalar scalar, TorchScalar alpha)
        {
            var res = THSTensor_add_scalar_(handle, scalar.Handle, alpha.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="batch1"></param>
        /// <param name="batch2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addbmm(TorchTensor batch1, TorchTensor batch2, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addbmm_(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="batch1"></param>
        /// <param name="batch2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addbmm_(TorchTensor batch1, TorchTensor batch2, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_addbmm_(handle, batch1.Handle, batch2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addcdiv(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor1"></param>
        /// <param name="tensor2"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public TorchTensor addcdiv(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcdiv(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addcdiv_(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor1"></param>
        /// <param name="tensor2"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public TorchTensor addcdiv_(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcdiv_(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addcmul(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor1"></param>
        /// <param name="tensor2"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public TorchTensor addcmul(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcmul(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addcmul_(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor1"></param>
        /// <param name="tensor2"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public TorchTensor addcmul_(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcmul_(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addmm(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mat1"></param>
        /// <param name="mat2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addmm(TorchTensor mat1, TorchTensor mat2, float beta, float alpha)
        {
            var res = THSTensor_addmm(handle, mat1.Handle, mat2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addmm_(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mat1"></param>
        /// <param name="mat2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addmm_(TorchTensor mat1, TorchTensor mat2, float beta, float alpha)
        {
            var res = THSTensor_addmm_(handle, mat1.Handle, mat2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addmv(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        /// <summary>
        /// Performs a matrix-vector product of the matrix mat and the vector vec. The vector input is added to the final result.
        /// </summary>
        /// <param name="vec1"></param>
        /// <param name="vec2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addmv(TorchTensor vec1, TorchTensor vec2, float beta = 1.0f, float alpha = 1.0f)
        {
            var res = THSTensor_addmv(handle, vec1.Handle, vec2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addmv_(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        /// <summary>
        /// Performs a matrix-vector product of the matrix mat and the vector vec. The vector input is added to the final result.
        /// </summary>
        /// <param name="vec1"></param>
        /// <param name="vec2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addmv_(TorchTensor vec1, TorchTensor vec2, float beta = 1.0f, float alpha = 1.0f)
        {
            var res = THSTensor_addmv_(handle, vec1.Handle, vec2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addr(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        /// <summary>
        /// Performs the outer-product of vectors vec1 and vec2 and adds it to the input tensor.
        /// </summary>
        /// <param name="vec1"></param>
        /// <param name="vec2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addr(TorchTensor vec1, TorchTensor vec2, float beta = 1.0f, float alpha = 1.0f)
        {
            var res = THSTensor_addr(handle, vec1.Handle, vec2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addr_(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        /// <summary>
        /// Performs the outer-product of vectors vec1 and vec2 and adds it to the input tensor.
        ///
        /// In-place version of 'addr'
        /// </summary>
        /// <param name="vec1"></param>
        /// <param name="vec2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addr_(TorchTensor vec1, TorchTensor vec2, float beta = 1.0f, float alpha = 1.0f)
        {
            var res = THSTensor_addr_(handle, vec1.Handle, vec2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_and(IntPtr tensor, IntPtr other);

        public TorchTensor bitwise_and(TorchTensor other)
        {
            var res = THSTensor_bitwise_and(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_and_(IntPtr tensor, IntPtr other);

        public TorchTensor bitwise_and_(TorchTensor other)
        {
            var res = THSTensor_bitwise_and_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_not(IntPtr tensor);

        public TorchTensor bitwise_not()
        {
            var res = THSTensor_bitwise_not(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_not_(IntPtr tensor);

        public TorchTensor bitwise_not_(TorchTensor other)
        {
            var res = THSTensor_bitwise_not_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_or(IntPtr tensor, IntPtr other);

        public TorchTensor bitwise_or(TorchTensor other)
        {
            var res = THSTensor_bitwise_or(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_or_(IntPtr tensor, IntPtr other);

        public TorchTensor bitwise_or_(TorchTensor other)
        {
            var res = THSTensor_bitwise_or_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_xor(IntPtr tensor, IntPtr other);

        public TorchTensor bitwise_xor(TorchTensor other)
        {
            var res = THSTensor_bitwise_xor(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_xor_(IntPtr tensor, IntPtr other);

        public TorchTensor bitwise_xor_(TorchTensor other)
        {
            var res = THSTensor_bitwise_xor_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ceil(IntPtr tensor);

        public TorchTensor ceil()
        {
            var res = THSTensor_ceil(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ceil_(IntPtr tensor);

        public TorchTensor ceil_()
        {
            var res = THSTensor_ceil_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div(IntPtr tensor, IntPtr trg);

        public TorchTensor div(TorchTensor target)
        {
            var res = THSTensor_div(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor divide(TorchTensor target) => div(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor div(TorchScalar target)
        {
            var res = THSTensor_div_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
        public TorchTensor divide(TorchScalar target) => div(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div_(IntPtr tensor, IntPtr trg);

        public TorchTensor div_(TorchTensor target)
        {
            var res = THSTensor_div_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor div_(TorchScalar target)
        {
            var res = THSTensor_div_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_floor(IntPtr tensor);

        public TorchTensor floor()
        {
            var res = THSTensor_floor(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_exp(IntPtr tensor);

        public TorchTensor exp()
        {
            var res = THSTensor_exp(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_exp_(IntPtr tensor);

        public TorchTensor exp_()
        {
            var res = THSTensor_exp_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_exp2(IntPtr tensor);

        public TorchTensor exp2()
        {
            var res = THSTensor_exp2(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_expm1(IntPtr tensor);

        public TorchTensor expm1()
        {
            var res = THSTensor_expm1(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_expm1_(IntPtr tensor);

        public TorchTensor expm1_()
        {
            var res = THSTensor_expm1_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fft(IntPtr tensor, long n, long dim, [MarshalAs(UnmanagedType.LPStr)] string norm);

        public TorchTensor fft(long? n, long dim = -1, string norm = "backward")
        {
            var res = THSTensor_fft(handle, n.GetValueOrDefault(-1), dim, norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ifft(IntPtr tensor, long n, long dim, [MarshalAs(UnmanagedType.LPStr)] string norm);

        public TorchTensor ifft(long? n, long dim = -1, string norm = "backward")
        {
            var res = THSTensor_ifft(handle, n.GetValueOrDefault(-1), dim, norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_irfft(IntPtr tensor, long n, long dim, [MarshalAs(UnmanagedType.LPStr)] string norm);

        public TorchTensor irfft(long? n, long dim = -1, string norm = "backward")
        {
            var res = THSTensor_irfft(handle, n.GetValueOrDefault(-1), dim, norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rfft(IntPtr tensor, long n, long dim, [MarshalAs(UnmanagedType.LPStr)] string norm);

        public TorchTensor rfft(long? n, long dim = -1, string norm = "backward")
        {
            var res = THSTensor_rfft(handle, n.GetValueOrDefault(-1), dim, norm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_float_power(IntPtr tensor, IntPtr trg);

        public TorchTensor float_power(TorchTensor target)
        {
            var res = THSTensor_float_power(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_floor_(IntPtr tensor);

        public TorchTensor floor_()
        {
            var res = THSTensor_floor_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod(IntPtr tensor, IntPtr trg);

        public TorchTensor fmod(TorchTensor target)
        {
            var res = THSTensor_fmod(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod_(IntPtr tensor, IntPtr trg);

        public TorchTensor fmod_(TorchTensor target)
        {
            var res = THSTensor_fmod_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor fmod(TorchScalar scalar)
        {
            var res = THSTensor_fmod_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod_scalar_(IntPtr tensor, IntPtr scalar);

        public TorchTensor fmod_(TorchScalar scalar)
        {
            var res = THSTensor_fmod_scalar_(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_frac(IntPtr tensor);

        public TorchTensor frac()
        {
            var res = THSTensor_frac(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_frac_(IntPtr tensor);

        public TorchTensor frac_()
        {
            var res = THSTensor_frac_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gcd(IntPtr tensor, IntPtr other);

        public TorchTensor gcd(TorchTensor other)
        {
            var res = THSTensor_gcd(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_gcd_(IntPtr tensor, IntPtr other);

        public TorchTensor gcd_(TorchTensor other)
        {
            var res = THSTensor_gcd_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hypot(IntPtr tensor, IntPtr other);

        public TorchTensor hypot(TorchTensor other)
        {
            var res = THSTensor_hypot(handle, other.handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log(IntPtr tensor);

        public TorchTensor log()
        {
            var res = THSTensor_log(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log_(IntPtr tensor);

        public TorchTensor log_()
        {
            var res = THSTensor_log_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logaddexp(IntPtr tensor, IntPtr other);

        public TorchTensor logaddexp(TorchTensor other)
        {
            var res = THSTensor_logaddexp(handle, other.handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logaddexp2(IntPtr tensor, IntPtr other);

        public TorchTensor logaddexp2(TorchTensor other)
        {
            var res = THSTensor_logaddexp2(handle, other.handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logcumsumexp(IntPtr tensor, long dim);

        public TorchTensor logcumsumexp(long dim)
        {
            var res = THSTensor_logcumsumexp(handle, dim);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logsumexp(IntPtr tensor, long dim, bool keepdim);

        public TorchTensor logsumexp(long dim, Boolean keepdim = false)
        {
            var res = THSTensor_logsumexp(handle, dim, keepdim);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log10(IntPtr tensor);

        public TorchTensor log10()
        {
            var res = THSTensor_log10(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log10_(IntPtr tensor);

        public TorchTensor log10_()
        {
            var res = THSTensor_log10_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log1p(IntPtr tensor);

        public TorchTensor log1p()
        {
            var res = THSTensor_log1p(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log1p_(IntPtr tensor);

        public TorchTensor log1p_()
        {
            var res = THSTensor_log1p_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log2(IntPtr tensor);

        public TorchTensor log2()
        {
            var res = THSTensor_log2(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log2_(IntPtr tensor);

        public TorchTensor log2_()
        {
            var res = THSTensor_log2_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_and(IntPtr tensor, IntPtr other);

        public TorchTensor logical_and(TorchTensor other)
        {
            var res = THSTensor_logical_and(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_and_(IntPtr tensor, IntPtr other);

        public TorchTensor logical_and_(TorchTensor other)
        {
            var res = THSTensor_logical_and_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_not(IntPtr tensor);

        public TorchTensor logical_not()
        {
            var res = THSTensor_logical_not(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_not_(IntPtr tensor);

        public TorchTensor logical_not_(TorchTensor other)
        {
            var res = THSTensor_logical_not_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_or(IntPtr tensor, IntPtr other);

        public TorchTensor logical_or(TorchTensor other)
        {
            var res = THSTensor_logical_or(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_or_(IntPtr tensor, IntPtr other);

        public TorchTensor logical_or_(TorchTensor other)
        {
            var res = THSTensor_logical_or_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_xor(IntPtr tensor, IntPtr other);

        public TorchTensor logical_xor(TorchTensor other)
        {
            var res = THSTensor_logical_xor(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_xor_(IntPtr tensor, IntPtr other);

        public TorchTensor logical_xor_(TorchTensor other)
        {
            var res = THSTensor_logical_xor_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logit(IntPtr tensor, IntPtr eps);

        public TorchTensor logit(double? eps = null)
        {
            var epsArr = eps.HasValue ? new double[] { eps.Value } : null;

            unsafe {
                fixed (double* pEps = epsArr) {
                    var res = THSTensor_logit(handle, (IntPtr)pEps);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mul(IntPtr tensor, IntPtr target);

        public TorchTensor mul(TorchTensor target)
        {
            var res = THSTensor_mul(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor multiply(TorchTensor target) => mul(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mul_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor mul(TorchScalar scalar)
        {
            var res = THSTensor_mul_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor multiply(TorchScalar target) => mul(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mul_(IntPtr tensor, IntPtr target);

        public TorchTensor mul_(TorchTensor target)
        {
            var res = THSTensor_mul_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mul_scalar_(IntPtr tensor, IntPtr target);

        public TorchTensor mul_(TorchScalar target)
        {
            var res = THSTensor_mul_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public static TorchTensor operator -(TorchTensor tensor)
        {
            return tensor.neg();
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_neg(IntPtr tensor);

        public TorchTensor neg()
        {
            var res = THSTensor_neg(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor negative() => neg();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_neg_(IntPtr tensor);

        public TorchTensor neg_()
        {
            var res = THSTensor_neg_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow(IntPtr tensor, IntPtr exponent);

        public TorchTensor pow(TorchTensor exponent)
        {
            var res = THSTensor_pow(handle, exponent.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow_(IntPtr tensor, IntPtr exponent);

        public TorchTensor pow_(TorchTensor exponent)
        {
            var res = THSTensor_pow_(handle, exponent.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor pow(TorchScalar scalar)
        {
            var res = THSTensor_pow_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow_scalar_(IntPtr tensor, IntPtr scalar);

        public TorchTensor pow_(TorchScalar scalar)
        {
            var res = THSTensor_pow_scalar_(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_reciprocal(IntPtr tensor);

        public TorchTensor reciprocal()
        {
            var res = THSTensor_reciprocal(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_reciprocal_(IntPtr tensor);

        public TorchTensor reciprocal_()
        {
            var res = THSTensor_reciprocal_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder(IntPtr tensor, IntPtr trg);

        public TorchTensor remainder(TorchTensor target)
        {
            var res = THSTensor_remainder(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder_(IntPtr tensor, IntPtr trg);

        public TorchTensor remainder_(TorchTensor target)
        {
            var res = THSTensor_remainder_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor remainder(TorchScalar scalar)
        {
            var res = THSTensor_remainder_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder_scalar_(IntPtr tensor, IntPtr scalar);

        public TorchTensor remainder_(TorchScalar scalar)
        {
            var res = THSTensor_remainder_scalar_(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_round(IntPtr tensor);

        public TorchTensor round()
        {
            var res = THSTensor_round(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_round_(IntPtr tensor);

        public TorchTensor round_()
        {
            var res = THSTensor_round_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rsqrt(IntPtr tensor);

        public TorchTensor rsqrt()
        {
            var res = THSTensor_rsqrt(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rsqrt_(IntPtr tensor);

        public TorchTensor rsqrt_()
        {
            var res = THSTensor_rsqrt_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sqrt(IntPtr tensor);

        public TorchTensor sqrt()
        {
            var res = THSTensor_sqrt(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sqrt_(IntPtr tensor);

        public TorchTensor sqrt_()
        {
            var res = THSTensor_sqrt_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sign(IntPtr tensor);

        public TorchTensor sign()
        {
            var res = THSTensor_sign(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sign_(IntPtr tensor);

        public TorchTensor sign_()
        {
            var res = THSTensor_sign_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_signbit(IntPtr tensor);

        public TorchTensor signbit()
        {
            var res = THSTensor_signbit(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub(IntPtr tensor, IntPtr trg);

        public TorchTensor sub(TorchTensor target)
        {
            var res = THSTensor_sub(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor sub(TorchScalar target)
        {
            var res = THSTensor_sub_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub_(IntPtr tensor, IntPtr trg);

        public TorchTensor sub_(TorchTensor target)
        {
            var res = THSTensor_sub_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor sub_(TorchScalar target)
        {
            var res = THSTensor_sub_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_trunc(IntPtr tensor);

        public TorchTensor trunc()
        {
            var res = THSTensor_trunc(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor fix() => trunc();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_trunc_(IntPtr tensor);

        public TorchTensor trunc_()
        {
            var res = THSTensor_trunc_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor fix_() => trunc_();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_xlogy(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Computes x * log(y)
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor xlogy(TorchTensor target)
        {
            var res = THSTensor_xlogy(handle, target.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        // Overloaded operators

        public static TorchTensor operator +(TorchTensor left, TorchTensor right)
        {
            return left.add(right);
        }

        public static TorchTensor operator +(TorchTensor left, TorchScalar right)
        {
            return left.add(right);
        }

        public static TorchTensor operator +(TorchScalar left, TorchTensor right)
        {
            return right.add(left);
        }

        public static TorchTensor operator *(TorchTensor left, TorchTensor right)
        {
            return left.mul(right);
        }

        public static TorchTensor operator *(TorchTensor left, TorchScalar right)
        {
            return left.mul(right);
        }

        public static TorchTensor operator *(TorchScalar left, TorchTensor right)
        {
            return right.mul(left);
        }

        public static TorchTensor operator -(TorchTensor left, TorchTensor right)
        {
            return left.sub(right);
        }

        public static TorchTensor operator -(TorchTensor left, TorchScalar right)
        {
            return left.sub(right);
        }

        public static TorchTensor operator /(TorchTensor left, TorchTensor right)
        {
            return left.div(right);
        }

        public static TorchTensor operator /(TorchTensor left, TorchScalar right)
        {
            return left.div(right);
        }

        public static TorchTensor operator %(TorchTensor left, TorchTensor right)
        {
            return left.remainder(right);
        }

        public static TorchTensor operator %(TorchTensor left, TorchScalar right)
        {
            return left.remainder(right);
        }

    }
}
