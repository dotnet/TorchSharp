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

        /// <summary>
        /// Compute the absolute value of each element in the tensor
        /// </summary>
        public TorchTensor abs()
        {
            var res = THSTensor_abs(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Compute the absolute value of each element in the tensor
        /// </summary>
        public TorchTensor absolute() => abs();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_abs_(IntPtr tensor);

        /// <summary>
        /// Compute the absolute value of each element in the tensor, in-place.
        /// </summary>
        public TorchTensor abs_()
        {
            var res = THSTensor_abs_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        /// <summary>
        /// Compute the absolute value of each element in the tensor, in-place
        /// </summary>
        public TorchTensor absolute_() => abs_();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_add(IntPtr tensor, IntPtr trg, IntPtr alpha);

        /// <summary>
        /// Add two tensors, element-wise
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor add(TorchTensor target)
        {
            return add(target, 1);
        }

        /// <summary>
        /// Add two tensors, element-wise, scaling the second operator by 'alpha'
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
        /// Add a scalar value to each element in the target tensor.
        /// </summary>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public TorchTensor add(TorchScalar scalar)
        {
            return add(scalar, 1);
        }

        /// <summary>
        /// Add a scalar value to each element in the target tensor, scaled by 'alpha'
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
        /// In-place element-wise addition.
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor add_(TorchTensor target)
        {
            return add_(target, 1);
        }

        /// <summary>
        /// In-place element-wise addition, with scaling
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
        /// In-place scalar addition.
        /// </summary>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public TorchTensor add_(TorchScalar scalar)
        {
            return add_(scalar, 1);
        }

        /// <summary>
        /// In-place scalar addition, scaled.
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
        /// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced
        /// add step (all matrix multiplications get accumulated along the first dimension).
        /// input is added to the final result.
        /// </summary>
        /// <param name="mat1"></param>
        /// <param name="mat2"></param>
        /// <param name="beta"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        public TorchTensor addbmm(TorchTensor mat1, TorchTensor mat2, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_addbmm(handle, mat1.Handle, mat2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_addbmm_(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        /// <summary>
        /// Performs an in-place batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced
        /// add step (all matrix multiplications get accumulated along the first dimension).
        /// input is added to the final result.
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
        /// Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
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
        /// Performs the in-place element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
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
        /// Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
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
        /// Performs the in-place element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
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
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
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
        /// Performs an in-place matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
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

        /// <summary>
        /// Element-wise bitwise AND
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor bitwise_and(TorchTensor other)
        {
            var res = THSTensor_bitwise_and(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_and_(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Element-wise bitwise AND, in-place
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor bitwise_and_(TorchTensor other)
        {
            var res = THSTensor_bitwise_and_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_not(IntPtr tensor);

        /// <summary>
        /// Element-wise bitwise NOT
        /// </summary>
        /// <returns></returns>
        public TorchTensor bitwise_not()
        {
            var res = THSTensor_bitwise_not(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_not_(IntPtr tensor);

        /// <summary>
        /// Element-wise bitwise NOT, in-place
        /// </summary>
        /// <returns></returns>
        public TorchTensor bitwise_not_(TorchTensor other)
        {
            var res = THSTensor_bitwise_not_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_or(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Element-wise bitwise OR
        /// </summary>
        /// <returns></returns>
        public TorchTensor bitwise_or(TorchTensor other)
        {
            var res = THSTensor_bitwise_or(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_or_(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Element-wise bitwise OR, in-place
        /// </summary>
        /// <returns></returns>
        public TorchTensor bitwise_or_(TorchTensor other)
        {
            var res = THSTensor_bitwise_or_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_xor(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Element-wise bitwise XOR
        /// </summary>
        /// <returns></returns>
        public TorchTensor bitwise_xor(TorchTensor other)
        {
            var res = THSTensor_bitwise_xor(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_bitwise_xor_(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Element-wise bitwise XOR, in-place.
        /// </summary>
        /// <returns></returns>
        public TorchTensor bitwise_xor_(TorchTensor other)
        {
            var res = THSTensor_bitwise_xor_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ceil(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.
        /// </summary>
        /// <returns></returns>
        public TorchTensor ceil()
        {
            var res = THSTensor_ceil(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_ceil_(IntPtr tensor);

        /// <summary>
        /// Replaces each element of the input with the smallest integer greater than or equal to the element.
        /// </summary>
        /// <returns></returns>
        public TorchTensor ceil_()
        {
            var res = THSTensor_ceil_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor div(TorchTensor target)
        {
            var res = THSTensor_div(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor divide(TorchTensor target) => div(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div_scalar(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Scalar division
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor div(TorchScalar target)
        {
            var res = THSTensor_div_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        /// <summary>
        /// Scalar division
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor divide(TorchScalar target) => div(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div_(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// In-place division
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor div_(TorchTensor target)
        {
            var res = THSTensor_div_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_div_scalar_(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// In-place scalar division
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor div_(TorchScalar target)
        {
            var res = THSTensor_div_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
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

        /// <summary>
        /// Returns a new tensor with the exponential of the elements of the input tensor input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor exp_()
        {
            var res = THSTensor_exp_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_exp2(IntPtr tensor);

        /// <summary>
        /// Computes the base 2 exponential function of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor exp2()
        {
            var res = THSTensor_exp2(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_expm1(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the exponential of the elements minus 1 of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor expm1()
        {
            var res = THSTensor_expm1(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_expm1_(IntPtr tensor);

        /// <summary>
        /// Replaces each element with the exponential of the element minus 1 of input.
        /// </summary>
        /// <returns></returns>

        public TorchTensor expm1_()
        {
            var res = THSTensor_expm1_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_float_power(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Raises input to the power of exponent, elementwise, in double precision.
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        /// <remarks> If neither input is complex returns a torch.float64 tensor, and if one or more inputs is complex returns a torch.complex128 tensor.</remarks>
        public TorchTensor float_power(TorchTensor target)
        {
            var res = THSTensor_float_power(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_floor(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.
        /// </summary>
        /// <returns></returns>
        public TorchTensor floor()
        {
            var res = THSTensor_floor(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_floor_(IntPtr tensor);

        /// <summary>
        /// Replaces each element with the floor of the input, the largest integer less than or equal to each element.
        /// </summary>
        /// <returns></returns>
        public TorchTensor floor_()
        {
            var res = THSTensor_floor_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor fmod(TorchTensor target)
        {
            var res = THSTensor_fmod(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod_(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Computes the element-wise remainder of division, in-place.
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor fmod_(TorchTensor target)
        {
            var res = THSTensor_fmod_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod_scalar(IntPtr tensor, IntPtr scalar);

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public TorchTensor fmod(TorchScalar scalar)
        {
            var res = THSTensor_fmod_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_fmod_scalar_(IntPtr tensor, IntPtr scalar);

        /// <summary>
        /// Computes the element-wise remainder of division, in-place
        /// </summary>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public TorchTensor fmod_(TorchScalar scalar)
        {
            var res = THSTensor_fmod_scalar_(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_frac(IntPtr tensor);

        /// <summary>
        /// Computes the fractional portion of each element in input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor frac()
        {
            var res = THSTensor_frac(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_frac_(IntPtr tensor);

        /// <summary>
        /// Computes the fractional portion of each element in input, in-place.
        /// </summary>
        /// <returns></returns>
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

        /// <summary>
        /// Computes the element-wise greatest common divisor (GCD) of input and other.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor gcd_(TorchTensor other)
        {
            var res = THSTensor_gcd_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_hypot(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Element-wise: given the legs of a right triangle, return its hypotenuse.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor hypot(TorchTensor other)
        {
            var res = THSTensor_hypot(handle, other.handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the natural logarithm of the input elements.
        /// </summary>
        /// <returns></returns>
        public TorchTensor log()
        {
            var res = THSTensor_log(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log_(IntPtr tensor);

        /// <summary>
        /// Replaces each elements with the natural logarithm of the input.
        /// </summary>
        public TorchTensor log_()
        {
            var res = THSTensor_log_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logaddexp(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Logarithm of the sum of exponentiations of the inputs.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor logaddexp(TorchTensor other)
        {
            var res = THSTensor_logaddexp(handle, other.handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logaddexp2(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Logarithm of the sum of exponentiations of the inputs in base-2.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor logaddexp2(TorchTensor other)
        {
            var res = THSTensor_logaddexp2(handle, other.handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logcumsumexp(IntPtr tensor, long dim);

        /// <summary>
        /// Returns the logarithm of the cumulative summation of the exponentiation of elements of input in the dimension dim.
        /// </summary>
        /// <param name="dim"></param>
        /// <returns></returns>
        public TorchTensor logcumsumexp(long dim)
        {
            var res = THSTensor_logcumsumexp(handle, dim);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logsumexp(IntPtr tensor, long dim, bool keepdim);

        /// <summary>
        /// Returns the log of summed exponentials of each row of the input tensor in the given dimension dim. 
        /// </summary>
        /// <param name="dim"></param>
        /// <param name="keepdim"></param>
        /// <returns></returns>
        /// <remarks>The computation is numerically stabilized.</remarks>
        public TorchTensor logsumexp(long dim, Boolean keepdim = false)
        {
            var res = THSTensor_logsumexp(handle, dim, keepdim);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log10(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the logarithm to the base 10 of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor log10()
        {
            var res = THSTensor_log10(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log10_(IntPtr tensor);

        /// <summary>
        /// Replaces elements with the logarithm to the base 10 of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor log10_()
        {
            var res = THSTensor_log10_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log1p(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the natural logarithm of (1 + input).
        /// </summary>
        /// <returns></returns>
        public TorchTensor log1p()
        {
            var res = THSTensor_log1p(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log1p_(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the natural logarithm of (1 + input), inplace.
        /// </summary>
        /// <returns></returns>
        public TorchTensor log1p_()
        {
            var res = THSTensor_log1p_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log2(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the logarithm to the base 2 of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor log2()
        {
            var res = THSTensor_log2(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_log2_(IntPtr tensor);

        /// <summary>
        /// Replaces each element with the logarithm to the base 2 of the input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor log2_()
        {
            var res = THSTensor_log2_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_and(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Logical AND
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor logical_and(TorchTensor other)
        {
            var res = THSTensor_logical_and(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_and_(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Logical AND, in place
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor logical_and_(TorchTensor other)
        {
            var res = THSTensor_logical_and_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_not(IntPtr tensor);

        /// <summary>
        /// Logical NOT
        /// </summary>
        /// <returns></returns>
        public TorchTensor logical_not()
        {
            var res = THSTensor_logical_not(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_not_(IntPtr tensor);

        /// <summary>
        /// Logical NOT, in place
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor logical_not_(TorchTensor other)
        {
            var res = THSTensor_logical_not_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_or(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Logical OR
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor logical_or(TorchTensor other)
        {
            var res = THSTensor_logical_or(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_or_(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Logical OR, in place
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor logical_or_(TorchTensor other)
        {
            var res = THSTensor_logical_or_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_xor(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Logical XOR
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor logical_xor(TorchTensor other)
        {
            var res = THSTensor_logical_xor(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logical_xor_(IntPtr tensor, IntPtr other);

        /// <summary>
        /// Logical XOR, in place
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public TorchTensor logical_xor_(TorchTensor other)
        {
            var res = THSTensor_logical_xor_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_logit(IntPtr tensor, IntPtr eps);

        /// <summary>
        /// Returns a new tensor with the logit of the elements of input.
        /// input is clamped to [eps, 1 - eps] when eps is not null
        /// </summary>
        /// <param name="eps"></param>
        /// <returns></returns>
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

        /// <summary>
        /// Element-wise multiplication
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor mul(TorchTensor target)
        {
            var res = THSTensor_mul(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        /// <summary>
        /// Element-wise multiplication
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor multiply(TorchTensor target) => mul(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mul_scalar(IntPtr tensor, IntPtr scalar);

        /// <summary>
        /// Element-wise multiplication
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor mul(TorchScalar target)
        {
            var res = THSTensor_mul_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        /// <summary>
        /// Element-wise multiplcation
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor multiply(TorchScalar target) => mul(target);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mul_(IntPtr tensor, IntPtr target);

        /// <summary>
        /// Element-wise multiplication, in place
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor mul_(TorchTensor target)
        {
            var res = THSTensor_mul_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_mul_scalar_(IntPtr tensor, IntPtr target);

        /// <summary>
        /// Element-wise multiplication, in place
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
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

        /// <summary>
        /// Negation
        /// </summary>
        /// <returns></returns>
        public TorchTensor neg()
        {
            var res = THSTensor_neg(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        /// <summary>
        /// Negation
        /// </summary>
        /// <returns></returns>
        public TorchTensor negative() => neg();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_neg_(IntPtr tensor);

        /// <summary>
        /// In-place negation
        /// </summary>
        /// <returns></returns>
        public TorchTensor neg_()
        {
            var res = THSTensor_neg_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow(IntPtr tensor, IntPtr exponent);

        /// <summary>
        /// Takes the power of each element in input with exponent and returns a tensor with the result.
        /// </summary>
        /// <param name="exponent"></param>
        /// <returns></returns>
        public TorchTensor pow(TorchTensor exponent)
        {
            var res = THSTensor_pow(handle, exponent.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow_(IntPtr tensor, IntPtr exponent);

        /// <summary>
        /// Replaces each element in input with the power of the element and the exponent.
        /// </summary>
        /// <param name="exponent"></param>
        /// <returns></returns>
        public TorchTensor pow_(TorchTensor exponent)
        {
            var res = THSTensor_pow_(handle, exponent.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow_scalar(IntPtr tensor, IntPtr scalar);

        /// <summary>
        /// Takes the power of each element in input with exponent and returns a tensor with the result.
        /// </summary>
        /// <param name="exponent"></param>
        /// <returns></returns>
        public TorchTensor pow(TorchScalar exponent)
        {
            var res = THSTensor_pow_scalar(handle, exponent.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_pow_scalar_(IntPtr tensor, IntPtr scalar);

        /// <summary>
        /// Replaces each element in input with the power of the element and the exponent.
        /// </summary>
        /// <param name="exponent"></param>
        /// <returns></returns>
        public TorchTensor pow_(TorchScalar exponent)
        {
            var res = THSTensor_pow_scalar_(handle, exponent.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_reciprocal(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the reciprocal of the elements of input
        /// </summary>
        /// <returns></returns>
        public TorchTensor reciprocal()
        {
            var res = THSTensor_reciprocal(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_reciprocal_(IntPtr tensor);

        /// <summary>
        /// Replaces each element with the reciprocal of the input
        /// </summary>
        /// <returns></returns>
        public TorchTensor reciprocal_()
        {
            var res = THSTensor_reciprocal_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor remainder(TorchTensor target)
        {
            var res = THSTensor_remainder(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder_(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Computes the element-wise remainder of division, in place
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor remainder_(TorchTensor target)
        {
            var res = THSTensor_remainder_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder_scalar(IntPtr tensor, IntPtr scalar);

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public TorchTensor remainder(TorchScalar scalar)
        {
            var res = THSTensor_remainder_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_remainder_scalar_(IntPtr tensor, IntPtr scalar);

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public TorchTensor remainder_(TorchScalar scalar)
        {
            var res = THSTensor_remainder_scalar_(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_round(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with each of the elements of input rounded to the closest integer.
        /// </summary>
        /// <returns></returns>
        public TorchTensor round()
        {
            var res = THSTensor_round(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_round_(IntPtr tensor);

        /// <summary>
        /// Replaces each of the elements of input with the element rounded to the closest integer.
        /// </summary>
        /// <returns></returns>
        public TorchTensor round_()
        {
            var res = THSTensor_round_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rsqrt(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor rsqrt()
        {
            var res = THSTensor_rsqrt(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_rsqrt_(IntPtr tensor);

        /// <summary>
        /// Replaces each of the elements of input with  the reciprocal of the square-root of each of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor rsqrt_()
        {
            var res = THSTensor_rsqrt_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        /// <summary>
        /// Computes the element-wise square
        /// </summary>
        /// <returns></returns>
        public TorchTensor square() => this.pow(2);

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sqrt(IntPtr tensor);

        /// <summary>
        /// Computes the element-wise square root
        /// </summary>
        /// <returns></returns>
        public TorchTensor sqrt()
        {
            var res = THSTensor_sqrt(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sqrt_(IntPtr tensor);

        /// <summary>
        /// Computes the element-wise square root, in place
        /// </summary>
        /// <returns></returns>
        public TorchTensor sqrt_()
        {
            var res = THSTensor_sqrt_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sign(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the signs (-1, 0, 1) of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor sign()
        {
            var res = THSTensor_sign(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sign_(IntPtr tensor);

        /// <summary>
        /// Replaces each element with the signs (-1, 0, 1) of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor sign_()
        {
            var res = THSTensor_sign_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_signbit(IntPtr tensor);

        /// <summary>
        /// Tests if each element of input has its sign bit set (is less than zero) or not.
        /// </summary>
        /// <returns>A boolean tensor of the same shape as the input.</returns>
        public TorchTensor signbit()
        {
            var res = THSTensor_signbit(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor sub(TorchTensor target)
        {
            var res = THSTensor_sub(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub_scalar(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor sub(TorchScalar target)
        {
            var res = THSTensor_sub_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub_(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Element-wise subtraction, in place
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor sub_(TorchTensor target)
        {
            var res = THSTensor_sub_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_sub_scalar_(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Element-wise subtraction, in-place
        /// </summary>
        /// <param name="target"></param>
        /// <returns></returns>
        public TorchTensor sub_(TorchScalar target)
        {
            var res = THSTensor_sub_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_trunc(IntPtr tensor);

        /// <summary>
        /// Returns a new tensor with the truncated integer values of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor trunc()
        {
            var res = THSTensor_trunc(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        /// <summary>
        /// Returns a new tensor with the truncated integer values of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor fix() => trunc();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_trunc_(IntPtr tensor);

        /// <summary>
        /// Replaces each element with the truncated integer values of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor trunc_()
        {
            var res = THSTensor_trunc_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        /// <summary>
        /// Replaces each element with the truncated integer values of the elements of input.
        /// </summary>
        /// <returns></returns>
        public TorchTensor fix_() => trunc_();

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_xlogy(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Computes x * log(y)
        /// </summary>
        /// <param name="y"></param>
        /// <returns></returns>
        public TorchTensor xlogy(TorchTensor y)
        {
            var res = THSTensor_xlogy(handle, y.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_xlogy_(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Computes x * log(y) in place
        /// </summary>
        /// <param name="y"></param>
        /// <returns></returns>
        public TorchTensor xlogy_(TorchTensor y)
        {
            var res = THSTensor_xlogy_(handle, y.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_xlogy_scalar(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Computes x * log(y)
        /// </summary>
        /// <param name="y"></param>
        /// <returns></returns>
        public TorchTensor xlogy(TorchScalar y)
        {
            var res = THSTensor_xlogy_scalar(handle, y.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_xlogy_scalar_(IntPtr tensor, IntPtr trg);

        /// <summary>
        /// Computes x * log(y) in place
        /// </summary>
        /// <param name="y"></param>
        /// <returns></returns>
        public TorchTensor xlogy_(TorchScalar y)
        {
            var res = THSTensor_xlogy_scalar_(handle, y.Handle);
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
