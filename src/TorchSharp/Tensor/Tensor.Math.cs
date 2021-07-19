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

        public enum RoundingMode
        {
            None,
            trunc,
            floor
        }

        // This file contains the mathematical operators on Tensor

        public sealed partial class Tensor
        {
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_abs(IntPtr tensor);

            /// <summary>
            /// Compute the absolute value of each element in the tensor
            /// </summary>
            public Tensor abs()
            {
                var res = THSTensor_abs(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Compute the absolute value of each element in the tensor
            /// </summary>
            public Tensor absolute() => abs();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_abs_(IntPtr tensor);

            /// <summary>
            /// Compute the absolute value of each element in the tensor, in-place.
            /// </summary>
            public Tensor abs_()
            {
                var res = THSTensor_abs_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Compute the absolute value of each element in the tensor, in-place
            /// </summary>
            public Tensor absolute_() => abs_();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_add(IntPtr tensor, IntPtr trg, IntPtr alpha);

            /// <summary>
            /// Add two tensors, element-wise
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor add(Tensor target)
            {
                return add(target, 1);
            }

            /// <summary>
            /// Add two tensors, element-wise, scaling the second operator by 'alpha'
            /// </summary>
            /// <param name="target"></param>
            /// <param name="alpha"></param>
            /// <returns></returns>
            public Tensor add(Tensor target, Scalar alpha)
            {
                var res = THSTensor_add(handle, target.Handle, alpha.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_add_scalar(IntPtr tensor, IntPtr trg, IntPtr alpha);

            /// <summary>
            /// Add a scalar value to each element in the target tensor.
            /// </summary>
            /// <param name="scalar"></param>
            /// <returns></returns>
            public Tensor add(Scalar scalar)
            {
                return add(scalar, 1);
            }

            /// <summary>
            /// Add a scalar value to each element in the target tensor, scaled by 'alpha'
            /// </summary>
            /// <param name="scalar"></param>
            /// <param name="alpha"></param>
            /// <returns></returns>
            public Tensor add(Scalar scalar, Scalar alpha)
            {
                return new Tensor(THSTensor_add_scalar(handle, scalar.Handle, alpha.Handle));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_add_(IntPtr tensor, IntPtr trg, IntPtr alpha);

            /// <summary>
            /// In-place element-wise addition.
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor add_(Tensor target)
            {
                return add_(target, 1);
            }

            /// <summary>
            /// In-place element-wise addition, with scaling
            /// </summary>
            /// <param name="target"></param>
            /// <param name="alpha"></param>
            /// <returns></returns>
            public Tensor add_(Tensor target, Scalar alpha)
            {
                return new Tensor(THSTensor_add_(handle, target.Handle, alpha.Handle));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_add_scalar_(IntPtr tensor, IntPtr trg, IntPtr alpha);

            /// <summary>
            /// In-place scalar addition.
            /// </summary>
            /// <param name="scalar"></param>
            /// <returns></returns>
            public Tensor add_(Scalar scalar)
            {
                return add_(scalar, 1);
            }

            /// <summary>
            /// In-place scalar addition, scaled.
            /// </summary>
            /// <param name="scalar"></param>
            /// <param name="alpha"></param>
            /// <returns></returns>
            public Tensor add_(Scalar scalar, Scalar alpha)
            {
                var res = THSTensor_add_scalar_(handle, scalar.Handle, alpha.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addbmm(Tensor mat1, Tensor mat2, float beta = 1, float alpha = 1)
            {
                var res = THSTensor_addbmm(handle, mat1.Handle, mat2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addbmm_(Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1)
            {
                var res = THSTensor_addbmm_(handle, batch1.Handle, batch2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addcdiv(Tensor tensor1, Tensor tensor2, Scalar value)
            {
                var res = THSTensor_addcdiv(handle, tensor1.Handle, tensor2.Handle, value.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addcdiv_(Tensor tensor1, Tensor tensor2, Scalar value)
            {
                var res = THSTensor_addcdiv_(handle, tensor1.Handle, tensor2.Handle, value.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addcmul(Tensor tensor1, Tensor tensor2, Scalar value)
            {
                var res = THSTensor_addcmul(handle, tensor1.Handle, tensor2.Handle, value.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addcmul_(Tensor tensor1, Tensor tensor2, Scalar value)
            {
                var res = THSTensor_addcmul_(handle, tensor1.Handle, tensor2.Handle, value.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addmm(Tensor mat1, Tensor mat2, float beta, float alpha)
            {
                var res = THSTensor_addmm(handle, mat1.Handle, mat2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addmm_(Tensor mat1, Tensor mat2, float beta, float alpha)
            {
                var res = THSTensor_addmm_(handle, mat1.Handle, mat2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addmv(Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f)
            {
                var res = THSTensor_addmv(handle, vec1.Handle, vec2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addmv_(Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f)
            {
                var res = THSTensor_addmv_(handle, vec1.Handle, vec2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addr(Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f)
            {
                var res = THSTensor_addr(handle, vec1.Handle, vec2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor addr_(Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f)
            {
                var res = THSTensor_addr_(handle, vec1.Handle, vec2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_and(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise AND
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor bitwise_and(Tensor other)
            {
                var res = THSTensor_bitwise_and(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_and_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise AND, in-place
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor bitwise_and_(Tensor other)
            {
                var res = THSTensor_bitwise_and_(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_not(IntPtr tensor);

            /// <summary>
            /// Element-wise bitwise NOT
            /// </summary>
            /// <returns></returns>
            public Tensor bitwise_not()
            {
                var res = THSTensor_bitwise_not(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_not_(IntPtr tensor);

            /// <summary>
            /// Element-wise bitwise NOT, in-place
            /// </summary>
            /// <returns></returns>
            public Tensor bitwise_not_()
            {
                var res = THSTensor_bitwise_not_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_or(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise OR
            /// </summary>
            /// <returns></returns>
            public Tensor bitwise_or(Tensor other)
            {
                var res = THSTensor_bitwise_or(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_or_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise OR, in-place
            /// </summary>
            /// <returns></returns>
            public Tensor bitwise_or_(Tensor other)
            {
                var res = THSTensor_bitwise_or_(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_xor(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise XOR
            /// </summary>
            /// <returns></returns>
            public Tensor bitwise_xor(Tensor other)
            {
                var res = THSTensor_bitwise_xor(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_xor_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise XOR, in-place.
            /// </summary>
            /// <returns></returns>
            public Tensor bitwise_xor_(Tensor other)
            {
                var res = THSTensor_bitwise_xor_(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ceil(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.
            /// </summary>
            /// <returns></returns>
            public Tensor ceil()
            {
                var res = THSTensor_ceil(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_ceil_(IntPtr tensor);

            /// <summary>
            /// Replaces each element of the input with the smallest integer greater than or equal to the element.
            /// </summary>
            /// <returns></returns>
            public Tensor ceil_()
            {
                var res = THSTensor_ceil_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_cummax(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

            public (Tensor values, Tensor indexes) cummax(long dimension)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_cummax(handle, pa.CreateArray, dimension);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_cummin(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

            public (Tensor values, Tensor indexes) cummin(long dimension)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_cummin(handle, pa.CreateArray, dimension);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cumsum(IntPtr tensor, long dimension, bool has_type, sbyte scalar_type);

            public Tensor cumsum(long dimension, ScalarType? type = null)
            {
                var res = THSTensor_cumsum(handle, dimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cumprod(IntPtr tensor, long dimension, bool has_type, sbyte scalar_type);

            public Tensor cumprod(long dimension, ScalarType? type = null)
            {
                var res = THSTensor_cumprod(handle, dimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_div(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string rounding_mode);

            /// <summary>
            /// Divides each element of the input by the corresponding element of other.
            /// </summary>
            /// <param name="target"></param>
            /// <param name="rounding_mode"></param>
            /// <returns></returns>
            public Tensor div(Tensor target, RoundingMode rounding_mode = RoundingMode.None)
            {
                var res = THSTensor_div(handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Divides each element of the input by the corresponding element of other.
            /// </summary>
            /// <param name="target"></param>
            /// <param name="rounding_mode"></param>
            /// <returns></returns>
            public Tensor divide(Tensor target, RoundingMode rounding_mode = RoundingMode.None) => div(target, rounding_mode);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_div_scalar(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string rounding_mode);

            /// <summary>
            /// Scalar division
            /// </summary>
            /// <param name="target"></param>
            /// <param name="rounding_mode"></param>
            /// <returns></returns>
            public Tensor div(Scalar target, RoundingMode rounding_mode = RoundingMode.None)
            {
                var res = THSTensor_div_scalar(handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Scalar division
            /// </summary>
            /// <param name="target"></param>
            /// <param name="rounding_mode"></param>
            /// <returns></returns>
            public Tensor divide(Scalar target, RoundingMode rounding_mode = RoundingMode.None) => div(target, rounding_mode);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_div_(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string rounding_mode);

            /// <summary>
            /// In-place division
            /// </summary>
            /// <param name="target"></param>
            /// <param name="rounding_mode"></param>
            /// <returns></returns>
            public Tensor div_(Tensor target, RoundingMode rounding_mode = RoundingMode.None)
            {
                var res = THSTensor_div_(handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// In-place division
            /// </summary>
            /// <param name="target"></param>
            /// <param name="rounding_mode"></param>
            /// <returns></returns>
            public Tensor divide_(Tensor target, RoundingMode rounding_mode = RoundingMode.None) => div_(target, rounding_mode);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_div_scalar_(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string rounding_mode);

            /// <summary>
            /// In-place scalar division
            /// </summary>
            /// <param name="target"></param>
            /// <param name="rounding_mode"></param>
            /// <returns></returns>
            public Tensor div_(Scalar target, RoundingMode rounding_mode = RoundingMode.None)
            {
                var res = THSTensor_div_scalar_(handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// In-place scalar division
            /// </summary>
            /// <param name="target"></param>
            /// <param name="rounding_mode"></param>
            /// <returns></returns>
            public Tensor divide_(Scalar target, RoundingMode rounding_mode = RoundingMode.None) => div_(target, rounding_mode);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_exp(IntPtr tensor);

            public Tensor exp()
            {
                var res = THSTensor_exp(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_exp_(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the exponential of the elements of the input tensor input.
            /// </summary>
            /// <returns></returns>
            public Tensor exp_()
            {
                var res = THSTensor_exp_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_exp2(IntPtr tensor);

            /// <summary>
            /// Computes the base 2 exponential function of input.
            /// </summary>
            /// <returns></returns>
            public Tensor exp2()
            {
                var res = THSTensor_exp2(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_expm1(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the exponential of the elements minus 1 of input.
            /// </summary>
            /// <returns></returns>
            public Tensor expm1()
            {
                var res = THSTensor_expm1(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_expm1_(IntPtr tensor);

            /// <summary>
            /// Replaces each element with the exponential of the element minus 1 of input.
            /// </summary>
            /// <returns></returns>
            public Tensor expm1_()
            {
                var res = THSTensor_expm1_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_float_power(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Raises input to the power of exponent, elementwise, in double precision.
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            /// <remarks> If neither input is complex returns a torch.float64 tensor, and if one or more inputs is complex returns a torch.complex128 tensor.</remarks>
            public Tensor float_power(Tensor target)
            {
                var res = THSTensor_float_power(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_floor(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.
            /// </summary>
            /// <returns></returns>
            public Tensor floor()
            {
                var res = THSTensor_floor(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_floor_(IntPtr tensor);

            /// <summary>
            /// Replaces each element with the floor of the input, the largest integer less than or equal to each element.
            /// </summary>
            /// <returns></returns>
            public Tensor floor_()
            {
                var res = THSTensor_floor_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmod(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor fmod(Tensor target)
            {
                var res = THSTensor_fmod(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmod_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes the element-wise remainder of division, in-place.
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor fmod_(Tensor target)
            {
                var res = THSTensor_fmod_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmod_scalar(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="scalar"></param>
            /// <returns></returns>
            public Tensor fmod(Scalar scalar)
            {
                var res = THSTensor_fmod_scalar(handle, scalar.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmod_scalar_(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Computes the element-wise remainder of division, in-place
            /// </summary>
            /// <param name="scalar"></param>
            /// <returns></returns>
            public Tensor fmod_(Scalar scalar)
            {
                var res = THSTensor_fmod_scalar_(handle, scalar.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_frac(IntPtr tensor);

            /// <summary>
            /// Computes the fractional portion of each element in input.
            /// </summary>
            /// <returns></returns>
            public Tensor frac()
            {
                var res = THSTensor_frac(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_frac_(IntPtr tensor);

            /// <summary>
            /// Computes the fractional portion of each element in input, in-place.
            /// </summary>
            /// <returns></returns>
            public Tensor frac_()
            {
                var res = THSTensor_frac_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_frexp(IntPtr tensor, out IntPtr exponent);

            /// <summary>
            /// Decomposes input into mantissa and exponent tensors 
            /// </summary>
            /// <returns></returns>
            public (Tensor Mantissa, Tensor Exponent) frexp()
            {
                var mantissa = THSTensor_frexp(handle, out var exponent);
                if (mantissa == IntPtr.Zero || exponent == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(mantissa), new Tensor(exponent));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gcd(IntPtr tensor, IntPtr other);

            public Tensor gcd(Tensor other)
            {
                var res = THSTensor_gcd(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gcd_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Computes the element-wise greatest common divisor (GCD) of input and other.
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor gcd_(Tensor other)
            {
                var res = THSTensor_gcd_(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_histc(IntPtr tensor, long bins, long min, long max);

            /// <summary>
            /// Computes the histogram of a tensor.
            /// The elements are sorted into equal width bins between min and max.If min and max are both zero, the minimum and maximum values of the data are used.
            /// Elements lower than min and higher than max are ignored.
            /// </summary>
            /// <param name="bins">Number of histogram bins</param>
            /// <param name="min">Lower end of the range (inclusive)</param>
            /// <param name="max">Upper end of the range (inclusive)</param>
            /// <returns></returns>
            public Tensor histc(long bins = 100, long min = 0, long max = 0)
            {
                var res = THSTensor_histc(handle, bins, min, max);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hypot(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise: given the legs of a right triangle, return its hypotenuse.
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor hypot(Tensor other)
            {
                var res = THSTensor_hypot(handle, other.handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_log(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the natural logarithm of the input elements.
            /// </summary>
            /// <returns></returns>
            public Tensor log()
            {
                var res = THSTensor_log(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_log_(IntPtr tensor);

            /// <summary>
            /// Replaces each elements with the natural logarithm of the input.
            /// </summary>
            public Tensor log_()
            {
                var res = THSTensor_log_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logaddexp(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logarithm of the sum of exponentiations of the inputs.
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor logaddexp(Tensor other)
            {
                var res = THSTensor_logaddexp(handle, other.handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logaddexp2(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logarithm of the sum of exponentiations of the inputs in base-2.
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor logaddexp2(Tensor other)
            {
                var res = THSTensor_logaddexp2(handle, other.handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logcumsumexp(IntPtr tensor, long dim);

            /// <summary>
            /// Returns the logarithm of the cumulative summation of the exponentiation of elements of input in the dimension dim.
            /// </summary>
            /// <param name="dim"></param>
            /// <returns></returns>
            public Tensor logcumsumexp(long dim)
            {
                var res = THSTensor_logcumsumexp(handle, dim);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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
            public Tensor logsumexp(long dim, Boolean keepdim = false)
            {
                var res = THSTensor_logsumexp(handle, dim, keepdim);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_log10(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the logarithm to the base 10 of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor log10()
            {
                var res = THSTensor_log10(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_log10_(IntPtr tensor);

            /// <summary>
            /// Replaces elements with the logarithm to the base 10 of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor log10_()
            {
                var res = THSTensor_log10_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_log1p(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the natural logarithm of (1 + input).
            /// </summary>
            /// <returns></returns>
            public Tensor log1p()
            {
                var res = THSTensor_log1p(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_log1p_(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the natural logarithm of (1 + input), inplace.
            /// </summary>
            /// <returns></returns>
            public Tensor log1p_()
            {
                var res = THSTensor_log1p_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_log2(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the logarithm to the base 2 of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor log2()
            {
                var res = THSTensor_log2(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_log2_(IntPtr tensor);

            /// <summary>
            /// Replaces each element with the logarithm to the base 2 of the input.
            /// </summary>
            /// <returns></returns>
            public Tensor log2_()
            {
                var res = THSTensor_log2_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_and(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical AND
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor logical_and(Tensor other)
            {
                var res = THSTensor_logical_and(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_and_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical AND, in place
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor logical_and_(Tensor other)
            {
                var res = THSTensor_logical_and_(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_not(IntPtr tensor);

            /// <summary>
            /// Logical NOT
            /// </summary>
            /// <returns></returns>
            public Tensor logical_not()
            {
                var res = THSTensor_logical_not(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_not_(IntPtr tensor);

            /// <summary>
            /// Logical NOT, in place
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor logical_not_(Tensor other)
            {
                var res = THSTensor_logical_not_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_or(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical OR
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor logical_or(Tensor other)
            {
                var res = THSTensor_logical_or(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_or_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical OR, in place
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor logical_or_(Tensor other)
            {
                var res = THSTensor_logical_or_(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_xor(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical XOR
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor logical_xor(Tensor other)
            {
                var res = THSTensor_logical_xor(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_xor_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical XOR, in place
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public Tensor logical_xor_(Tensor other)
            {
                var res = THSTensor_logical_xor_(handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logit(IntPtr tensor, IntPtr eps);

            /// <summary>
            /// Returns a new tensor with the logit of the elements of input.
            /// input is clamped to [eps, 1 - eps] when eps is not null
            /// </summary>
            /// <param name="eps"></param>
            /// <returns></returns>
            public Tensor logit(double? eps = null)
            {
                var epsArr = eps.HasValue ? new double[] { eps.Value } : null;

                unsafe {
                    fixed (double* pEps = epsArr) {
                        var res = THSTensor_logit(handle, (IntPtr)pEps);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
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
            public Tensor mul(Tensor target)
            {
                var res = THSTensor_mul(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise multiplication
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor multiply(Tensor target) => mul(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mul_scalar(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Element-wise multiplication
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor mul(Scalar target)
            {
                var res = THSTensor_mul_scalar(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise multiplcation
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor multiply(Scalar target) => mul(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mul_(IntPtr tensor, IntPtr target);

            /// <summary>
            /// Element-wise multiplication, in place
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor mul_(Tensor target)
            {
                var res = THSTensor_mul_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mul_scalar_(IntPtr tensor, IntPtr target);

            /// <summary>
            /// Element-wise multiplication, in place
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor mul_(Scalar target)
            {
                var res = THSTensor_mul_scalar_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public static Tensor operator -(Tensor tensor)
            {
                return tensor.neg();
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_neg(IntPtr tensor);

            /// <summary>
            /// Negation
            /// </summary>
            /// <returns></returns>
            public Tensor neg()
            {
                var res = THSTensor_neg(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Negation
            /// </summary>
            /// <returns></returns>
            public Tensor negative() => neg();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_neg_(IntPtr tensor);

            /// <summary>
            /// In-place negation
            /// </summary>
            /// <returns></returns>
            public Tensor neg_()
            {
                var res = THSTensor_neg_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_pow(IntPtr tensor, IntPtr exponent);

            /// <summary>
            /// Takes the power of each element in input with exponent and returns a tensor with the result.
            /// </summary>
            /// <param name="exponent"></param>
            /// <returns></returns>
            public Tensor pow(Tensor exponent)
            {
                var res = THSTensor_pow(handle, exponent.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_pow_(IntPtr tensor, IntPtr exponent);

            /// <summary>
            /// Replaces each element in input with the power of the element and the exponent.
            /// </summary>
            /// <param name="exponent"></param>
            /// <returns></returns>
            public Tensor pow_(Tensor exponent)
            {
                var res = THSTensor_pow_(handle, exponent.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_pow_scalar(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Takes the power of each element in input with exponent and returns a tensor with the result.
            /// </summary>
            /// <param name="exponent"></param>
            /// <returns></returns>
            public Tensor pow(Scalar exponent)
            {
                var res = THSTensor_pow_scalar(handle, exponent.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_pow_scalar_(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Replaces each element in input with the power of the element and the exponent.
            /// </summary>
            /// <param name="exponent"></param>
            /// <returns></returns>
            public Tensor pow_(Scalar exponent)
            {
                var res = THSTensor_pow_scalar_(handle, exponent.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_reciprocal(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the reciprocal of the elements of input
            /// </summary>
            /// <returns></returns>
            public Tensor reciprocal()
            {
                var res = THSTensor_reciprocal(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_reciprocal_(IntPtr tensor);

            /// <summary>
            /// Replaces each element with the reciprocal of the input
            /// </summary>
            /// <returns></returns>
            public Tensor reciprocal_()
            {
                var res = THSTensor_reciprocal_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_remainder(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor remainder(Tensor target)
            {
                var res = THSTensor_remainder(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_remainder_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes the element-wise remainder of division, in place
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor remainder_(Tensor target)
            {
                var res = THSTensor_remainder_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_remainder_scalar(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="scalar"></param>
            /// <returns></returns>
            public Tensor remainder(Scalar scalar)
            {
                var res = THSTensor_remainder_scalar(handle, scalar.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_remainder_scalar_(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="scalar"></param>
            /// <returns></returns>
            public Tensor remainder_(Scalar scalar)
            {
                var res = THSTensor_remainder_scalar_(handle, scalar.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_round(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with each of the elements of input rounded to the closest integer.
            /// </summary>
            /// <returns></returns>
            public Tensor round()
            {
                var res = THSTensor_round(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_round_(IntPtr tensor);

            /// <summary>
            /// Replaces each of the elements of input with the element rounded to the closest integer.
            /// </summary>
            /// <returns></returns>
            public Tensor round_()
            {
                var res = THSTensor_round_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_rsqrt(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor rsqrt()
            {
                var res = THSTensor_rsqrt(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_rsqrt_(IntPtr tensor);

            /// <summary>
            /// Replaces each of the elements of input with  the reciprocal of the square-root of each of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor rsqrt_()
            {
                var res = THSTensor_rsqrt_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise square
            /// </summary>
            /// <returns></returns>
            public Tensor square() => pow(2);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sqrt(IntPtr tensor);

            /// <summary>
            /// Computes the element-wise square root
            /// </summary>
            /// <returns></returns>
            public Tensor sqrt()
            {
                var res = THSTensor_sqrt(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sqrt_(IntPtr tensor);

            /// <summary>
            /// Computes the element-wise square root, in place
            /// </summary>
            /// <returns></returns>
            public Tensor sqrt_()
            {
                var res = THSTensor_sqrt_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sign(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the signs (-1, 0, 1) of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sign()
            {
                var res = THSTensor_sign(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sign_(IntPtr tensor);

            /// <summary>
            /// Replaces each element with the signs (-1, 0, 1) of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sign_()
            {
                var res = THSTensor_sign_(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_signbit(IntPtr tensor);

            /// <summary>
            /// Tests if each element of input has its sign bit set (is less than zero) or not.
            /// </summary>
            /// <returns>A boolean tensor of the same shape as the input.</returns>
            public Tensor signbit()
            {
                var res = THSTensor_signbit(handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sub(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Element-wise subtraction
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor sub(Tensor target)
            {
                var res = THSTensor_sub(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sub_scalar(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Element-wise subtraction
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor sub(Scalar target)
            {
                var res = THSTensor_sub_scalar(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sub_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Element-wise subtraction, in place
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor sub_(Tensor target)
            {
                var res = THSTensor_sub_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sub_scalar_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Element-wise subtraction, in-place
            /// </summary>
            /// <param name="target"></param>
            /// <returns></returns>
            public Tensor sub_(Scalar target)
            {
                var res = THSTensor_sub_scalar_(handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_trunc(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the truncated integer values of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor trunc()
            {
                var res = THSTensor_trunc(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with the truncated integer values of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor fix() => trunc();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_trunc_(IntPtr tensor);

            /// <summary>
            /// Replaces each element with the truncated integer values of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor trunc_()
            {
                var res = THSTensor_trunc_(handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each element with the truncated integer values of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor fix_() => trunc_();

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_xlogy(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes x * log(y)
            /// </summary>
            /// <param name="y"></param>
            /// <returns></returns>
            public Tensor xlogy(Tensor y)
            {
                var res = THSTensor_xlogy(handle, y.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_xlogy_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes x * log(y) in place
            /// </summary>
            /// <param name="y"></param>
            /// <returns></returns>
            public Tensor xlogy_(Tensor y)
            {
                var res = THSTensor_xlogy_(handle, y.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_xlogy_scalar(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes x * log(y)
            /// </summary>
            /// <param name="y"></param>
            /// <returns></returns>
            public Tensor xlogy(Scalar y)
            {
                var res = THSTensor_xlogy_scalar(handle, y.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_xlogy_scalar_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes x * log(y) in place
            /// </summary>
            /// <param name="y"></param>
            /// <returns></returns>
            public Tensor xlogy_(Scalar y)
            {
                var res = THSTensor_xlogy_scalar_(handle, y.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            // Overloaded operators

            public static Tensor operator +(Tensor left, Tensor right)
            {
                return left.add(right);
            }

            public static Tensor operator +(Tensor left, Scalar right)
            {
                return left.add(right);
            }

            public static Tensor operator +(Scalar left, Tensor right)
            {
                return right.add(left);
            }

            public static Tensor operator *(Tensor left, Tensor right)
            {
                return left.mul(right);
            }

            public static Tensor operator *(Tensor left, Scalar right)
            {
                return left.mul(right);
            }

            public static Tensor operator *(Scalar left, Tensor right)
            {
                return right.mul(left);
            }

            public static Tensor operator -(Tensor left, Tensor right)
            {
                return left.sub(right);
            }

            public static Tensor operator -(Tensor left, Scalar right)
            {
                return left.sub(right);
            }

            public static Tensor operator -(Scalar left, Tensor right)
            {
                return right.negative().add(left);
            }

            public static Tensor operator /(Tensor left, Tensor right)
            {
                return left.div(right);
            }

            public static Tensor operator /(Tensor left, Scalar right)
            {
                return left.div(right);
            }

            public static Tensor operator /(Scalar left, Tensor right)
            {
                return right.reciprocal().mul(left);
            }

            public static Tensor operator %(Tensor left, Tensor right)
            {
                return left.remainder(right);
            }

            public static Tensor operator %(Tensor left, Scalar right)
            {
                return left.remainder(right);
            }

            public static Tensor operator &(Tensor left, Tensor right)
            {
                return left.bitwise_and(right);
            }

            public static Tensor operator |(Tensor left, Tensor right)
            {
                return left.bitwise_or(right);
            }

            public static Tensor operator ^(Tensor left, Tensor right)
            {
                return left.bitwise_xor(right);
            }

            public static Tensor operator ~(Tensor left)
            {
                return left.bitwise_not();
            }

        }

        // Duplication of tensor math opertors in the 'torch' namespace

        /// <summary>
        /// Compute the absolute value of each element in the tensor
        /// </summary>
        public static Tensor abs(Tensor input) => input.abs();

        /// <summary>
        /// Compute the absolute value of each element in the tensor
        /// </summary>
        public static Tensor absolute(Tensor input) => input.abs();

        /// <summary>
        /// Compute the absolute value of each element in the tensor, in-place
        /// </summary>
        public static Tensor abs_(Tensor input) => input.abs_();

        /// <summary>
        /// Compute the absolute value of each element in the tensor, in-place
        /// </summary>
        public static Tensor absolute_(Tensor input) => input.abs_();

        /// <summary>
        /// Add two tensors, element-wise
        /// </summary>
        /// <returns></returns>
        public static Tensor add(Tensor left, Tensor right) => left.add(right);

        /// <summary>
        /// Add a scalar value to each element in the target tensor.
        /// </summary>
        public static Tensor add(Tensor left, Scalar right) => left.add(right);

        /// <summary>
        /// Add two tensors, element-wise, scaling the second operator by 'alpha'
        /// </summary>
        public static Tensor add(Tensor left, Tensor right, Scalar scale) => left.add(right, scale);

        /// <summary>
        /// Add a scalar value to each element in the target tensor, scaled by 'alpha'
        /// </summary>
        public static Tensor add(Tensor left, Scalar right, Scalar scale) => left.add(right, scale);

        /// <summary>
        /// Add two tensors, element-wise, in place
        /// </summary>
        public static Tensor add_(Tensor left, Tensor right) => left.add_(right);

        /// <summary>
        /// Add a scalar value to each element in the target tensor, in place.
        /// </summary>
        public static Tensor add_(Tensor left, Scalar right) => left.add_(right);

        /// <summary>
        /// Add two tensors, element-wise, scaling the second operator by 'alpha', in place
        /// </summary>
        public static Tensor add_(Tensor left, Tensor right, Scalar scale) => left.add_(right, scale);

        /// <summary>
        /// Add a scalar value to each element in the target tensor, scaled by 'alpha', in place
        /// </summary>
        public static Tensor add_(Tensor left, Scalar right, Scalar scale) => left.add_(right, scale);

        /// <summary>
        /// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced
        /// add step (all matrix multiplications get accumulated along the first dimension).
        /// input is added to the final result.
        /// </summary>
        /// <returns></returns>
        public static Tensor addbmm(Tensor input, Tensor mat1, Tensor mat2, float beta = 1, float alpha = 1) => input.addbmm(mat1, mat2, beta, alpha);

        /// <summary>
        /// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced
        /// add step (all matrix multiplications get accumulated along the first dimension).
        /// input is added to the final result.
        /// In-place version of addbmm.
        /// </summary>
        /// <returns></returns>
        public static Tensor addbmm_(Tensor input, Tensor mat1, Tensor mat2, float beta = 1, float alpha = 1) => input.addbmm_(mat1, mat2, beta, alpha);

        /// <summary>
        /// Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// </summary>
        /// <returns></returns>
        public static Tensor addcdiv(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value) => input.addcdiv(tensor1, tensor2, value);

        /// <summary>
        /// Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// In-place version of addcdiv.
        /// </summary>
        /// <returns></returns>
        public static Tensor addcdiv_(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value) => input.addcdiv_(tensor1, tensor2, value);

        /// <summary>
        /// Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// </summary>
        /// <returns></returns>
        public static Tensor addcmul(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value) => input.addcmul(tensor1, tensor2, value);

        /// <summary>
        /// Performs the element-wise divismultiplicationion of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// In-place version of addcdiv.
        /// </summary>
        /// <returns></returns>
        public static Tensor addcmul_(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value) => input.addcmul_(tensor1, tensor2, value);

        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <returns></returns>
        public static Tensor addmm(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmm(mat1, mat2, beta, alpha);

        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <returns></returns>
        public static Tensor addmm_(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmm_(mat1, mat2, beta, alpha);

        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <returns></returns>
        public static Tensor addmv(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmv(mat1, mat2, beta, alpha);

        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <returns></returns>
        public static Tensor addmv_(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmv_(mat1, mat2, beta, alpha);

        /// <summary>
        /// Performs the outer-product of vectors vec1 and vec2 and adds it to the input tensor.
        /// </summary>
        /// <returns></returns>
        public static Tensor addr(Tensor input, Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f) => input.addr(vec1, vec2, beta, alpha);

        /// <summary>
        /// Performs the outer-product of vectors vec1 and vec2 and adds it to the input tensor.
        /// </summary>
        /// <returns></returns>
        public static Tensor addr_(Tensor input, Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f) => input.addr_(vec1, vec2, beta, alpha);

        public static Tensor bincount(Tensor input, Tensor weights = null, long minlength = 0) => input.bincount(weights, minlength);

        /// <summary>
        /// Element-wise bitwise AND
        /// </summary>
        /// <returns></returns>
        public static Tensor bitwise_and(Tensor left, Tensor right) => left.bitwise_and(right);

        /// <summary>
        /// Element-wise bitwise AND, in place.
        /// </summary>
        /// <returns></returns>
        public static Tensor bitwise_and_(Tensor left, Tensor right) => left.bitwise_and_(right);

        /// <summary>
        /// Element-wise bitwise NOT
        /// </summary>
        /// <returns></returns>
        public static Tensor bitwise_not(Tensor input) => input.bitwise_not();

        /// <summary>
        /// Element-wise bitwise NOT, in place.
        /// </summary>
        /// <returns></returns>
        public static Tensor bitwise_not_(Tensor input) => input.bitwise_not_();

        /// <summary>
        /// Element-wise bitwise OR
        /// </summary>
        /// <returns></returns>
        public static Tensor bitwise_or(Tensor left, Tensor right) => left.bitwise_or(right);

        /// <summary>
        /// Element-wise bitwiseXOR, in place.
        /// </summary>
        /// <returns></returns>
        public static Tensor bitwise_or_(Tensor left, Tensor right) => left.bitwise_or_(right);

        /// <summary>
        /// Element-wise bitwise XOR
        /// </summary>
        /// <returns></returns>
        public static Tensor bitwise_xor(Tensor left, Tensor right) => left.bitwise_xor(right);

        /// <summary>
        /// Element-wise bitwise XOR, in place.
        /// </summary>
        /// <returns></returns>
        public static Tensor bitwise_xor_(Tensor left, Tensor right) => left.bitwise_xor_(right);

        /// <summary>
        /// Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.
        /// </summary>
        /// <returns></returns>
        public static Tensor ceil(Tensor input) => input.ceil();

        /// <summary>
        /// Replaces each element of the input with the smallest integer greater than or equal to the element.
        /// </summary>
        /// <returns></returns>
        public static Tensor ceil_(Tensor input) => input.ceil_();

        public static Tensor cumsum(Tensor input, long dimension, ScalarType? type = null) => input.cumsum(dimension, type);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor div(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor divide(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <returns></returns>
        public static Tensor div(Tensor left, Scalar right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <returns></returns>
        public static Tensor divide(Tensor left, Scalar right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor div_(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div_(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor divide_(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div_(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor div_(Tensor left, Scalar right, RoundingMode rounding_mode = RoundingMode.None) => left.div_(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor divide_(Tensor left, Scalar right) => left.div_(right);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_einsum([MarshalAs(UnmanagedType.LPStr)] string location, IntPtr tensors, int len);

        /// <summary>
        /// Sums the product of the elements of the input operands along dimensions specified using a notation based on the Einstein summation convention.
        /// </summary>
        /// <param name="equation">The subscripts for the Einstein summation.</param>
        /// <param name="tensors">The operands to compute the Einstein sum of.</param>
        /// <remarks>
        /// Einsum allows computing many common multi-dimensional linear algebraic array operations by representing them in a short-hand format based on the
        /// Einstein summation convention, given by equation.The details of this format are described below, but the general idea is to label every dimension
        /// of the input operands with some subscript and define which subscripts are part of the output. The output is then computed by summing the product
        /// of the elements of the operands along the dimensions whose subscripts are not part of the output.For example, matrix multiplication can be computed
        /// using einsum as torch.einsum(“ij,jk->ik”, A, B). Here, j is the summation subscript and i and k the output subscripts(see section below for more details on why).
        /// </remarks>
        /// <returns></returns>
        public static Tensor einsum(string equation, params Tensor[] tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_einsum(equation, tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        /// <summary>
        /// Returns a new tensor with the exponential of the elements of the input tensor input.
        /// </summary>
        /// <returns></returns>
        public static Tensor exp(Tensor input) => input.exp();

        /// <summary>
        /// Replaces each element of the input with the exponential of the elements of the input tensor input.
        /// </summary>
        /// <returns></returns>
        public static Tensor exp_(Tensor input) => input.exp_();

        /// <summary>
        /// Computes the base 2 exponential function of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor exp2(Tensor input) => input.exp2();

        /// <summary>
        /// Returns a new tensor with the exponential of the elements minus 1 of input.
        /// </summary>
        public static Tensor expm1(Tensor input) => input.expm1();

        /// <summary>
        /// Replaces each element with the exponential of the element minus 1 of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor expm1_(Tensor input) => input.expm1_();

        /// <summary>
        /// Raises input to the power of exponent, elementwise, in double precision.
        /// </summary>
        /// <returns></returns>
        /// <remarks> If neither input is complex returns a torch.float64 tensor, and if one or more inputs is complex returns a torch.complex128 tensor.</remarks>
        public static Tensor float_power(Tensor input, Tensor target) => input.float_power(target);

        /// <summary>
        /// Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.
        /// </summary>
        /// <returns></returns>
        public static Tensor floor(Tensor input) => input.floor();

        /// <summary>
        /// Replaces each element with the floor of the input, the largest integer less than or equal to each element.
        /// </summary>
        /// <returns></returns>
        public static Tensor floor_(Tensor input) => input.exp_();

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        public static Tensor fmod(Tensor left, Tensor right) => left.fmod(right);

        /// <summary>
        /// Computes the element-wise remainder of division, in place.
        /// </summary>
        public static Tensor fmod_(Tensor left, Tensor right) => left.fmod_(right);

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        public static Tensor fmod(Tensor left, Scalar right) => left.fmod(right);

        /// <summary>
        /// Computes the element-wise remainder of division, in place.
        /// </summary>
        public static Tensor fmod_(Tensor left, Scalar right) => left.fmod_(right);

        /// <summary>
        /// Computes the fractional portion of each element in input.
        /// </summary>
        public static Tensor frac(Tensor input) => input.frac();

        /// <summary>
        /// Computes the fractional portion of each element in input, in-place.
        /// </summary>
        public static Tensor frac_(Tensor input) => input.frac_();

        /// <summary>
        /// Decomposes input into mantissa and exponent tensors 
        /// </summary>
        public static (Tensor Mantissa, Tensor Exponent) frexp(Tensor input) => input.frexp();

        public static Tensor gcd(Tensor left, Tensor right) => left.gcd(right);

        public static Tensor gcd_(Tensor left, Tensor right) => left.gcd_(right);

        public static Tensor histc(Tensor input, long bins = 100, long min = 0, long max = 0) => input.histc(bins, min, max);

        /// <summary>
        /// Element-wise: given the legs of a right triangle, return its hypotenuse.
        /// </summary>
        public static Tensor hypot(Tensor left, Tensor right) => left.hypot(right);

        /// <summary>
        /// Returns a new tensor with the natural logarithm of the input elements.
        /// </summary>
        public static Tensor log(Tensor input) => input.log();

        /// <summary>
        /// Replaces each elements with the natural logarithm of the input.
        /// </summary>
        public static Tensor log_(Tensor input) => input.log_();

        /// <summary>
        /// Returns a new tensor with the natural logarithm of (1 + input).
        /// </summary>
        public static Tensor log1p(Tensor input) => input.log1p();

        /// <summary>
        /// Replaces each elements with the natural logarithm of (1 + input), in place.
        /// </summary>
        public static Tensor log1p_(Tensor input) => input.log1p_();

        /// <summary>
        /// Logarithm of the sum of exponentiations of the inputs.
        /// </summary>
        public static Tensor logaddexp(Tensor left, Tensor right) => left.logaddexp(right);

        /// <summary>
        /// Logarithm of the sum of exponentiations of the inputs in base-2.
        /// </summary>
        public static Tensor logaddexp2(Tensor left, Tensor right) => left.logaddexp2(right);

        /// <summary>
        /// Returns the logarithm of the cumulative summation of the exponentiation of elements of input in the dimension dim.
        /// </summary>
        public static Tensor logcumsumexp(Tensor input, long dim) => input.logcumsumexp(dim);

        /// <summary>
        /// Returns the log of summed exponentials of each row of the input tensor in the given dimension dim. 
        /// </summary>
        public static Tensor logsumexp(Tensor input, long dim, Boolean keepdim = false) => input.logsumexp(dim, keepdim);

        /// <summary>
        /// Returns a new tensorwith the logarithm to the base 10 of the elements of input.
        /// </summary>
        public static Tensor log10(Tensor input) => input.log();

        /// <summary>
        /// Replaces each elements with the logarithm to the base 10 of the elements of input.
        /// </summary>
        public static Tensor log10_(Tensor input) => input.log_();

        /// <summary>
        /// Returns a new tensorwith the logarithm to the base 10 of the elements of input.
        /// </summary>
        public static Tensor log2(Tensor input) => input.log2();

        /// <summary>
        /// Replaces each elements with the logarithm to the base 10 of the elements of input.
        /// </summary>
        public static Tensor log2_(Tensor input) => input.log2_();

        /// <summary>
        /// Element-wise logical AND
        /// </summary>
        /// <returns></returns>
        public static Tensor logical_and(Tensor left, Tensor right) => left.logical_and(right);

        /// <summary>
        /// Element-wise logical AND, in place.
        /// </summary>
        /// <returns></returns>
        public static Tensor logical_and_(Tensor left, Tensor right) => left.logical_and_(right);

        /// <summary>
        /// Element-wise logical NOT
        /// </summary>
        /// <returns></returns>
        public static Tensor logical_not(Tensor input) => input.logical_not();

        /// <summary>
        /// Element-wise logical OR
        /// </summary>
        /// <returns></returns>
        public static Tensor logical_or(Tensor left, Tensor right) => left.logical_or(right);

        /// <summary>
        /// Element-wise logicalXOR, in place.
        /// </summary>
        /// <returns></returns>
        public static Tensor logical_or_(Tensor left, Tensor right) => left.logical_or_(right);

        /// <summary>
        /// Element-wise logical XOR
        /// </summary>
        /// <returns></returns>
        public static Tensor logical_xor(Tensor left, Tensor right) => left.logical_xor(right);

        /// <summary>
        /// Returns a new tensor with the logit of the elements of input.
        /// input is clamped to [eps, 1 - eps] when eps is not null
        /// </summary>
        /// <returns></returns>
        public static Tensor logit(Tensor input, double? eps = null) => input.logit(eps);

        public static Tensor max(Tensor input) => input.max();

        static public Tensor max(Tensor input, Tensor other) => input.max(other);

        static public (Tensor values, Tensor indexes) max(Tensor input, long dimension, bool keepDim = false) => input.max(dimension, keepDim);

        public static Tensor mean(Tensor input) => input.mean();

        public static Tensor mean(Tensor input, long[] dimensions, bool keepDimension = false, ScalarType? type = null) => input.mean(dimensions, keepDimension, type);

        public static Tensor min(Tensor input) => input.min();

        static public Tensor min(Tensor input, Tensor other) => input.min(other);

        static public (Tensor values, Tensor indexes) min(Tensor input, long dimension, bool keepDim = false) => input.min(dimension, keepDim);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor mul(Tensor left, Tensor right) => left.mul(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor multiply(Tensor left, Tensor right) => left.mul(right);

        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <returns></returns>
        public static Tensor mul(Tensor left, Scalar right) => left.mul(right);

        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <returns></returns>
        public static Tensor multiply(Tensor left, Scalar right) => left.mul(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor mul_(Tensor left, Tensor right) => left.mul_(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor multiply_(Tensor left, Tensor right) => left.mul_(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor mul_(Tensor left, Scalar right) => left.mul_(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <returns></returns>
        public static Tensor multiply_(Tensor left, Scalar right) => left.mul_(right);

        /// <summary>
        /// Negation
        /// </summary>
        /// <returns></returns>
        public static Tensor neg(Tensor input) => input.neg();

        /// <summary>
        /// Negation
        /// </summary>
        /// <returns></returns>
        public static Tensor negative(Tensor input) => input.neg();

        /// <summary>
        /// In-place negation
        /// </summary>
        /// <returns></returns>
        public static Tensor neg_(Tensor input) => input.neg_();

        /// <summary>
        /// In-place negation
        /// </summary>
        /// <returns></returns>
        public static Tensor negative_(Tensor input) => input.neg_();

        /// <summary>
        /// Takes the power of each element in input with exponent and returns a tensor with the result.
        /// </summary>
        /// <returns></returns>
        public static Tensor pow(Tensor left, Tensor exponent) => left.pow(exponent);

        /// <summary>
        /// Takes the power of each element in input with exponent and returns a tensor with the result.
        /// </summary>
        /// <returns></returns>
        public static Tensor pow(Tensor left, Scalar exponent) => left.pow(exponent);

        /// <summary>
        /// Replaces each element in input with the power of the element and the exponent.
        /// </summary>
        public static Tensor pow_(Tensor left, Tensor exponent) => left.pow_(exponent);

        /// <summary>
        /// Replaces each element in input with the power of the element and the exponent.
        /// </summary>
        public static Tensor pow_(Tensor left, Scalar exponent) => left.pow_(exponent);

        /// <summary>
        /// Returns a new tensor with the reciprocal of the elements of input
        /// </summary>
        /// <returns></returns>
        public static Tensor reciprocal(Tensor input) => input.reciprocal();

        /// <summary>
        /// Replaces each element with the reciprocal of the input
        /// </summary>
        /// <returns></returns>
        public static Tensor reciprocal_(Tensor input) => input.reciprocal_();

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <returns></returns>
        public static Tensor remainder(Tensor left, Tensor right) => left.remainder(right);

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <returns></returns>
        public static Tensor remainder(Tensor left, Scalar right) => left.remainder(right);

        /// <summary>
        /// Computes the element-wise remainder of division, in place
        /// </summary>
        public static Tensor remainder_(Tensor left, Tensor right) => left.remainder_(right);

        /// <summary>
        /// Computes the element-wise remainder of division, in place
        /// </summary>
        public static Tensor remainder_(Tensor left, Scalar right) => left.remainder_(right);

        /// <summary>
        /// Returns a new tensor with each of the elements of input rounded to the closest integer.
        /// </summary>
        public static Tensor round(Tensor input) => input.round();

        /// <summary>
        /// Replaces each of the elements of input with the element rounded to the closest integer.
        /// </summary>
        public static Tensor round_(Tensor input) => input.round_();

        /// <summary>
        /// Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
        /// </summary>
        public static Tensor rsqrt(Tensor input) => input.rsqrt();

        /// <summary>
        /// Replaces each of the elements of input with  the reciprocal of the square-root of each of the elements of input.
        /// </summary>
        public static Tensor rsqrt_(Tensor input) => input.rsqrt_();

        /// <summary>
        /// Computes the element-wise square
        /// </summary>
        /// <returns></returns>
        public static Tensor square(Tensor input) => input.pow(2);

        /// <summary>
        /// Computes the element-wise square root
        /// </summary>
        public static Tensor sqrt(Tensor input) => input.sqrt();

        /// <summary>
        /// Computes the element-wise square root, in place
        /// </summary>
        public static Tensor sqrt_(Tensor input) => input.sqrt_();

        /// <summary>
        /// Returns a new tensor with the signs (-1, 0, 1) of the elements of input.
        /// </summary>
        public static Tensor sign(Tensor input) => input.sign();

        /// <summary>
        /// Replaces each element with the signs (-1, 0, 1) of the elements of input.
        /// </summary>
        public static Tensor sign_(Tensor input) => input.sign_();

        /// <summary>
        /// Tests whether each element of input has its sign bit set (is less than zero) or not.
        /// </summary>
        /// <returns>A boolean tensor of the same shape as the input.</returns>
        public static Tensor signbit(Tensor input) => input.signbit();

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <returns></returns>
        public static Tensor sub(Tensor left, Tensor right) => left.sub(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        public static Tensor sub(Tensor left, Scalar right) => left.sub(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <returns></returns>
        public static Tensor subtract(Tensor left, Tensor right) => left.sub(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        public static Tensor subtract(Tensor left, Scalar right) => left.sub(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <returns></returns>
        public static Tensor sub_(Tensor left, Tensor right) => left.sub_(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        public static Tensor sub_(Tensor left, Scalar right) => left.sub_(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <returns></returns>
        public static Tensor subtract_(Tensor left, Tensor right) => left.sub_(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        public static Tensor subtract_(Tensor left, Scalar right) => left.sub_(right);

        /// <summary>
        /// Returns a new tensor with the truncated integer values of the elements of input.
        /// </summary>
        public static Tensor trunc(Tensor input) => input.trunc();

        /// <summary>
        /// Replaces each element with the truncated integer values of the elements of input.
        /// </summary>
        public static Tensor trunc_(Tensor input) => input.trunc_();

        /// <summary>
        /// Returns a new tensor with the truncated integer values of the elements of input.
        /// </summary>
        public static Tensor fix(Tensor input) => input.fix();

        /// <summary>
        /// Replaces each element with the truncated integer values of the elements of input.
        /// </summary>
        public static Tensor fix_(Tensor input) => input.fix_();

        /// <summary>
        /// Computes x * log(y)
        /// </summary>
        /// <returns></returns>
        public static Tensor xlogy(Tensor left, Tensor right) => left.xlogy(right);

        /// <summary>
        /// Computes x * log(y)
        /// </summary>
        public static Tensor xlogy(Tensor left, Scalar right) => left.xlogy(right);

        /// <summary>
        /// Computes x * log(y) in place
        /// </summary>
        /// <returns></returns>
        public static Tensor xlogy_(Tensor left, Tensor right) => left.xlogy_(right);

        /// <summary>
        /// Computes x * log(y) in place
        /// </summary>
        public static Tensor xlogy_(Tensor left, Scalar right) => left.xlogy_(right);


        // Duplication of random distribution opertors in the 'torch' namespace

        public static Tensor rand_out(Tensor input, params long[] sizes) => input.randn_out(sizes);

        public static Tensor randint_out(Tensor input, long high, long[] sizes) => input.randint_out(high, sizes);

        public static Tensor rand_like(Tensor input, ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false) => input.rand_like(dtype, device, requiresGrad);

        public static Tensor randn_like(Tensor input, ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false) => input.randn_like(dtype, device, requiresGrad);

        public static Tensor randint_like(Tensor input, long low, long high, ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false) => input.randint_like(low, high, dtype, device, requiresGrad);

        public static Tensor randperm_out(Tensor input, long n) => input.randperm_out(n);

        public static Tensor bernoulli(Tensor input, torch.Generator generator = null) => input.bernoulli(generator);

        public static Tensor poisson(Tensor input, torch.Generator generator = null) => input.poisson(generator);

        public static Tensor multinomial(Tensor input, long num_samples, bool replacement = false, torch.Generator generator = null) => input.multinomial(num_samples, replacement, generator);
    }
}
