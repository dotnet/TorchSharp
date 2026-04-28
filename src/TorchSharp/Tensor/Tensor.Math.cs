// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using static TorchSharp.PInvoke.NativeMethods;

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
        public partial class Tensor
        {
            /// <summary>
            /// Compute the absolute value of each element in the tensor
            /// </summary>
            public Tensor abs()
            {
                var res = THSTensor_abs(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Compute the absolute value of each element in the tensor
            /// </summary>
            public Tensor absolute() => abs();

            /// <summary>
            /// Compute the absolute value of each element in the tensor, in-place.
            /// </summary>
            public Tensor abs_()
            {
                THSTensor_abs_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Compute the absolute value of each element in the tensor, in-place
            /// </summary>
            public Tensor absolute_() => abs_();

            /// <summary>
            /// Add two tensors, element-wise
            /// </summary>
            /// <param name="target">The right-hand-side operand.</param>
            /// <returns></returns>
            public Tensor add(Tensor target)
            {
                return add(target, 1);
            }

            /// <summary>
            /// Add two tensors, element-wise, scaling the second operator by 'alpha'
            /// </summary>
            /// <param name="target">The right-hand-side operand.</param>
            /// <param name="alpha">The RHS scale factor</param>
            /// <returns></returns>
            public Tensor add(Tensor target, Scalar alpha)
            {
                var res = THSTensor_add(Handle, target.Handle, alpha.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Add a scalar value to each element in the target tensor.
            /// </summary>
            /// <param name="scalar">The right-hand-side operand.</param>
            /// <returns></returns>
            public Tensor add(Scalar scalar)
            {
                return add(scalar, 1);
            }

            /// <summary>
            /// Add a scalar value to each element in the target tensor, scaled by 'alpha'
            /// </summary>
            /// <param name="scalar">The right-hand-side operand.</param>
            /// <param name="alpha">The RHS scale factor</param>
            /// <returns></returns>
            public Tensor add(Scalar scalar, Scalar alpha)
            {
                var res = THSTensor_add_scalar(Handle, scalar.Handle, alpha.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// In-place element-wise addition.
            /// </summary>
            /// <param name="target">The right-hand operand.</param>
            /// <returns></returns>
            public Tensor add_(Tensor target)
            {
                return add_(target, 1);
            }

            /// <summary>
            /// In-place element-wise addition, with scaling
            /// </summary>
            /// <param name="target">The right-hand operand.</param>
            /// <param name="alpha">Scale factor for the right-hand operand.</param>
            /// <returns></returns>
            public Tensor add_(Tensor target, Scalar alpha)
            {
                THSTensor_add_(Handle, target.Handle, alpha.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// In-place scalar addition.
            /// </summary>
            /// <param name="scalar">The right-hand operand.</param>
            /// <returns></returns>
            public Tensor add_(Scalar scalar)
            {
                return add_(scalar, 1);
            }

            /// <summary>
            /// In-place scalar addition, scaled.
            /// </summary>
            /// <param name="scalar">The right-hand operand.</param>
            /// <param name="alpha">Scale factor for the right-hand operand.</param>
            /// <returns></returns>
            public Tensor add_(Scalar scalar, Scalar alpha)
            {
                THSTensor_add_scalar_(Handle, scalar.Handle, alpha.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced
            /// add step (all matrix multiplications get accumulated along the first dimension).
            /// input is added to the final result.
            /// </summary>
            /// <param name="batch1">The first batch of matrices to be multiplied</param>
            /// <param name="batch2">The second batch of matrices to be multiplied</param>
            /// <param name="beta">Nultiplier for input (β)</param>
            /// <param name="alpha">Multiplier for batch1 @ batch2 (α)</param>
            /// <returns></returns>
            public Tensor addbmm(Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1)
            {
                var res = THSTensor_addbmm(Handle, batch1.Handle, batch2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Performs an in-place batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced
            /// add step (all matrix multiplications get accumulated along the first dimension).
            /// input is added to the final result.
            /// </summary>
            /// <param name="batch1">The first batch of matrices to be multiplied</param>
            /// <param name="batch2">The second batch of matrices to be multiplied</param>
            /// <param name="beta">Nultiplier for input (β)</param>
            /// <param name="alpha">Multiplier for batch1 @ batch2 (α)</param>
            /// <returns></returns>
            public Tensor addbmm_(Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1)
            {
                THSTensor_addbmm_(Handle, batch1.Handle, batch2.Handle, beta, alpha);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
            /// </summary>
            /// <param name="tensor1">First tensor</param>
            /// <param name="tensor2">Second tensor</param>
            /// <param name="value">Scale factor</param>
            /// <returns></returns>
            public Tensor addcdiv(Tensor tensor1, Tensor tensor2, Scalar value)
            {
                var res = THSTensor_addcdiv(Handle, tensor1.Handle, tensor2.Handle, value.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Perform the element-wise division of tensor1 by tensor2 and add it to input.
            /// </summary>
            /// <param name="tensor1">First tensor</param>
            /// <param name="tensor2">Second tensor</param>
            /// <returns></returns>
            public Tensor addcdiv(Tensor tensor1, Tensor tensor2)
            {
                return addcdiv(tensor1, tensor2, 1);
            }

            /// <summary>
            /// Performs the in-place element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
            /// </summary>
            /// <param name="tensor1">First tensor</param>
            /// <param name="tensor2">Second tensor</param>
            /// <param name="value">Scale factor</param>
            /// <returns></returns>
            public Tensor addcdiv_(Tensor tensor1, Tensor tensor2, Scalar value)
            {
                THSTensor_addcdiv_(Handle, tensor1.Handle, tensor2.Handle, value.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Performs the in-place element-wise division of tensor1 by tensor2 and add it to input.
            /// </summary>
            /// <param name="tensor1">First tensor</param>
            /// <param name="tensor2">Second tensor</param>
            /// <returns></returns>
            public Tensor addcdiv_(Tensor tensor1, Tensor tensor2)
            {
                return addcdiv_(tensor1, tensor2, 1);
            }

            /// <summary>
            /// Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
            /// </summary>
            /// <param name="tensor1">First tensor</param>
            /// <param name="tensor2">Second tensor</param>
            /// <param name="value">Scale factor</param>
            /// <returns></returns>
            public Tensor addcmul(Tensor tensor1, Tensor tensor2, Scalar value)
            {
                var res = THSTensor_addcmul(Handle, tensor1.Handle, tensor2.Handle, value.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Performs the in-place element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
            /// </summary>
            /// <param name="tensor1">First tensor</param>
            /// <param name="tensor2">Second tensor</param>
            /// <param name="value">Scale factor</param>
            /// <returns></returns>
            public Tensor addcmul_(Tensor tensor1, Tensor tensor2, Scalar value)
            {
                THSTensor_addcmul_(Handle, tensor1.Handle, tensor2.Handle, value.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
            /// </summary>
            /// <param name="mat1">First matrix</param>
            /// <param name="mat2">Second matrix</param>
            /// <param name="beta">Input scale factor</param>
            /// <param name="alpha">Matrix multiplication scale factor</param>
            /// <returns></returns>
            public Tensor addmm(Tensor mat1, Tensor mat2, float beta = 1, float alpha = 1)
            {
                var res = THSTensor_addmm(Handle, mat1.Handle, mat2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Performs an in-place matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
            /// </summary>
            /// <param name="mat1">First matrix</param>
            /// <param name="mat2">Second matrix</param>
            /// <param name="beta">Input scale factor</param>
            /// <param name="alpha">Matrix multiplication scale factor</param>
            /// <returns></returns>
            public Tensor addmm_(Tensor mat1, Tensor mat2, float beta = 1, float alpha = 1)
            {
                THSTensor_addmm_(Handle, mat1.Handle, mat2.Handle, beta, alpha);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Performs a matrix-vector product of the matrix mat and the vector vec. The vector input is added to the final result.
            /// </summary>
            /// <param name="mat">Matrix to be matrix multiplied</param>
            /// <param name="vec">Vector to be matrix multiplied</param>
            /// <param name="beta">Input scale factor</param>
            /// <param name="alpha">Matrix multiplication scale factor</param>
            /// <returns></returns>
            public Tensor addmv(Tensor mat, Tensor vec, float beta = 1.0f, float alpha = 1.0f)
            {
                var res = THSTensor_addmv(Handle, mat.Handle, vec.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Performs a matrix-vector product of the matrix mat and the vector vec. The vector input is added to the final result.
            /// </summary>
            /// <param name="mat">Matrix to be matrix multiplied</param>
            /// <param name="vec">Vector to be matrix multiplied</param>
            /// <param name="beta">Input scale factor</param>
            /// <param name="alpha">Matrix multiplication scale factor</param>
            /// <returns></returns>
            public Tensor addmv_(Tensor mat, Tensor vec, float beta = 1.0f, float alpha = 1.0f)
            {
                THSTensor_addmv_(Handle, mat.Handle, vec.Handle, beta, alpha);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Performs the outer-product of vectors vec1 and vec2 and adds it to the input tensor.
            /// </summary>
            /// <param name="vec1">The first vector of the outer product</param>
            /// <param name="vec2">The second vector of the outer product</param>
            /// <param name="beta">Input scale factor</param>
            /// <param name="alpha">Outer-product scale factor</param>
            /// <returns></returns>
            public Tensor addr(Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f)
            {
                var res = THSTensor_addr(Handle, vec1.Handle, vec2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Performs the outer-product of vectors vec1 and vec2 and adds it to the input tensor.
            ///
            /// In-place version of 'addr'
            /// </summary>
            /// <param name="vec1">The first vector of the outer product</param>
            /// <param name="vec2">The second vector of the outer product</param>
            /// <param name="beta">Input scale factor</param>
            /// <param name="alpha">Outer-product scale factor</param>
            /// <returns></returns>
            public Tensor addr_(Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f)
            {
                THSTensor_addr_(Handle, vec1.Handle, vec2.Handle, beta, alpha);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Element-wise bitwise AND
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_and(Tensor other)
            {
                var res = THSTensor_bitwise_and(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise bitwise AND, in-place
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_and_(Tensor other)
            {
                THSTensor_bitwise_and_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Element-wise bitwise NOT
            /// </summary>
            /// <returns></returns>
            public Tensor bitwise_not()
            {
                var res = THSTensor_bitwise_not(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise bitwise NOT, in-place
            /// </summary>
            /// <returns></returns>
            public Tensor bitwise_not_()
            {
                THSTensor_bitwise_not_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Element-wise bitwise OR
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_or(Tensor other)
            {
                var res = THSTensor_bitwise_or(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise bitwise OR, in-place
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_or_(Tensor other)
            {
                THSTensor_bitwise_or_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Element-wise bitwise XOR
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_xor(Tensor other)
            {
                var res = THSTensor_bitwise_xor(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise bitwise XOR, in-place.
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_xor_(Tensor other)
            {
                THSTensor_bitwise_xor_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Element-wise bitwise left_shift
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_left_shift(Tensor other)
            {
                var res = THSTensor_bitwise_left_shift(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise bitwise left_shift, in-place.
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_left_shift_(Tensor other)
            {
                THSTensor_bitwise_left_shift_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Element-wise bitwise right_shift
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_right_shift(Tensor other)
            {
                var res = THSTensor_bitwise_right_shift(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise bitwise right_shift, in-place.
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_right_shift_(Tensor other)
            {
                THSTensor_bitwise_right_shift_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.
            /// </summary>
            /// <returns></returns>
            public Tensor ceil()
            {
                var res = THSTensor_ceil(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each element of the input with the smallest integer greater than or equal to the element.
            /// </summary>
            /// <returns></returns>
            public Tensor ceil_()
            {
                THSTensor_ceil_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a view of input with a flipped conjugate bit. If input has a non-complex dtype, this function just returns input.
            /// </summary>
            /// <returns></returns>
            public Tensor conj()
            {
                var res = THSTensor_conj(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise conjugate of the given input tensor. If input has a non-complex dtype, this function just returns input.
            /// </summary>
            /// <returns></returns>
            public Tensor conj_physical()
            {
                var res = THSTensor_conj_physical(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// In-place version of conj_physical
            /// </summary>
            /// <returns></returns>
            public Tensor conj_physical_()
            {
                THSTensor_conj_physical_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns true if the input is a conjugated tensor, i.e. its conjugate bit is set to True.
            /// </summary>
            /// <returns></returns>
            public bool is_conj()
            {
                var res = THSTensor_is_conj(Handle);
                if (res == -1) CheckForErrors();
                return res != 0;
            }

            /// <summary>
            /// Returns a new tensor with materialized conjugation if input’s conjugate bit is set to True, else returns input.
            /// The output tensor will always have its conjugate bit set to False.
            /// </summary>
            /// <returns></returns>
            public Tensor resolve_conj()
            {
                var res = THSTensor_resolve_conj(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns true if the input's negative bit is set to True.
            /// </summary>
            public bool is_neg()
            {
                var res = THSTensor_is_neg(Handle);
                if (res == -1) CheckForErrors();
                return res != 0;
            }

            /// <summary>
            /// Returns a new tensor with materialized negation if input’s negative bit is set to True, else returns input.
            /// The output tensor will always have its negative bit set to False.
            /// </summary>
            /// <returns></returns>
            public Tensor resolve_neg()
            {
                var res = THSTensor_resolve_neg(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a tuple (values, indices) where values is the cumulative maximum of elements of input in the dimension dim.
            /// Indices is the index location of each maximum value found in the dimension dim.
            /// </summary>
            /// <param name="dim">The dimension to do the operation over</param>
            public (Tensor values, Tensor indexes) cummax(long dim)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_cummax(Handle, pa.CreateArray, dim);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            /// <summary>
            /// Returns a tuple (values, indices) where values is the cumulative minimum of elements of input in the dimension dim.
            /// Indices is the index location of each minimum value found in the dimension dim.
            /// </summary>
            /// <param name="dim">The dimension to do the operation over</param>
            public (Tensor values, Tensor indexes) cummin(long dim)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_cummin(Handle, pa.CreateArray, dim);
                    CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            /// <summary>
            /// Returns the cumulative sum of elements of input in the dimension dim.
            /// </summary>
            /// <param name="dim">The dimension to do the operation over</param>
            /// <param name="type">The desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed.
            /// This is useful for preventing data type overflows.</param>
            /// <returns></returns>
            public Tensor cumsum(long dim, ScalarType? type = null)
            {
                var res = THSTensor_cumsum(Handle, dim, type.HasValue, (sbyte)type.GetValueOrDefault());
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the cumulative product of elements of input in the dimension dim.
            /// </summary>
            /// <param name="dim">The dimension to do the operation over</param>
            /// <param name="type">The desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed.
            /// This is useful for preventing data type overflows.</param>
            /// <returns></returns>
            public Tensor cumprod(long dim, ScalarType? type = null)
            {
                var res = THSTensor_cumprod(Handle, dim, type.HasValue, (sbyte)type.GetValueOrDefault());
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Divides each element of the input by the corresponding element of other.
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor div(Tensor target, RoundingMode rounding_mode = RoundingMode.None)
            {
                var res = THSTensor_div(Handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Divides each element of the input by the corresponding element of other.
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor divide(Tensor target, RoundingMode rounding_mode = RoundingMode.None) => div(target, rounding_mode);

            /// <summary>
            /// Scalar division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor div(Scalar target, RoundingMode rounding_mode = RoundingMode.None)
            {
                var res = THSTensor_div_scalar(Handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Scalar division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor divide(Scalar target, RoundingMode rounding_mode = RoundingMode.None) => div(target, rounding_mode);

            /// <summary>
            /// In-place division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor div_(Tensor target, RoundingMode rounding_mode = RoundingMode.None)
            {
                THSTensor_div_(Handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// In-place division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor divide_(Tensor target, RoundingMode rounding_mode = RoundingMode.None) => div_(target, rounding_mode);

            /// <summary>
            /// In-place scalar division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor div_(Scalar target, RoundingMode rounding_mode = RoundingMode.None)
            {
                THSTensor_div_scalar_(Handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// In-place scalar division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor divide_(Scalar target, RoundingMode rounding_mode = RoundingMode.None) => div_(target, rounding_mode);

            /// <summary>
            /// Returns a new tensor with the exponential of the elements of the input tensor.
            /// </summary>
            public Tensor exp()
            {
                var res = THSTensor_exp(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces the tensor with the exponential of the elements of the input tensor.
            /// </summary>
            public Tensor exp_()
            {
                THSTensor_exp_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the base 2 exponential function of input.
            /// </summary>
            /// <returns></returns>
            public Tensor exp2()
            {
                var res = THSTensor_exp2(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the base 2 exponential function of input, in-place.
            /// </summary>
            /// <returns></returns>
            public Tensor exp2_()
            {
                THSTensor_exp2_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a new tensor with the exponential of the elements minus 1 of input.
            /// </summary>
            /// <returns></returns>
            public Tensor expm1()
            {
                var res = THSTensor_expm1(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each element with the exponential of the element minus 1 of input, in-place.
            /// </summary>
            /// <returns></returns>
            public Tensor expm1_()
            {
                THSTensor_expm1_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Raises input to the power of exponent, elementwise, in double precision.
            /// </summary>
            /// <param name="target">The exponent.</param>
            /// <returns></returns>
            /// <remarks> If neither input is complex returns a torch.float64 tensor, and if one or more inputs is complex returns a torch.complex128 tensor.</remarks>
            public Tensor float_power(Tensor target)
            {
                var res = THSTensor_float_power(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Raises input to the power of exponent, elementwise, in double precision, in-place.
            /// </summary>
            /// <param name="target">The exponent.</param>
            /// <returns></returns>
            public Tensor float_power_(Tensor target)
            {
                THSTensor_float_power_(Handle, target.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.
            /// </summary>
            /// <returns></returns>
            public Tensor floor()
            {
                var res = THSTensor_floor(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each element with the floor of the input, the largest integer less than or equal to each element.
            /// </summary>
            /// <returns></returns>
            public Tensor floor_()
            {
                THSTensor_floor_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes input divided by other, elementwise, and floors the result.
            /// </summary>
            /// <param name="other">the divisor</param>
            public Tensor floor_divide(Tensor other)
            {
                var res = THSTensor_floor_divide(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes input divided by other, elementwise, and floors the result.
            /// </summary>
            /// <param name="other">the divisor</param>
            public Tensor floor_divide(Scalar other)
            {
                var res = THSTensor_floor_divide_scalar(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes input divided by other, elementwise, and floors the result, computation done in place.
            /// </summary>
            /// <param name="other">the divisor</param>
            public Tensor floor_divide_(Tensor other)
            {
                THSTensor_floor_divide_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes input divided by other, elementwise, and floors the result, computation done in place.
            /// </summary>
            /// <param name="other">the divisor</param>
            public Tensor floor_divide_(Scalar other)
            {
                THSTensor_floor_divide_scalar_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <returns></returns>
            public Tensor fmod(Tensor target)
            {
                var res = THSTensor_fmod(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise remainder of division, in-place.
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <returns></returns>
            public Tensor fmod_(Tensor target)
            {
                THSTensor_fmod_(Handle, target.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="scalar">Denominator</param>
            /// <returns></returns>
            public Tensor fmod(Scalar scalar)
            {
                var res = THSTensor_fmod_scalar(Handle, scalar.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise remainder of division, in-place
            /// </summary>
            /// <param name="scalar">Denominator</param>
            /// <returns></returns>
            public Tensor fmod_(Scalar scalar)
            {
                THSTensor_fmod_scalar_(Handle, scalar.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the fractional portion of each element in input.
            /// </summary>
            /// <returns></returns>
            public Tensor frac()
            {
                var res = THSTensor_frac(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the fractional portion of each element in input, in-place.
            /// </summary>
            /// <returns></returns>
            public Tensor frac_()
            {
                THSTensor_frac_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Decomposes input into mantissa and exponent tensors
            /// </summary>
            /// <returns></returns>
            public (Tensor Mantissa, Tensor Exponent) frexp()
            {
                var mantissa = THSTensor_frexp(Handle, out var exponent);
                if (mantissa == IntPtr.Zero || exponent == IntPtr.Zero)
                    CheckForErrors();
                return (new Tensor(mantissa), new Tensor(exponent));
            }

            /// <summary>
            /// Computes the element-wise greatest common divisor (GCD) of input and other.
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            public Tensor gcd(Tensor other)
            {
                var res = THSTensor_gcd(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise greatest common divisor (GCD) of input and other.
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor gcd_(Tensor other)
            {
                THSTensor_gcd_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

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
                var res = THSTensor_histc(Handle, bins, min, max);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise: given the legs of a right triangle, return its hypotenuse.
            /// </summary>
            /// <param name="other">The second input tensor.</param>
            /// <returns></returns>
            public Tensor hypot(Tensor other)
            {
                var res = THSTensor_hypot(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with the natural logarithm of the input elements.
            /// </summary>
            /// <returns></returns>
            public Tensor log()
            {
                var res = THSTensor_log(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each elements with the natural logarithm of the input.
            /// </summary>
            public Tensor log_()
            {
                THSTensor_log_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Logarithm of the sum of exponentiations of the inputs.
            /// </summary>
            /// <param name="other">The second input tensor.</param>
            /// <returns></returns>
            public Tensor logaddexp(Tensor other)
            {
                var res = THSTensor_logaddexp(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Logarithm of the sum of exponentiations of the inputs in base-2.
            /// </summary>
            /// <param name="other">The second input tensor.</param>
            /// <returns></returns>
            public Tensor logaddexp2(Tensor other)
            {
                var res = THSTensor_logaddexp2(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the logarithm of the cumulative summation of the exponentiation of elements of input in the dimension dim.
            /// </summary>
            /// <param name="dim">The dimension to do the operation over</param>
            /// <returns></returns>
            public Tensor logcumsumexp(long dim)
            {
                var res = THSTensor_logcumsumexp(Handle, dim);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns the log of summed exponentials of each row of the input tensor in the given dimension dim.
            /// </summary>
            /// <param name="dim">The dimension to do the operation over</param>
            /// <param name="keepdim">Thether the output tensor has dim retained or not.</param>
            /// <returns></returns>
            /// <remarks>The computation is numerically stabilized.</remarks>
            public Tensor logsumexp(long dim, bool keepdim = false)
            {
                var res = THSTensor_logsumexp(Handle, dim, keepdim);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with the logit of the elements of input.
            /// </summary>
            /// <param name="dim">A dimension along which log_softmax will be computed.</param>
            /// <param name="dtype">The desired data type of returned tensor.</param>
            public Tensor log_softmax(long dim, ScalarType? dtype = null) => torch.special.log_softmax(this, dim, dtype);

            /// <summary>
            /// Returns a new tensor with the logarithm to the base 10 of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor log10()
            {
                var res = THSTensor_log10(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces elements with the logarithm to the base 10 of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor log10_()
            {
                THSTensor_log10_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a new tensor with the natural logarithm of (1 + input).
            /// </summary>
            /// <returns></returns>
            public Tensor log1p()
            {
                var res = THSTensor_log1p(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with the natural logarithm of (1 + input), inplace.
            /// </summary>
            /// <returns></returns>
            public Tensor log1p_()
            {
                THSTensor_log1p_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a new tensor with the logarithm to the base 2 of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor log2()
            {
                var res = THSTensor_log2(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each element with the logarithm to the base 2 of the input.
            /// </summary>
            /// <returns></returns>
            public Tensor log2_()
            {
                THSTensor_log2_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Logical AND
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_and(Tensor other)
            {
                var res = THSTensor_logical_and(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Logical AND, in place
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_and_(Tensor other)
            {
                THSTensor_logical_and_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Logical NOT
            /// </summary>
            /// <returns></returns>
            public Tensor logical_not()
            {
                var res = THSTensor_logical_not(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Logical NOT, in place
            /// </summary>
            /// <returns></returns>
            public Tensor logical_not_()
            {
                THSTensor_logical_not_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Logical OR
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_or(Tensor other)
            {
                var res = THSTensor_logical_or(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Logical OR, in place
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_or_(Tensor other)
            {
                THSTensor_logical_or_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Logical XOR
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_xor(Tensor other)
            {
                var res = THSTensor_logical_xor(Handle, other.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Logical XOR, in place
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_xor_(Tensor other)
            {
                THSTensor_logical_xor_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a new tensor with the logit of the elements of input.
            /// input is clamped to [eps, 1 - eps] when eps is not null
            /// </summary>
            /// <param name="eps">The epsilon for input clamp bound.</param>
            /// <returns></returns>
            public Tensor logit(double? eps = null)
            {
                var epsArr = eps.HasValue ? new double[] { eps.Value } : null;

                unsafe {
                    fixed (double* pEps = epsArr) {
                        var res = THSTensor_logit(Handle, (IntPtr)pEps);
                        if (res == IntPtr.Zero) { CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            /// <summary>
            /// Returns the logit of the elements of input, in-place.
            /// input is clamped to [eps, 1 - eps] when eps is not null
            /// </summary>
            /// <param name="eps">The epsilon for input clamp bound.</param>
            /// <returns></returns>
            public Tensor logit_(double? eps = null)
            {
                var epsArr = eps.HasValue ? new double[] { eps.Value } : null;

                unsafe {
                    fixed (double* pEps = epsArr) {
                        THSTensor_logit_(Handle, (IntPtr)pEps);
                        CheckForErrors();
                        return this;
                    }
                }
            }

            /// <summary>
            /// Element-wise multiplication
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor mul(Tensor target)
            {
                var res = THSTensor_mul(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise multiplication
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor multiply(Tensor target) => mul(target);

            /// <summary>
            /// Element-wise multiplication
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor mul(Scalar target)
            {
                var res = THSTensor_mul_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise multiplcation
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor multiply(Scalar target) => mul(target);

            /// <summary>
            /// Element-wise multiplication, in place
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor mul_(Tensor target)
            {
                THSTensor_mul_(Handle, target.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Element-wise multiplication, in place
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor mul_(Scalar target)
            {
                THSTensor_mul_scalar_(Handle, target.Handle);
                CheckForErrors();
                return this;
            }

            public static Tensor operator -(Tensor tensor)
            {
                return tensor.neg();
            }

            /// <summary>
            /// Negation
            /// </summary>
            /// <returns></returns>
            public Tensor neg()
            {
                var res = THSTensor_neg(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Negation
            /// </summary>
            /// <returns></returns>
            public Tensor negative() => neg();

            /// <summary>
            /// In-place negation
            /// </summary>
            /// <returns></returns>
            public Tensor neg_()
            {
                THSTensor_neg_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Takes the power of each element in input with exponent and returns a tensor with the result.
            /// </summary>
            /// <param name="exponent">The exponent.</param>
            /// <returns></returns>
            public Tensor pow(Tensor exponent)
            {
                var res = THSTensor_pow(Handle, exponent.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each element in input with the power of the element and the exponent.
            /// </summary>
            /// <param name="exponent">The exponent.</param>
            /// <returns></returns>
            public Tensor pow_(Tensor exponent)
            {
                THSTensor_pow_(Handle, exponent.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Takes the power of each element in input with exponent and returns a tensor with the result.
            /// </summary>
            /// <param name="exponent">The exponent.</param>
            /// <returns></returns>
            public Tensor pow(Scalar exponent)
            {
                var res = THSTensor_pow_scalar(Handle, exponent.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each element in input with the power of the element and the exponent.
            /// </summary>
            /// <param name="exponent">The exponent.</param>
            /// <returns></returns>
            public Tensor pow_(Scalar exponent)
            {
                THSTensor_pow_scalar_(Handle, exponent.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a new tensor with the reciprocal of the elements of input
            /// </summary>
            /// <returns></returns>
            public Tensor reciprocal()
            {
                var res = THSTensor_reciprocal(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each element with the reciprocal of the input
            /// </summary>
            /// <returns></returns>
            public Tensor reciprocal_()
            {
                THSTensor_reciprocal_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <returns></returns>
            public Tensor remainder(Tensor target)
            {
                var res = THSTensor_remainder(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise remainder of division, in place
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <returns></returns>
            public Tensor remainder_(Tensor target)
            {
                THSTensor_remainder_(Handle, target.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="scalar">Denominator</param>
            /// <returns></returns>
            public Tensor remainder(Scalar scalar)
            {
                var res = THSTensor_remainder_scalar(Handle, scalar.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="scalar">Denominator</param>
            /// <returns></returns>
            public Tensor remainder_(Scalar scalar)
            {
                THSTensor_remainder_scalar_(Handle, scalar.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a new tensor with each of the elements of input rounded to the closest value with the given number of decimals.
            /// </summary>
            /// <param name="decimals">Number of decimal places to round to (default: 0). If decimals is negative, it specifies the number of positions to the left of the decimal point.</param>
            /// <returns></returns>
            public Tensor round(long decimals = 0L)
            {
                var res = THSTensor_round(Handle, decimals);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each of the elements of input with the element rounded to the closest value with the given number of decimals.
            /// </summary>
            /// <param name="decimals">Number of decimal places to round to (default: 0). If decimals is negative, it specifies the number of positions to the left of the decimal point.</param>
            /// <returns></returns>
            public Tensor round_(long decimals = 0L)
            {
                THSTensor_round_(Handle, decimals);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor rsqrt()
            {
                var res = THSTensor_rsqrt(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each of the elements of input with  the reciprocal of the square-root of each of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor rsqrt_()
            {
                THSTensor_rsqrt_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes the element-wise square
            /// </summary>
            /// <returns></returns>
            public Tensor square() => pow(2);

            /// <summary>
            /// Computes the element-wise square root
            /// </summary>
            /// <returns></returns>
            public Tensor sqrt()
            {
                var res = THSTensor_sqrt(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the element-wise square root, in place
            /// </summary>
            /// <returns></returns>
            public Tensor sqrt_()
            {
                THSTensor_sqrt_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a new tensor with the signs (-1, 0, 1) of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sign()
            {
                var res = THSTensor_sign(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Replaces each element with the signs (-1, 0, 1) of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor sign_()
            {
                THSTensor_sign_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// This function is an extension of torch.sign() to complex tensors.
            /// It computes a new tensor whose elements have the same angles as the corresponding
            /// elements of input and absolute values (i.e. magnitudes) of one for complex tensors
            /// and is equivalent to torch.sign() for non-complex tensors.
            /// </summary>
            /// <returns></returns>
            public Tensor sgn()
            {
                var res = THSTensor_sgn(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// This function is an extension of torch.sign() to complex tensors.
            /// It computes a new tensor whose elements have the same angles as the corresponding
            /// elements of input and absolute values (i.e. magnitudes) of one for complex tensors
            /// and is equivalent to torch.sign() for non-complex tensors. In-place version.
            /// </summary>
            /// <returns></returns>
            public Tensor sgn_()
            {
                THSTensor_sgn_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Tests if each element of input has its sign bit set (is less than zero) or not.
            /// </summary>
            /// <returns>A boolean tensor of the same shape as the input.</returns>
            public Tensor signbit()
            {
                var res = THSTensor_signbit(Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise subtraction
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor sub(Tensor target)
            {
                var res = THSTensor_sub(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise subtraction
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor sub(Scalar target)
            {
                var res = THSTensor_sub_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Tensor subtract(Scalar target) => sub(target);
            public Tensor subtract(Tensor target) => sub(target);
            public Tensor subtract_(Scalar target) => sub_(target);
            public Tensor subtract_(Tensor target) => sub_(target);

            /// <summary>
            /// Element-wise subtraction, in place
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor sub_(Tensor target)
            {
                THSTensor_sub_(Handle, target.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Element-wise subtraction, in-place
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor sub_(Scalar target)
            {
                THSTensor_sub_scalar_(Handle, target.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Cumulatively computes the trapezoidal rule along dim. By default the spacing between elements is assumed to be 1,
            /// but dx can be used to specify a different constant spacing.
            /// </summary>
            /// <param name="dx">Constant spacing between values.</param>
            /// <param name="dim">The dimension along which to compute the trapezoidal rule. The last (inner-most) dimension by default.</param>
            /// <returns></returns>
            public Tensor cumulative_trapezoid(double dx = 1, long dim = -1)
            {
                IntPtr res = THSTensor_cumulative_trapezoid_dx(Handle, dx, dim);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Cumulatively computes the trapezoidal rule along dim. By default the spacing between elements is assumed to be 1,
            /// but x can be used to specify arbitrary spacing along dim.
            /// </summary>
            /// <param name="x">Defines spacing between values as specified above.</param>
            /// <param name="dim">The dimension along which to compute the trapezoidal rule. The last (inner-most) dimension by default.</param>
            /// <returns></returns>
            public Tensor cumulative_trapezoid(Tensor x, long dim = -1)
            {
                IntPtr res = THSTensor_cumulative_trapezoid_x(Handle, x.Handle, dim);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the trapezoidal rule along dim. By default the spacing between elements is assumed to be 1,
            /// but dx can be used to specify a different constant spacing.
            /// </summary>
            /// <param name="dx">Constant spacing between values.</param>
            /// <param name="dim">The dimension along which to compute the trapezoidal rule. The last (inner-most) dimension by default.</param>
            /// <returns></returns>
            public Tensor trapezoid(double dx = 1, long dim = -1)
            {
                IntPtr res = THSTensor_trapezoid_dx(Handle, dx, dim);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes the trapezoidal rule along dim. By default the spacing between elements is assumed to be 1,
            /// but x can be used to specify arbitrary spacing along dim.
            /// </summary>
            /// <param name="x">Defines spacing between values as specified above.</param>
            /// <param name="dim">The dimension along which to compute the trapezoidal rule. The last (inner-most) dimension by default.</param>
            /// <returns></returns>
            public Tensor trapezoid(Tensor x, long dim = -1)
            {
                IntPtr res = THSTensor_trapezoid_x(Handle, x.Handle, dim);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Computes input divided by other, elementwise, and floors the result.
            /// </summary>
            /// <param name="other">the divisor</param>
            public Tensor true_divide(Tensor other)
            {
                var res = THSTensor_true_divide(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes input divided by other, elementwise, and floors the result.
            /// </summary>
            /// <param name="other">the divisor</param>
            public Tensor true_divide(Scalar other)
            {
                var res = THSTensor_true_divide_scalar(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes input divided by other, elementwise, and floors the result, computation done in place.
            /// </summary>
            /// <param name="other">the divisor</param>
            public Tensor true_divide_(Tensor other)
            {
                THSTensor_true_divide_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes input divided by other, elementwise, and floors the result, computation done in place.
            /// </summary>
            /// <param name="other">the divisor</param>
            public Tensor true_divide_(Scalar other)
            {
                THSTensor_true_divide_scalar_(Handle, other.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Returns a new tensor with the truncated integer values of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor trunc()
            {
                var res = THSTensor_trunc(Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Returns a new tensor with the truncated integer values of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor fix() => trunc();

            /// <summary>
            /// Replaces each element with the truncated integer values of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor trunc_()
            {
                THSTensor_trunc_(Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Replaces each element with the truncated integer values of the elements of input.
            /// </summary>
            /// <returns></returns>
            public Tensor fix_() => trunc_();

            /// <summary>
            /// Computes x * log(y)
            /// </summary>
            /// <param name="y">The 'y' operand.</param>
            /// <returns></returns>
            public Tensor xlogy(Tensor y)
            {
                var res = THSTensor_xlogy(Handle, y.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes x * log(y) in place
            /// </summary>
            /// <param name="y">The 'y' operand.</param>
            /// <returns></returns>
            public Tensor xlogy_(Tensor y)
            {
                THSTensor_xlogy_(Handle, y.Handle);
                CheckForErrors();
                return this;
            }

            /// <summary>
            /// Computes x * log(y)
            /// </summary>
            /// <param name="y">The 'y' operand.</param>
            /// <returns></returns>
            public Tensor xlogy(Scalar y)
            {
                var res = THSTensor_xlogy_scalar(Handle, y.Handle);
                if (res == IntPtr.Zero)
                    CheckForErrors();
                return new Tensor(res);
            }

            /// <summary>
            /// Computes x * log(y) in place
            /// </summary>
            /// <param name="y">The 'y' operand.</param>
            /// <returns></returns>
            public Tensor xlogy_(Scalar y)
            {
                THSTensor_xlogy_scalar_(Handle, y.Handle);
                CheckForErrors();
                return this;
            }
        }
    }
}
