// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Runtime.InteropServices;

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
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_abs(IntPtr tensor);

            /// <summary>
            /// Compute the absolute value of each element in the tensor
            /// </summary>
            public Tensor abs()
            {
                var res = THSTensor_abs(Handle);
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
                var res = THSTensor_abs_(Handle);
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
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_add_scalar(IntPtr tensor, IntPtr trg, IntPtr alpha);

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
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_add_(IntPtr tensor, IntPtr trg, IntPtr alpha);

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
                var res = THSTensor_add_(Handle, target.Handle, alpha.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_add_scalar_(IntPtr tensor, IntPtr trg, IntPtr alpha);

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
                var res = THSTensor_add_scalar_(Handle, scalar.Handle, alpha.Handle);
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
            /// <param name="batch1">The first batch of matrices to be multiplied</param>
            /// <param name="batch2">The second batch of matrices to be multiplied</param>
            /// <param name="beta">Nultiplier for input (β)</param>
            /// <param name="alpha">Multiplier for batch1 @ batch2 (α)</param>
            /// <returns></returns>
            public Tensor addbmm(Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1)
            {
                var res = THSTensor_addbmm(Handle, batch1.Handle, batch2.Handle, beta, alpha);
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
            /// <param name="batch1">The first batch of matrices to be multiplied</param>
            /// <param name="batch2">The second batch of matrices to be multiplied</param>
            /// <param name="beta">Nultiplier for input (β)</param>
            /// <param name="alpha">Multiplier for batch1 @ batch2 (α)</param>
            /// <returns></returns>
            public Tensor addbmm_(Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1)
            {
                var res = THSTensor_addbmm_(Handle, batch1.Handle, batch2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_addcdiv(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

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
                    torch.CheckForErrors();
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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_addcdiv_(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

            /// <summary>
            /// Performs the in-place element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
            /// </summary>
            /// <param name="tensor1">First tensor</param>
            /// <param name="tensor2">Second tensor</param>
            /// <param name="value">Scale factor</param>
            /// <returns></returns>
            public Tensor addcdiv_(Tensor tensor1, Tensor tensor2, Scalar value)
            {
                var res = THSTensor_addcdiv_(Handle, tensor1.Handle, tensor2.Handle, value.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
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

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_addcmul(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

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
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_addcmul_(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

            /// <summary>
            /// Performs the in-place element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
            /// </summary>
            /// <param name="tensor1">First tensor</param>
            /// <param name="tensor2">Second tensor</param>
            /// <param name="value">Scale factor</param>
            /// <returns></returns>
            public Tensor addcmul_(Tensor tensor1, Tensor tensor2, Scalar value)
            {
                var res = THSTensor_addcmul_(Handle, tensor1.Handle, tensor2.Handle, value.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_addmm(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

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
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_addmm_(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

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
                var res = THSTensor_addmm_(Handle, mat1.Handle, mat2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_addmv(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta = 1, float alpha = 1);

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
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_addmv_(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

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
                var res = THSTensor_addmv_(Handle, mat.Handle, vec.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_addr(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

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
            /// <param name="vec1">The first vector of the outer product</param>
            /// <param name="vec2">The second vector of the outer product</param>
            /// <param name="beta">Input scale factor</param>
            /// <param name="alpha">Outer-product scale factor</param>
            /// <returns></returns>
            public Tensor addr_(Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f)
            {
                var res = THSTensor_addr_(Handle, vec1.Handle, vec2.Handle, beta, alpha);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_and(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise AND
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_and(Tensor other)
            {
                var res = THSTensor_bitwise_and(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_and_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise AND, in-place
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_and_(Tensor other)
            {
                var res = THSTensor_bitwise_and_(Handle, other.Handle);
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
                var res = THSTensor_bitwise_not(Handle);
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
                var res = THSTensor_bitwise_not_(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_or(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise OR
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_or(Tensor other)
            {
                var res = THSTensor_bitwise_or(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_or_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise OR, in-place
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_or_(Tensor other)
            {
                var res = THSTensor_bitwise_or_(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_xor(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise XOR
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_xor(Tensor other)
            {
                var res = THSTensor_bitwise_xor(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_xor_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise XOR, in-place.
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_xor_(Tensor other)
            {
                var res = THSTensor_bitwise_xor_(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_left_shift(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise left_shift
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_left_shift(Tensor other)
            {
                var res = THSTensor_bitwise_left_shift(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_left_shift_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise left_shift, in-place.
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_left_shift_(Tensor other)
            {
                var res = THSTensor_bitwise_left_shift_(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_right_shift(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise right_shift
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_right_shift(Tensor other)
            {
                var res = THSTensor_bitwise_right_shift(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_bitwise_right_shift_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise bitwise right_shift, in-place.
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor bitwise_right_shift_(Tensor other)
            {
                var res = THSTensor_bitwise_right_shift_(Handle, other.Handle);
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
                var res = THSTensor_ceil(Handle);
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
                var res = THSTensor_ceil_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_conj(IntPtr tensor);

            /// <summary>
            /// Returns a view of input with a flipped conjugate bit. If input has a non-complex dtype, this function just returns input.
            /// </summary>
            /// <returns></returns>
            public Tensor conj()
            {
                var res = THSTensor_conj(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_conj_physical(IntPtr tensor);

            /// <summary>
            /// Computes the element-wise conjugate of the given input tensor. If input has a non-complex dtype, this function just returns input.
            /// </summary>
            /// <returns></returns>
            public Tensor conj_physical()
            {
                var res = THSTensor_conj_physical(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_conj_physical_(IntPtr tensor);

            /// <summary>
            /// In-place version of conj_physical
            /// </summary>
            /// <returns></returns>
            public Tensor conj_physical_()
            {
                var res = THSTensor_conj_physical_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern long THSTensor_is_conj(IntPtr tensor);

            /// <summary>
            /// Returns true if the input is a conjugated tensor, i.e. its conjugate bit is set to True.
            /// </summary>
            /// <returns></returns>
            public bool is_conj()
            {
                var res = THSTensor_is_conj(Handle);
                torch.CheckForErrors();
                return res != 0;
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_resolve_conj(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with materialized conjugation if input’s conjugate bit is set to True, else returns input.
            /// The output tensor will always have its conjugate bit set to False.
            /// </summary>
            /// <returns></returns>
            public Tensor resolve_conj()
            {
                var res = THSTensor_resolve_conj(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_cummax(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

            /// <summary>
            /// Returns a tuple (values, indices) where values is the cumulative maximum of elements of input in the dimension dim.
            /// Indices is the index location of each maximum value found in the dimension dim.
            /// </summary>
            /// <param name="dimension">The dimension to do the operation over</param>
            public (Tensor values, Tensor indexes) cummax(long dimension)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_cummax(Handle, pa.CreateArray, dimension);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            [DllImport("LibTorchSharp")]
            static extern void THSTensor_cummin(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

            /// <summary>
            /// Returns a tuple (values, indices) where values is the cumulative minimum of elements of input in the dimension dim.
            /// Indices is the index location of each minimum value found in the dimension dim.
            /// </summary>
            /// <param name="dimension">The dimension to do the operation over</param>
            public (Tensor values, Tensor indexes) cummin(long dimension)
            {
                IntPtr[] ptrArray;

                using (var pa = new PinnedArray<IntPtr>()) {
                    THSTensor_cummin(Handle, pa.CreateArray, dimension);
                    torch.CheckForErrors();
                    ptrArray = pa.Array;
                }

                return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cumsum(IntPtr tensor, long dimension, bool has_type, sbyte scalar_type);

            /// <summary>
            /// Returns the cumulative sum of elements of input in the dimension dim.
            /// </summary>
            /// <param name="dimension">The dimension to do the operation over</param>
            /// <param name="type">The desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed.
            /// This is useful for preventing data type overflows.</param>
            /// <returns></returns>
            public Tensor cumsum(long dimension, ScalarType? type = null)
            {
                var res = THSTensor_cumsum(Handle, dimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cumprod(IntPtr tensor, long dimension, bool has_type, sbyte scalar_type);

            /// <summary>
            /// Returns the cumulative product of elements of input in the dimension dim.
            /// </summary>
            /// <param name="dimension">The dimension to do the operation over</param>
            /// <param name="type">The desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed.
            /// This is useful for preventing data type overflows.</param>
            /// <returns></returns>
            public Tensor cumprod(long dimension, ScalarType? type = null)
            {
                var res = THSTensor_cumprod(Handle, dimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_div(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string rounding_mode);

            /// <summary>
            /// Divides each element of the input by the corresponding element of other.
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor div(Tensor target, RoundingMode rounding_mode = RoundingMode.None)
            {
                var res = THSTensor_div(Handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Divides each element of the input by the corresponding element of other.
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor divide(Tensor target, RoundingMode rounding_mode = RoundingMode.None) => div(target, rounding_mode);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_div_scalar(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string rounding_mode);

            /// <summary>
            /// Scalar division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor div(Scalar target, RoundingMode rounding_mode = RoundingMode.None)
            {
                var res = THSTensor_div_scalar(Handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Scalar division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor divide(Scalar target, RoundingMode rounding_mode = RoundingMode.None) => div(target, rounding_mode);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_div_(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string rounding_mode);

            /// <summary>
            /// In-place division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor div_(Tensor target, RoundingMode rounding_mode = RoundingMode.None)
            {
                var res = THSTensor_div_(Handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// In-place division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor divide_(Tensor target, RoundingMode rounding_mode = RoundingMode.None) => div_(target, rounding_mode);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_div_scalar_(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string rounding_mode);

            /// <summary>
            /// In-place scalar division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor div_(Scalar target, RoundingMode rounding_mode = RoundingMode.None)
            {
                var res = THSTensor_div_scalar_(Handle, target.Handle, rounding_mode == RoundingMode.trunc ? "trunc" : rounding_mode == RoundingMode.floor ? "floor" : null);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// In-place scalar division
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <param name="rounding_mode">Rounding mode.</param>
            /// <returns></returns>
            public Tensor divide_(Scalar target, RoundingMode rounding_mode = RoundingMode.None) => div_(target, rounding_mode);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_exp(IntPtr tensor);

            /// <summary>
            /// Returns a new tensor with the exponential of the elements of the input tensor.
            /// </summary>
            public Tensor exp()
            {
                var res = THSTensor_exp(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_exp_(IntPtr tensor);

            /// <summary>
            /// Replaces the tensor with the exponential of the elements of the input tensor.
            /// </summary>
            public Tensor exp_()
            {
                var res = THSTensor_exp_(Handle);
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
                var res = THSTensor_exp2(Handle);
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
                var res = THSTensor_expm1(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_expm1_(IntPtr tensor);

            /// <summary>
            /// Replaces each element with the exponential of the element minus 1 of input, in-place.
            /// </summary>
            /// <returns></returns>
            public Tensor expm1_()
            {
                var res = THSTensor_expm1_(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_float_power(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Raises input to the power of exponent, elementwise, in double precision.
            /// </summary>
            /// <param name="target">The exponent.</param>
            /// <returns></returns>
            /// <remarks> If neither input is complex returns a torch.float64 tensor, and if one or more inputs is complex returns a torch.complex128 tensor.</remarks>
            public Tensor float_power(Tensor target)
            {
                var res = THSTensor_float_power(Handle, target.Handle);
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
                var res = THSTensor_floor(Handle);
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
                var res = THSTensor_floor_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmod(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <returns></returns>
            public Tensor fmod(Tensor target)
            {
                var res = THSTensor_fmod(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmod_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes the element-wise remainder of division, in-place.
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <returns></returns>
            public Tensor fmod_(Tensor target)
            {
                var res = THSTensor_fmod_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmod_scalar(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="scalar">Denominator</param>
            /// <returns></returns>
            public Tensor fmod(Scalar scalar)
            {
                var res = THSTensor_fmod_scalar(Handle, scalar.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_fmod_scalar_(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Computes the element-wise remainder of division, in-place
            /// </summary>
            /// <param name="scalar">Denominator</param>
            /// <returns></returns>
            public Tensor fmod_(Scalar scalar)
            {
                var res = THSTensor_fmod_scalar_(Handle, scalar.Handle);
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
                var res = THSTensor_frac(Handle);
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
                var res = THSTensor_frac_(Handle);
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
                var mantissa = THSTensor_frexp(Handle, out var exponent);
                if (mantissa == IntPtr.Zero || exponent == IntPtr.Zero)
                    torch.CheckForErrors();
                return (new Tensor(mantissa), new Tensor(exponent));
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gcd(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Computes the element-wise greatest common divisor (GCD) of input and other.
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            public Tensor gcd(Tensor other)
            {
                var res = THSTensor_gcd(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_gcd_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Computes the element-wise greatest common divisor (GCD) of input and other.
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor gcd_(Tensor other)
            {
                var res = THSTensor_gcd_(Handle, other.Handle);
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
                var res = THSTensor_histc(Handle, bins, min, max);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_hypot(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Element-wise: given the legs of a right triangle, return its hypotenuse.
            /// </summary>
            /// <param name="other">The second input tensor.</param>
            /// <returns></returns>
            public Tensor hypot(Tensor other)
            {
                var res = THSTensor_hypot(Handle, other.Handle);
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
                var res = THSTensor_log(Handle);
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
                var res = THSTensor_log_(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logaddexp(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logarithm of the sum of exponentiations of the inputs.
            /// </summary>
            /// <param name="other">The second input tensor.</param>
            /// <returns></returns>
            public Tensor logaddexp(Tensor other)
            {
                var res = THSTensor_logaddexp(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logaddexp2(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logarithm of the sum of exponentiations of the inputs in base-2.
            /// </summary>
            /// <param name="other">The second input tensor.</param>
            /// <returns></returns>
            public Tensor logaddexp2(Tensor other)
            {
                var res = THSTensor_logaddexp2(Handle, other.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logcumsumexp(IntPtr tensor, long dim);

            /// <summary>
            /// Returns the logarithm of the cumulative summation of the exponentiation of elements of input in the dimension dim.
            /// </summary>
            /// <param name="dim">The dimension to do the operation over</param>
            /// <returns></returns>
            public Tensor logcumsumexp(long dim)
            {
                var res = THSTensor_logcumsumexp(Handle, dim);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logsumexp(IntPtr tensor, long dim, bool keepdim);

            /// <summary>
            /// Returns the log of summed exponentials of each row of the input tensor in the given dimension dim. 
            /// </summary>
            /// <param name="dim">The dimension to do the operation over</param>
            /// <param name="keepdim">Thether the output tensor has dim retained or not.</param>
            /// <returns></returns>
            /// <remarks>The computation is numerically stabilized.</remarks>
            public Tensor logsumexp(long dim, Boolean keepdim = false)
            {
                var res = THSTensor_logsumexp(Handle, dim, keepdim);
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
                var res = THSTensor_log10(Handle);
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
                var res = THSTensor_log10_(Handle);
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
                var res = THSTensor_log1p(Handle);
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
                var res = THSTensor_log1p_(Handle);
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
                var res = THSTensor_log2(Handle);
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
                var res = THSTensor_log2_(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_and(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical AND
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_and(Tensor other)
            {
                var res = THSTensor_logical_and(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_and_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical AND, in place
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_and_(Tensor other)
            {
                var res = THSTensor_logical_and_(Handle, other.Handle);
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
                var res = THSTensor_logical_not(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_not_(IntPtr tensor);

            /// <summary>
            /// Logical NOT, in place
            /// </summary>
            /// <returns></returns>
            public Tensor logical_not_()
            {
                var res = THSTensor_logical_not_(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_or(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical OR
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_or(Tensor other)
            {
                var res = THSTensor_logical_or(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_or_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical OR, in place
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_or_(Tensor other)
            {
                var res = THSTensor_logical_or_(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_xor(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical XOR
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_xor(Tensor other)
            {
                var res = THSTensor_logical_xor(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logical_xor_(IntPtr tensor, IntPtr other);

            /// <summary>
            /// Logical XOR, in place
            /// </summary>
            /// <param name="other">Right-hand operand.</param>
            /// <returns></returns>
            public Tensor logical_xor_(Tensor other)
            {
                var res = THSTensor_logical_xor_(Handle, other.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_logit(IntPtr tensor, IntPtr eps);

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
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor mul(Tensor target)
            {
                var res = THSTensor_mul(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise multiplication
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor multiply(Tensor target) => mul(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mul_scalar(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Element-wise multiplication
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor mul(Scalar target)
            {
                var res = THSTensor_mul_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            /// <summary>
            /// Element-wise multiplcation
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor multiply(Scalar target) => mul(target);

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mul_(IntPtr tensor, IntPtr target);

            /// <summary>
            /// Element-wise multiplication, in place
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor mul_(Tensor target)
            {
                var res = THSTensor_mul_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_mul_scalar_(IntPtr tensor, IntPtr target);

            /// <summary>
            /// Element-wise multiplication, in place
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor mul_(Scalar target)
            {
                var res = THSTensor_mul_scalar_(Handle, target.Handle);
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
                var res = THSTensor_neg(Handle);
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
                var res = THSTensor_neg_(Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_pow(IntPtr tensor, IntPtr exponent);

            /// <summary>
            /// Takes the power of each element in input with exponent and returns a tensor with the result.
            /// </summary>
            /// <param name="exponent">The exponent.</param>
            /// <returns></returns>
            public Tensor pow(Tensor exponent)
            {
                var res = THSTensor_pow(Handle, exponent.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_pow_(IntPtr tensor, IntPtr exponent);

            /// <summary>
            /// Replaces each element in input with the power of the element and the exponent.
            /// </summary>
            /// <param name="exponent">The exponent.</param>
            /// <returns></returns>
            public Tensor pow_(Tensor exponent)
            {
                var res = THSTensor_pow_(Handle, exponent.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_pow_scalar(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Takes the power of each element in input with exponent and returns a tensor with the result.
            /// </summary>
            /// <param name="exponent">The exponent.</param>
            /// <returns></returns>
            public Tensor pow(Scalar exponent)
            {
                var res = THSTensor_pow_scalar(Handle, exponent.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_pow_scalar_(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Replaces each element in input with the power of the element and the exponent.
            /// </summary>
            /// <param name="exponent">The exponent.</param>
            /// <returns></returns>
            public Tensor pow_(Scalar exponent)
            {
                var res = THSTensor_pow_scalar_(Handle, exponent.Handle);
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
                var res = THSTensor_reciprocal(Handle);
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
                var res = THSTensor_reciprocal_(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_remainder(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <returns></returns>
            public Tensor remainder(Tensor target)
            {
                var res = THSTensor_remainder(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_remainder_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes the element-wise remainder of division, in place
            /// </summary>
            /// <param name="target">Denominator</param>
            /// <returns></returns>
            public Tensor remainder_(Tensor target)
            {
                var res = THSTensor_remainder_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_remainder_scalar(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="scalar">Denominator</param>
            /// <returns></returns>
            public Tensor remainder(Scalar scalar)
            {
                var res = THSTensor_remainder_scalar(Handle, scalar.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_remainder_scalar_(IntPtr tensor, IntPtr scalar);

            /// <summary>
            /// Computes the element-wise remainder of division.
            /// </summary>
            /// <param name="scalar">Denominator</param>
            /// <returns></returns>
            public Tensor remainder_(Scalar scalar)
            {
                var res = THSTensor_remainder_scalar_(Handle, scalar.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_round(IntPtr tensor, long decimals);

            /// <summary>
            /// Returns a new tensor with each of the elements of input rounded to the closest value with the given number of decimals.
            /// </summary>
            /// <param name="decimals">Number of decimal places to round to (default: 0). If decimals is negative, it specifies the number of positions to the left of the decimal point.</param>
            /// <returns></returns>
            public Tensor round(long decimals = 0L)
            {
                var res = THSTensor_round(Handle, decimals);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_round_(IntPtr tensor, long decimals);

            /// <summary>
            /// Replaces each of the elements of input with the element rounded to the closest value with the given number of decimals.
            /// </summary>
            /// <param name="decimals">Number of decimal places to round to (default: 0). If decimals is negative, it specifies the number of positions to the left of the decimal point.</param>
            /// <returns></returns>
            public Tensor round_(long decimals = 0L)
            {
                var res = THSTensor_round_(Handle, decimals);
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
                var res = THSTensor_rsqrt(Handle);
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
                var res = THSTensor_rsqrt_(Handle);
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
                var res = THSTensor_sqrt(Handle);
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
                var res = THSTensor_sqrt_(Handle);
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
                var res = THSTensor_sign(Handle);
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
                var res = THSTensor_sign_(Handle);
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
                var res = THSTensor_signbit(Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sub(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Element-wise subtraction
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor sub(Tensor target)
            {
                var res = THSTensor_sub(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sub_scalar(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Element-wise subtraction
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor sub(Scalar target)
            {
                var res = THSTensor_sub_scalar(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sub_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Element-wise subtraction, in place
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor sub_(Tensor target)
            {
                var res = THSTensor_sub_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_sub_scalar_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Element-wise subtraction, in-place
            /// </summary>
            /// <param name="target">Right-hand operand</param>
            /// <returns></returns>
            public Tensor sub_(Scalar target)
            {
                var res = THSTensor_sub_scalar_(Handle, target.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cumulative_trapezoid_x(IntPtr y, IntPtr x, long dim);
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_cumulative_trapezoid_dx(IntPtr y, double dx, long dim);

            /// <summary>
            /// Cumulatively computes the trapezoidal rule along dim. By default the spacing between elements is assumed to be 1,
            /// but dx can be used to specify a different constant spacing.
            /// </summary>
            /// <param name="dx">Constant spacing between values.</param>
            /// <param name="dim">The dimension along which to compute the trapezoidal rule. The last (inner-most) dimension by default.</param>
            /// <returns></returns>
            public Tensor cumulative_trapezoid(double dx = 1, long dim = -1)
            {
                IntPtr res = THSTensor_trapezoid_dx(Handle, dx, dim);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
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
                IntPtr res = THSTensor_trapezoid_x(Handle, x.Handle, dim);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_trapezoid_x(IntPtr y, IntPtr x, long dim);
            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_trapezoid_dx(IntPtr y, double dx, long dim);

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
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
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
                var res = THSTensor_trunc(Handle);
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
                var res = THSTensor_trunc_(Handle);
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
            /// <param name="y">The 'y' operand.</param>
            /// <returns></returns>
            public Tensor xlogy(Tensor y)
            {
                var res = THSTensor_xlogy(Handle, y.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_xlogy_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes x * log(y) in place
            /// </summary>
            /// <param name="y">The 'y' operand.</param>
            /// <returns></returns>
            public Tensor xlogy_(Tensor y)
            {
                var res = THSTensor_xlogy_(Handle, y.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_xlogy_scalar(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes x * log(y)
            /// </summary>
            /// <param name="y">The 'y' operand.</param>
            /// <returns></returns>
            public Tensor xlogy(Scalar y)
            {
                var res = THSTensor_xlogy_scalar(Handle, y.Handle);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }


            [DllImport("LibTorchSharp")]
            static extern IntPtr THSTensor_xlogy_scalar_(IntPtr tensor, IntPtr trg);

            /// <summary>
            /// Computes x * log(y) in place
            /// </summary>
            /// <param name="y">The 'y' operand.</param>
            /// <returns></returns>
            public Tensor xlogy_(Scalar y)
            {
                var res = THSTensor_xlogy_scalar_(Handle, y.Handle);
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
        /// <param name="input">The input tensor.</param>
        public static Tensor abs(Tensor input) => input.abs();

        /// <summary>
        /// Compute the absolute value of each element in the tensor
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor absolute(Tensor input) => input.abs();

        /// <summary>
        /// Compute the absolute value of each element in the tensor, in-place
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor abs_(Tensor input) => input.abs_();

        /// <summary>
        /// Compute the absolute value of each element in the tensor, in-place
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor absolute_(Tensor input) => input.abs_();

        /// <summary>
        /// Add two tensors, element-wise
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        /// <returns></returns>
        public static Tensor add(Tensor left, Tensor right) => left.add(right);

        /// <summary>
        /// Add a scalar value to each element in the target tensor.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor add(Tensor left, Scalar right) => left.add(right);

        /// <summary>
        /// Add two tensors, element-wise, scaling the second operator by 'alpha'
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        /// <param name="alpha">RHS scale factor.</param>
        public static Tensor add(Tensor left, Tensor right, Scalar alpha) => left.add(right, alpha);

        /// <summary>
        /// Add a scalar value to each element in the target tensor, scaled by 'alpha'
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        /// <param name="alpha">RHS scale factor.</param>
        public static Tensor add(Tensor left, Scalar right, Scalar alpha) => left.add(right, alpha);

        /// <summary>
        /// Add two tensors, element-wise, in place
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor add_(Tensor left, Tensor right) => left.add_(right);

        /// <summary>
        /// Add a scalar value to each element in the target tensor, in place.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor add_(Tensor left, Scalar right) => left.add_(right);

        /// <summary>
        /// Add two tensors, element-wise, scaling the second operator by 'alpha', in place
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        /// <param name="alpha">RHS scale factor.</param>
        public static Tensor add_(Tensor left, Tensor right, Scalar alpha) => left.add_(right, alpha);

        /// <summary>
        /// Add a scalar value to each element in the target tensor, scaled by 'alpha', in place
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        /// <param name="alpha">RHS scale factor.</param>
        public static Tensor add_(Tensor left, Scalar right, Scalar alpha) => left.add_(right, alpha);

        /// <summary>
        /// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced
        /// add step (all matrix multiplications get accumulated along the first dimension).
        /// input is added to the final result.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="batch1">The first batch of matrices to be multiplied</param>
        /// <param name="batch2">The second batch of matrices to be multiplied</param>
        /// <param name="beta">Nultiplier for input (β)</param>
        /// <param name="alpha">Multiplier for batch1 @ batch2 (α)</param>
        public static Tensor addbmm(Tensor input, Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1) => input.addbmm(batch1, batch2, beta, alpha);

        /// <summary>
        /// Performs a batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced
        /// add step (all matrix multiplications get accumulated along the first dimension).
        /// input is added to the final result.
        /// In-place version of addbmm.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="batch1">The first batch of matrices to be multiplied</param>
        /// <param name="batch2">The second batch of matrices to be multiplied</param>
        /// <param name="beta">Nultiplier for input (β)</param>
        /// <param name="alpha">Multiplier for batch1 @ batch2 (α)</param>
        public static Tensor addbmm_(Tensor input, Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1) => input.addbmm_(batch1, batch2, beta, alpha);

        /// <summary>
        /// Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="tensor1">First tensor</param>
        /// <param name="tensor2">Second tensor</param>
        /// <param name="value">Scale factor</param>
        public static Tensor addcdiv(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value) => input.addcdiv(tensor1, tensor2, value);

        /// <summary>
        /// Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// In-place version of addcdiv.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="tensor1">First tensor</param>
        /// <param name="tensor2">Second tensor</param>
        /// <param name="value">Scale factor</param>
        public static Tensor addcdiv_(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value) => input.addcdiv_(tensor1, tensor2, value);

        /// <summary>
        /// Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="tensor1">First tensor</param>
        /// <param name="tensor2">Second tensor</param>
        /// <param name="value">Scale factor</param>
        public static Tensor addcmul(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value) => input.addcmul(tensor1, tensor2, value);

        /// <summary>
        /// Performs the element-wise divismultiplicationion of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// In-place version of addcdiv.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="tensor1">First tensor</param>
        /// <param name="tensor2">Second tensor</param>
        /// <param name="value">Scale factor</param>
        public static Tensor addcmul_(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value) => input.addcmul_(tensor1, tensor2, value);

        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="mat1">First matrix</param>
        /// <param name="mat2">Second matrix</param>
        /// <param name="beta">Input scale factor</param>
        /// <param name="alpha">Matrix multiplication scale factor</param>
        public static Tensor addmm(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmm(mat1, mat2, beta, alpha);

        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="mat1">First matrix</param>
        /// <param name="mat2">Second matrix</param>
        /// <param name="beta">Input scale factor</param>
        /// <param name="alpha">Matrix multiplication scale factor</param>
        public static Tensor addmm_(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmm_(mat1, mat2, beta, alpha);

        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="mat1">First matrix</param>
        /// <param name="mat2">Second matrix</param>
        /// <param name="beta">Input scale factor</param>
        /// <param name="alpha">Matrix multiplication scale factor</param>
        public static Tensor addmv(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmv(mat1, mat2, beta, alpha);

        /// <summary>
        /// Performs a matrix multiplication of the matrices mat1 and mat2. The matrix input is added to the final result.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="mat1">First matrix</param>
        /// <param name="mat2">Second matrix</param>
        /// <param name="beta">Input scale factor</param>
        /// <param name="alpha">Matrix multiplication scale factor</param>
        public static Tensor addmv_(Tensor input, Tensor mat1, Tensor mat2, float beta, float alpha) => input.addmv_(mat1, mat2, beta, alpha);

        /// <summary>
        /// Performs the outer-product of vectors vec1 and vec2 and adds it to the input tensor.
        /// </summary>
        /// <returns></returns>
        public static Tensor addr(Tensor input, Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f) => input.addr(vec1, vec2, beta, alpha);

        /// <summary>
        /// Performs the outer-product of vectors vec1 and vec2 and adds it to the input tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="vec1">The first vector of the outer product</param>
        /// <param name="vec2">The second vector of the outer product</param>
        /// <param name="beta">Input scale factor</param>
        /// <param name="alpha">Outer-product scale factor</param>
        public static Tensor addr_(Tensor input, Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f) => input.addr_(vec1, vec2, beta, alpha);

        public static Tensor bincount(Tensor input, Tensor weights = null, long minlength = 0) => input.bincount(weights, minlength);

        /// <summary>
        /// Element-wise bitwise AND
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_and(Tensor left, Tensor right) => left.bitwise_and(right);

        /// <summary>
        /// Element-wise bitwise AND, in place.
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_and_(Tensor left, Tensor right) => left.bitwise_and_(right);

        /// <summary>
        /// Element-wise bitwise NOT
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor bitwise_not(Tensor input) => input.bitwise_not();

        /// <summary>
        /// Element-wise bitwise NOT, in place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor bitwise_not_(Tensor input) => input.bitwise_not_();

        /// <summary>
        /// Element-wise bitwise OR
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_or(Tensor left, Tensor right) => left.bitwise_or(right);

        /// <summary>
        /// Element-wise bitwiseXOR, in place.
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_or_(Tensor left, Tensor right) => left.bitwise_or_(right);

        /// <summary>
        /// Element-wise bitwise XOR
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_xor(Tensor left, Tensor right) => left.bitwise_xor(right);

        /// <summary>
        /// Element-wise bitwise XOR, in place.
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_xor_(Tensor left, Tensor right) => left.bitwise_xor_(right);

        /// <summary>
        /// Performs a batch matrix-matrix product of matrices stored in input and mat2.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="batch2">the second batch of matrices to be multiplied</param>
        public static Tensor bmm(Tensor input, Tensor batch2) => input.bmm(batch2);

        /// <summary>
        /// Performs a batch matrix-matrix product of matrices in batch1 and batch2. input is added to the final result.
        /// batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
        /// </summary>
        /// <param name="input">The tensor to be added</param>
        /// <param name="batch1">The first batch of matrices to be multiplied</param>
        /// <param name="batch2">The second batch of matrices to be multiplied</param>
        /// <param name="beta">A multiplier for input</param>
        /// <param name="alpha">A multiplier for batch1 @ batch2</param>
        public static Tensor baddbmm(Tensor input, Tensor batch1, Tensor batch2, float beta = 1, float alpha = 1) => input.baddbmm(batch1, batch2, beta, alpha);

        /// <summary>
        /// Element-wise bitwise left shift
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_left_shift(Tensor left, Tensor right) => left.bitwise_left_shift(right);

        /// <summary>
        /// Element-wise bitwise left shift, in place.
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_left_shift_(Tensor left, Tensor right) => left.bitwise_left_shift_(right);

        /// <summary>
        /// Element-wise bitwise right shift
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_right_shift(Tensor left, Tensor right) => left.bitwise_right_shift(right);

        /// <summary>
        /// Element-wise bitwise right shift, in place.
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_right_shift_(Tensor left, Tensor right) => left.bitwise_right_shift_(right);

        /// <summary>
        /// Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor ceil(Tensor input) => input.ceil();

        /// <summary>
        /// Replaces each element of the input with the smallest integer greater than or equal to the element.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor ceil_(Tensor input) => input.ceil_();

        /// <summary>
        /// Returns a view of input with a flipped conjugate bit. If input has a non-complex dtype, this function just returns input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor conj(Tensor input) => input.conj();

        /// <summary>
        /// Returns true if the input is a conjugated tensor, i.e. its conjugate bit is set to True.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static bool is_conj(Tensor input) => input.is_conj();

        /// <summary>
        /// Computes the element-wise conjugate of the given input tensor. If input has a non-complex dtype, this function just returns input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor conj_physical(Tensor input) => input.conj_physical();

        /// <summary>
        /// In-place version of conj_physical
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor conj_physical_(Tensor input) => input.conj_physical_();

        /// <summary>
        /// Returns a new tensor with materialized conjugation if input’s conjugate bit is set to True, else returns input.
        /// The output tensor will always have its conjugate bit set to False.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor resolve_conj(Tensor input) => input.resolve_conj();

        /// <summary>
        /// Returns the cumulative sum of elements of input in the dimension dim.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimension">The dimension to do the operation over</param>
        /// <param name="type">The desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed.
        /// This is useful for preventing data type overflows.</param>
        public static Tensor cumsum(Tensor input, long dimension, ScalarType? type = null) => input.cumsum(dimension, type);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor div(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor divide(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor div(Tensor left, Scalar right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor divide(Tensor left, Scalar right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor div_(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div_(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor divide_(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div_(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor div_(Tensor left, Scalar right, RoundingMode rounding_mode = RoundingMode.None) => left.div_(right, rounding_mode);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
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
        /// <param name="input">The input tensor.</param>
        public static Tensor exp(Tensor input) => input.exp();

        /// <summary>
        /// Replaces each element of the input with the exponential of the elements of the input tensor input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor exp_(Tensor input) => input.exp_();

        /// <summary>
        /// Computes the base 2 exponential function of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor exp2(Tensor input) => input.exp2();

        /// <summary>
        /// Returns a new tensor with the exponential of the elements minus 1 of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor expm1(Tensor input) => input.expm1();

        /// <summary>
        /// Replaces each element with the exponential of the element minus 1 of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor expm1_(Tensor input) => input.expm1_();

        /// <summary>
        /// Raises input to the power of exponent, elementwise, in double precision.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="target">The exponent.</param>
        /// <remarks> If neither input is complex returns a torch.float64 tensor, and if one or more inputs is complex returns a torch.complex128 tensor.</remarks>
        public static Tensor float_power(Tensor input, Tensor target) => input.float_power(target);

        /// <summary>
        /// Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor floor(Tensor input) => input.floor();

        /// <summary>
        /// Replaces each element with the floor of the input, the largest integer less than or equal to each element.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor floor_(Tensor input) => input.exp_();

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        public static Tensor fmod(Tensor left, Tensor right) => left.fmod(right);

        /// <summary>
        /// Computes the element-wise remainder of division, in place.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor fmod_(Tensor left, Tensor right) => left.fmod_(right);

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor fmod(Tensor left, Scalar right) => left.fmod(right);

        /// <summary>
        /// Computes the element-wise remainder of division, in place.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor fmod_(Tensor left, Scalar right) => left.fmod_(right);

        /// <summary>
        /// Computes the fractional portion of each element in input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor frac(Tensor input) => input.frac();

        /// <summary>
        /// Computes the fractional portion of each element in input, in-place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor frac_(Tensor input) => input.frac_();

        /// <summary>
        /// Decomposes input into mantissa and exponent tensors 
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static (Tensor Mantissa, Tensor Exponent) frexp(Tensor input) => input.frexp();

        /// <summary>
        /// Computes the element-wise greatest common divisor (GCD) of input and other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor gcd(Tensor left, Tensor right) => left.gcd(right);

        /// <summary>
        /// Computes the element-wise greatest common divisor (GCD) of input and other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor gcd_(Tensor left, Tensor right) => left.gcd_(right);

        /// <summary>
        /// Computes the histogram of a tensor.
        /// The elements are sorted into equal width bins between min and max.If min and max are both zero, the minimum and maximum values of the data are used.
        /// Elements lower than min and higher than max are ignored.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="bins">Number of histogram bins</param>
        /// <param name="min">Lower end of the range (inclusive)</param>
        /// <param name="max">Upper end of the range (inclusive)</param>
        public static Tensor histc(Tensor input, long bins = 100, long min = 0, long max = 0) => input.histc(bins, min, max);

        /// <summary>
        /// Element-wise: given the legs of a right triangle, return its hypotenuse.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor hypot(Tensor left, Tensor right) => left.hypot(right);

        /// <summary>
        /// Computes the logarithmic derivative of the gamma function on input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor digamma(Tensor input) => input.digamma();

        /// <summary>
        /// Computes the logarithmic derivative of the gamma function on input, in place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor digamma_(Tensor input) => input.digamma_();

        /// <summary>
        /// Computes the logarithm of the gamma function on input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor lgamma(Tensor input) => input.lgamma();

        /// <summary>
        /// Computes the logarithm of the gamma function on input, in place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor lgamma_(Tensor input) => input.lgamma();

        /// <summary>
        /// Computes the multivariate log-gamma function) with dimension pp element-wise
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="p">The number of dimensions</param>
        /// <returns></returns>
        public static Tensor mvlgamma(Tensor input, long p) => input.mvlgamma(p);

        /// <summary>
        /// Computes the multivariate log-gamma function) with dimension pp element-wise, in place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="p">The number of dimensions</param>
        /// <returns></returns>
        public static Tensor mvlgamma_(Tensor input, long p) => input.mvlgamma_(p);


        /// <summary>
        /// Computes the Nth derivative of the digamma function on input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="p">The number of dimensions</param>
        /// <returns></returns>
        public static Tensor polygamma(Tensor input, long p) => input.polygamma(p);

        /// <summary>
        /// Computes the Nth derivative of the digamma function on input, in-place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="p">The number of dimensions</param>
        public static Tensor polygamma_(Tensor input, long p) => input.polygamma_(p);

        /// <summary>
        /// Returns a new tensor with the natural logarithm of the input elements.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log(Tensor input) => input.log();

        /// <summary>
        /// Replaces each elements with the natural logarithm of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log_(Tensor input) => input.log_();

        /// <summary>
        /// Returns a new tensor with the natural logarithm of (1 + input).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log1p(Tensor input) => input.log1p();

        /// <summary>
        /// Replaces each elements with the natural logarithm of (1 + input), in place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log1p_(Tensor input) => input.log1p_();

        /// <summary>
        /// Logarithm of the sum of exponentiations of the inputs.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor logaddexp(Tensor left, Tensor right) => left.logaddexp(right);

        /// <summary>
        /// Logarithm of the sum of exponentiations of the inputs in base-2.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor logaddexp2(Tensor left, Tensor right) => left.logaddexp2(right);

        /// <summary>
        /// Returns the logarithm of the cumulative summation of the exponentiation of elements of input in the dimension dim.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to do the operation over</param>
        public static Tensor logcumsumexp(Tensor input, long dim) => input.logcumsumexp(dim);

        /// <summary>
        /// Returns the log of summed exponentials of each row of the input tensor in the given dimension dim. 
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to do the operation over</param>
        /// <param name="keepdim">Thether the output tensor has dim retained or not.</param>
        public static Tensor logsumexp(Tensor input, long dim, Boolean keepdim = false) => input.logsumexp(dim, keepdim);

        /// <summary>
        /// Returns a new tensorwith the logarithm to the base 10 of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log10(Tensor input) => input.log();

        /// <summary>
        /// Replaces each elements with the logarithm to the base 10 of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log10_(Tensor input) => input.log_();

        /// <summary>
        /// Returns a new tensorwith the logarithm to the base 10 of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log2(Tensor input) => input.log2();

        /// <summary>
        /// Replaces each elements with the logarithm to the base 10 of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log2_(Tensor input) => input.log2_();

        /// <summary>
        /// Element-wise logical AND
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor logical_and(Tensor left, Tensor right) => left.logical_and(right);

        /// <summary>
        /// Element-wise logical AND, in place.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor logical_and_(Tensor left, Tensor right) => left.logical_and_(right);

        /// <summary>
        /// Element-wise logical NOT
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor logical_not(Tensor input) => input.logical_not();

        /// <summary>
        /// Element-wise logical OR
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor logical_or(Tensor left, Tensor right) => left.logical_or(right);

        /// <summary>
        /// Element-wise logicalXOR, in place.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor logical_or_(Tensor left, Tensor right) => left.logical_or_(right);

        /// <summary>
        /// Element-wise logical XOR
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor logical_xor(Tensor left, Tensor right) => left.logical_xor(right);

        /// <summary>
        /// Returns a new tensor with the logit of the elements of input.
        /// input is clamped to [eps, 1 - eps] when eps is not null
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="eps">The epsilon for input clamp bound.</param>
        public static Tensor logit(Tensor input, double? eps = null) => input.logit(eps);

        public static Tensor max(Tensor input) => input.max();

        static public Tensor maximum(Tensor input, Tensor other) => input.maximum(other);

        static public (Tensor values, Tensor indexes) max(Tensor input, long dimension, bool keepDim = false) => input.max(dimension, keepDim);

        public static Tensor mean(Tensor input) => input.mean();

        public static Tensor mean(Tensor input, long[] dimensions, bool keepDimension = false, ScalarType? type = null) => input.mean(dimensions, keepDimension, type);

        public static Tensor min(Tensor input) => input.min();

        static public Tensor minimum(Tensor input, Tensor other) => input.minimum(other);

        static public (Tensor values, Tensor indexes) min(Tensor input, long dimension, bool keepDim = false) => input.min(dimension, keepDim);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor mul(Tensor left, Tensor right) => left.mul(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor multiply(Tensor left, Tensor right) => left.mul(right);

        /// <summary>
        /// Computes the matrix exponential of a square matrix or of each square matrix in a batch.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor matric_exp(Tensor input) => input.matrix_exp();

        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor mul(Tensor left, Scalar right) => left.mul(right);

        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor multiply(Tensor left, Scalar right) => left.mul(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor mul_(Tensor left, Tensor right) => left.mul_(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor multiply_(Tensor left, Tensor right) => left.mul_(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor mul_(Tensor left, Scalar right) => left.mul_(right);

        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor multiply_(Tensor left, Scalar right) => left.mul_(right);

        /// <summary>
        /// Negation
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor neg(Tensor input) => input.neg();

        /// <summary>
        /// Negation
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor negative(Tensor input) => input.neg();

        /// <summary>
        /// In-place negation
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor neg_(Tensor input) => input.neg_();

        /// <summary>
        /// In-place negation
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor negative_(Tensor input) => input.neg_();

        /// <summary>
        /// Takes the power of each element in input with exponent and returns a tensor with the result.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="exponent">The right-hand operand.</param>
        public static Tensor pow(Tensor left, Tensor exponent) => left.pow(exponent);

        /// <summary>
        /// Takes the power of each element in input with exponent and returns a tensor with the result.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="exponent">The right-hand operand.</param>
        public static Tensor pow(Tensor left, Scalar exponent) => left.pow(exponent);

        /// <summary>
        /// Replaces each element in input with the power of the element and the exponent.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="exponent">The right-hand operand.</param>
        public static Tensor pow_(Tensor left, Tensor exponent) => left.pow_(exponent);

        /// <summary>
        /// Replaces each element in input with the power of the element and the exponent.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="exponent">The right-hand operand.</param>
        public static Tensor pow_(Tensor left, Scalar exponent) => left.pow_(exponent);

        /// <summary>
        /// Returns a new tensor with the reciprocal of the elements of input
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor reciprocal(Tensor input) => input.reciprocal();

        /// <summary>
        /// Replaces each element with the reciprocal of the input
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor reciprocal_(Tensor input) => input.reciprocal_();

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor remainder(Tensor left, Tensor right) => left.remainder(right);

        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor remainder(Tensor left, Scalar right) => left.remainder(right);

        /// <summary>
        /// Computes the element-wise remainder of division, in place
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor remainder_(Tensor left, Tensor right) => left.remainder_(right);

        /// <summary>
        /// Computes the element-wise remainder of division, in place
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor remainder_(Tensor left, Scalar right) => left.remainder_(right);

        /// <summary>
        /// Returns a new tensor with each of the elements of input rounded to the closest value with the given number of decimals.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="decimals">Number of decimal places to round to (default: 0). If decimals is negative, it specifies the number of positions to the left of the decimal point.</param>
        public static Tensor round(Tensor input, long decimals = 0L) => input.round(decimals);

        /// <summary>
        /// Replaces each of the elements of input with the element rounded to the closest  value with the given number of decimals.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="decimals">Number of decimal places to round to (default: 0). If decimals is negative, it specifies the number of positions to the left of the decimal point.</param>
        public static Tensor round_(Tensor input, long decimals = 0L) => input.round_(decimals);

        /// <summary>
        /// Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor rsqrt(Tensor input) => input.rsqrt();

        /// <summary>
        /// Replaces each of the elements of input with  the reciprocal of the square-root of each of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor rsqrt_(Tensor input) => input.rsqrt_();

        /// <summary>
        /// Computes the element-wise square
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor square(Tensor input) => input.pow(2);

        /// <summary>
        /// Computes the element-wise square root
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor sqrt(Tensor input) => input.sqrt();

        /// <summary>
        /// Computes the element-wise square root, in place
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor sqrt_(Tensor input) => input.sqrt_();

        /// <summary>
        /// Computes the logistic sigmoid function of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor sigmoid(Tensor input) => input.sigmoid();

        /// <summary>
        /// Computes the logistic sigmoid function of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor sigmoid_(Tensor input) => input.sigmoid_();

        /// <summary>
        /// Returns a new tensor with the signs (-1, 0, 1) of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor sign(Tensor input) => input.sign();

        /// <summary>
        /// Replaces each element with the signs (-1, 0, 1) of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor sign_(Tensor input) => input.sign_();

        /// <summary>
        /// Tests whether each element of input has its sign bit set (is less than zero) or not.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>A boolean tensor of the same shape as the input.</returns>
        public static Tensor signbit(Tensor input) => input.signbit();

        /// <summary>
        /// Calculates the standard deviation and mean of all elements in the tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        public static Tensor std(Tensor input, bool unbiased = true) => input.std(unbiased);

        /// <summary>
        /// Calculates the standard deviation and mean of all elements in the tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        public static (Tensor std, Tensor mean) std_mean(Tensor input, bool unbiased = true) => input.std_mean(unbiased);

        /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        public static Tensor std(Tensor input, long[] dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            => input.std(dimensions, unbiased, keepDimension, type);

        /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        public static Tensor std(Tensor input, long dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            => input.std(dimensions, unbiased, keepDimension, type);

        /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        public static Tensor std(Tensor input, (long,long) dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            => input.std(dimensions, unbiased, keepDimension, type);

        /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        public static Tensor std(Tensor input, (long,long,long) dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            => input.std(dimensions, unbiased, keepDimension, type);

        /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        public static (Tensor std, Tensor mean) std_mean(Tensor input, long[] dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            => input.std_mean(dimensions, unbiased, keepDimension, type);

        /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        public static (Tensor std, Tensor mean) std_mean(Tensor input, ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            => input.std_mean(dimensions, unbiased, keepDimension, type);

        /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        public static (Tensor std, Tensor mean) std_mean(Tensor input, long dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            => input.std_mean(dimensions, unbiased, keepDimension, type);

        /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        public static (Tensor std, Tensor mean) std_mean(Tensor input, (long,long) dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            => input.std_mean(dimensions, unbiased, keepDimension, type);

        /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepDimension">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        public static (Tensor std, Tensor mean) std_mean(Tensor input, (long,long,long) dimensions, bool unbiased = true, bool keepDimension = false, ScalarType? type = null)
            => input.std_mean(dimensions, unbiased, keepDimension, type);


        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor sub(Tensor left, Tensor right) => left.sub(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor sub(Tensor left, Scalar right) => left.sub(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor subtract(Tensor left, Tensor right) => left.sub(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor subtract(Tensor left, Scalar right) => left.sub(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor sub_(Tensor left, Tensor right) => left.sub_(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor sub_(Tensor left, Scalar right) => left.sub_(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <returns></returns>
        public static Tensor subtract_(Tensor left, Tensor right) => left.sub_(right);

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor subtract_(Tensor left, Scalar right) => left.sub_(right);

        /// <summary>
        /// Returns a new tensor with the truncated integer values of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor trunc(Tensor input) => input.trunc();

        /// <summary>
        /// Replaces each element with the truncated integer values of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor trunc_(Tensor input) => input.trunc_();

        /// <summary>
        /// Returns a new tensor with the truncated integer values of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor fix(Tensor input) => input.fix();

        /// <summary>
        /// Replaces each element with the truncated integer values of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor fix_(Tensor input) => input.fix_();

        /// <summary>
        /// Computes x * log(y)
        /// </summary>
        /// <param name="x">The 'x' operand.</param>
        /// <param name="y">The 'y' operand.</param>
        public static Tensor xlogy(Tensor x, Tensor y) => x.xlogy(y);

        /// <summary>
        /// Computes x * log(y)
        /// </summary>
        /// <param name="x">The 'x' operand.</param>
        /// <param name="y">The 'y' operand.</param>
        public static Tensor xlogy(Tensor x, Scalar y) => x.xlogy(y);

        /// <summary>
        /// Computes x * log(y) in place
        /// </summary>
        /// <param name="x">The 'x' operand.</param>
        /// <param name="y">The 'y' operand.</param>
        public static Tensor xlogy_(Tensor x, Tensor y) => x.xlogy_(y);

        /// <summary>
        /// Computes x * log(y) in place
        /// </summary>
        /// <param name="x">The 'x' operand.</param>
        /// <param name="y">The 'y' operand.</param>
        public static Tensor xlogy_(Tensor x, Scalar y) => x.xlogy_(y);


        // Duplication of random distribution opertors in the 'torch' namespace

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand_out(Tensor input, params long[] sizes) => input.randn_out(sizes);

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randint_out(Tensor input, long high, long[] sizes) => input.randint_out(high, sizes);

        /// <summary>
        /// Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0,1) .
        /// </summary>
        public static Tensor rand_like(Tensor input, ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false) => input.rand_like(dtype, device, requiresGrad);

        /// <summary>
        /// Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1. 
        /// </summary>
        public static Tensor randn_like(Tensor input, ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false) => input.randn_like(dtype, device, requiresGrad);

        /// <summary>
        /// Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly in the range [low,high).
        /// </summary>
        public static Tensor randint_like(Tensor input, long low, long high, ScalarType? dtype = null, torch.Device device = null, bool requiresGrad = false) => input.randint_like(low, high, dtype, device, requiresGrad);

        /// <summary>
        ///  Mutates the tensor to be a 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        public static Tensor randperm_out(Tensor input, long n) => input.randperm_out(n);

        /// <summary>
        /// Draws binary random numbers (0 or 1) from a Bernoulli distribution.
        /// </summary>
        /// <param name="input">The input tensor of probability values for the Bernoulli distribution</param>
        /// <param name="generator">Optional random number generator</param>
        /// <returns></returns>
        public static Tensor bernoulli(Tensor input, torch.Generator generator = null) => input.bernoulli(generator);

        /// <summary>
        /// Draws a binomial distribution given a trial count and probabilities.
        /// </summary>
        /// <param name="count">Trial count</param>
        /// <param name="probs">Probability vector</param>
        /// <param name="generator">Optional random number generator</param>
        /// <returns></returns>
        public static Tensor binomial(Tensor count, Tensor probs, torch.Generator generator = null) => count.binomial(probs, generator);

        /// <summary>
        /// Returns a tensor of the same size as input with each element sampled from a Poisson distribution with rate parameter given by the corresponding element in input
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="generator">Optional random number generator</param>
        /// <returns></returns>
        public static Tensor poisson(Tensor input, torch.Generator generator = null) => input.poisson(generator);

        /// <summary>
        /// Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.
        /// </summary>
        /// <param name="input">A probabilities tensor</param>
        /// <param name="num_samples">Number of samples to draw</param>
        /// <param name="replacement">Whether to draw with replacement or not</param>
        /// <param name="generator">Optional random number generator</param>
        public static Tensor multinomial(Tensor input, long num_samples, bool replacement = false, torch.Generator generator = null) => input.multinomial(num_samples, replacement, generator);

        /// <summary>
        /// Returns a tensor containing the result of Short-time Fourier transform (STFT).
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="n_fft">The size of Fourier transform</param>
        /// <param name="hop_length">The hop length</param>
        /// <param name="win_length">The window length</param>
        /// <param name="window">The window function</param>
        /// <param name="center">Whether the t-th frame is centered around t * hop_window, or not.</param>
        /// <param name="pad_mode">The padding mode used when center is true.</param>
        /// <param name="normalized">Whether the output is normalized, or not.</param>
        /// <param name="onesided">Whether the output is onesided or not.</param>
        /// <param name="return_complex">Whether a complex tensor is returned, or not.</param>
        /// <returns>A tensor containing the result of Short-time Fourier transform (STFT).</returns>
        public static Tensor stft(Tensor input, long n_fft, long hop_length = -1, long win_length = -1, Tensor window = null, bool center = true, PaddingModes pad_mode = PaddingModes.Reflect, bool normalized = false, bool? onesided = null, bool? return_complex = null) => input.stft(n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided, return_complex);

        /// <summary>
        /// Returns a tensor containing the result of Inverse Short-time Fourier transform.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="n_fft">The size of Fourier transform</param>
        /// <param name="hop_length">The hop length</param>
        /// <param name="win_length">The window length</param>
        /// <param name="window">The window function</param>
        /// <param name="center">Whether the t-th frame is centered around t * hop_window, or not.</param>
        /// <param name="normalized">Whether the output is normalized, or not.</param>
        /// <param name="onesided">Whether the output is onesided or not.</param>
        /// <param name="length">The length of the output tensor.</param>
        /// <param name="return_complex">Whether a complex tensor is returned, or not.</param>
        /// <returns>A tensor containing the result of Inverse Short-time Fourier transform</returns>
        public static Tensor istft(Tensor input, long n_fft, long hop_length = -1, long win_length = -1, Tensor window = null, bool center = true, bool normalized = false, bool? onesided = null, long length = -1, bool return_complex = false) => input.istft(n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);
    }
}
