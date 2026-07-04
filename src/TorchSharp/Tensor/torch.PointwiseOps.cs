// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using ICSharpCode.SharpZipLib.BZip2;

namespace TorchSharp
{
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.abs
        /// <summary>
        /// Compute the absolute value of each element in the tensor
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor abs(Tensor input) => input.abs();

        // https://pytorch.org/docs/stable/generated/torch.abs
        /// <summary>
        /// Compute the absolute value of each element in the tensor, in-place
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor abs_(Tensor input) => input.abs_();

        // https://pytorch.org/docs/stable/generated/torch.absolute
        /// <summary>
        /// Compute the absolute value of each element in the tensor
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor absolute(Tensor input) => input.absolute();

        // https://pytorch.org/docs/stable/generated/torch.absolute
        /// <summary>
        /// Compute the absolute value of each element in the tensor, in-place
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor absolute_(Tensor input) => input.absolute_();

        // https://pytorch.org/docs/stable/generated/torch.acos
        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor acos(Tensor input) => input.acos();

        // https://pytorch.org/docs/stable/generated/torch.acos
        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor acos_(Tensor input) => input.acos_();

        // https://pytorch.org/docs/stable/generated/torch.arccos
        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor arccos(Tensor input) => input.arccos();

        // https://pytorch.org/docs/stable/generated/torch.arccos
        /// <summary>
        /// Computes the arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arccos_(Tensor input) => input.arccos_();

        // https://pytorch.org/docs/stable/generated/torch.acosh
        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor acosh(Tensor input) => input.acosh();

        // https://pytorch.org/docs/stable/generated/torch.acosh
        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor acosh_(Tensor input) => input.acosh_();

        // https://pytorch.org/docs/stable/generated/torch.arccosh
        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor arccosh(Tensor input) => input.arccosh();

        // https://pytorch.org/docs/stable/generated/torch.arccosh
        /// <summary>
        /// Computes the hyperbolic arccosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arccosh_(Tensor input) => input.arccosh_();

        // https://pytorch.org/docs/stable/generated/torch.add
        /// <summary>
        /// Add two tensors, element-wise
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        /// <returns></returns>
        [Pure]public static Tensor add(Tensor left, Tensor right) => left.add(right);

        // https://pytorch.org/docs/stable/generated/torch.add
        /// <summary>
        /// Add a scalar value to each element in the target tensor.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]public static Tensor add(Tensor left, Scalar right) => left.add(right);

        // https://pytorch.org/docs/stable/generated/torch.add
        /// <summary>
        /// Add two tensors, element-wise, scaling the second operator by 'alpha'
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        /// <param name="alpha">RHS scale factor.</param>
        [Pure]public static Tensor add(Tensor left, Tensor right, Scalar alpha) => left.add(right, alpha);

        // https://pytorch.org/docs/stable/generated/torch.add
        /// <summary>
        /// Add a scalar value to each element in the target tensor, scaled by 'alpha'
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        /// <param name="alpha">RHS scale factor.</param>
        [Pure]public static Tensor add(Tensor left, Scalar right, Scalar alpha) => left.add(right, alpha);

        // https://pytorch.org/docs/stable/generated/torch.add
        /// <summary>
        /// Add two tensors, element-wise, in place
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor add_(Tensor left, Tensor right) => left.add_(right);

        // https://pytorch.org/docs/stable/generated/torch.add
        /// <summary>
        /// Add a scalar value to each element in the target tensor, in place.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor add_(Tensor left, Scalar right) => left.add_(right);

        // https://pytorch.org/docs/stable/generated/torch.add
        /// <summary>
        /// Add two tensors, element-wise, scaling the second operator by 'alpha', in place
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        /// <param name="alpha">RHS scale factor.</param>
        public static Tensor add_(Tensor left, Tensor right, Scalar alpha) => left.add_(right, alpha);

        // https://pytorch.org/docs/stable/generated/torch.add
        /// <summary>
        /// Add a scalar value to each element in the target tensor, scaled by 'alpha', in place
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        /// <param name="alpha">RHS scale factor.</param>
        public static Tensor add_(Tensor left, Scalar right, Scalar alpha) => left.add_(right, alpha);

        // https://pytorch.org/docs/stable/generated/torch.addcdiv
        /// <summary>
        /// Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="tensor1">First tensor</param>
        /// <param name="tensor2">Second tensor</param>
        /// <param name="value">Scale factor</param>
        [Pure]public static Tensor addcdiv(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value)
            => input.addcdiv(tensor1, tensor2, value);

        // https://pytorch.org/docs/stable/generated/torch.addcdiv
        /// <summary>
        /// Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// In-place version of <see cref="addcdiv"/>.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="tensor1">First tensor</param>
        /// <param name="tensor2">Second tensor</param>
        /// <param name="value">Scale factor</param>
        public static Tensor addcdiv_(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value)
            => input.addcdiv_(tensor1, tensor2, value);

        // https://pytorch.org/docs/stable/generated/torch.addcmul
        /// <summary>
        /// Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="tensor1">First tensor</param>
        /// <param name="tensor2">Second tensor</param>
        /// <param name="value">Scale factor</param>
        [Pure]public static Tensor addcmul(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value)
            => input.addcmul(tensor1, tensor2, value);

        // https://pytorch.org/docs/stable/generated/torch.addcmul
        /// <summary>
        /// Performs the element-wise divismultiplicationion of tensor1 by tensor2, multiply the result by the scalar value and add it to input.
        /// In-place version of <see cref="addcmul"/>.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="tensor1">First tensor</param>
        /// <param name="tensor2">Second tensor</param>
        /// <param name="value">Scale factor</param>
        public static Tensor addcmul_(Tensor input, Tensor tensor1, Tensor tensor2, Scalar value)
            => input.addcmul_(tensor1, tensor2, value);

        // https://pytorch.org/docs/stable/generated/torch.angle
        /// <summary>
        /// Computes the element-wise angle (in radians) of the given input tensor.
        /// </summary>
        /// <returns></returns>
        /// <remarks>
        /// Starting in Torch 1.8, angle returns pi for negative real numbers, zero for non-negative real numbers, and propagates NaNs.
        /// Previously the function would return zero for all real numbers and not propagate floating-point NaNs.
        /// </remarks>
        [Pure]public static Tensor angle(Tensor input) => input.angle();

        // https://pytorch.org/docs/stable/generated/torch.asin
        /// <summary>
        /// Computes the arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor asin(Tensor input) => input.asin();

        // https://pytorch.org/docs/stable/generated/torch.asin
        /// <summary>
        /// Computes the arcsine of the elements of input, in place.
        /// </summary>
        /// <returns></returns>
        public static Tensor asin_(Tensor input) => input.asin_();

        // https://pytorch.org/docs/stable/generated/torch.arcsin
        /// <summary>
        /// Computes the arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor arcsin(Tensor input) => input.arcsin();

        // https://pytorch.org/docs/stable/generated/torch.arcsin
        /// <summary>
        /// Computes the arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arcsin_(Tensor input) => input.arcsin_();

        // https://pytorch.org/docs/stable/generated/torch.asinh
        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor asinh(Tensor input) => input.asinh();

        // https://pytorch.org/docs/stable/generated/torch.asinh
        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor asinh_(Tensor input) => input.asinh_();

        // https://pytorch.org/docs/stable/generated/torch.arcsinh
        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor arcsinh(Tensor input) => input.arcsinh();

        // https://pytorch.org/docs/stable/generated/torch.arcsinh
        /// <summary>
        /// Computes the hyperbolic arcsine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arcsinh_(Tensor input) => input.arcsinh_();

        // https://pytorch.org/docs/stable/generated/torch.atan
        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor atan(Tensor input) => input.atan();

        // https://pytorch.org/docs/stable/generated/torch.atan
        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor atan_(Tensor input) => input.atan_();

        // https://pytorch.org/docs/stable/generated/torch.arctan
        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor arctan(Tensor input) => input.arctan();

        // https://pytorch.org/docs/stable/generated/torch.arctan
        /// <summary>
        /// Computes the arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arctan_(Tensor input) => input.arctan_();

        // https://pytorch.org/docs/stable/generated/torch.atanh
        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor atanh(Tensor input) => input.atanh();

        // https://pytorch.org/docs/stable/generated/torch.atanh
        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor atanh_(Tensor input) => input.atanh_();

        // https://pytorch.org/docs/stable/generated/torch.arctanh
        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor arctanh(Tensor input) => input.arctanh();

        /// <summary>
        /// Computes the hyperbolic arctangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor arctanh_(Tensor input) => input.arctanh_();

        // https://pytorch.org/docs/stable/generated/torch.atan2
        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor atan2(Tensor input, Tensor other) => input.atan2(other);

        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <returns></returns>
        public static Tensor atan2_(Tensor input, Tensor other) => input.atan2_(other);

        // https://pytorch.org/docs/stable/generated/torch.arctan2
        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor arctan2(Tensor input, Tensor other) => input.arctan2(other);

        // https://pytorch.org/docs/stable/generated/torch.arctan2
        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <returns></returns>
        public static Tensor arctan2_(Tensor input, Tensor other) => input.arctan2_(other);

        // https://pytorch.org/docs/stable/generated/torch.arctan
        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor arctan(Tensor input, Tensor other) => input.arctan(other);

        // https://pytorch.org/docs/stable/generated/torch.arctan
        /// <summary>
        /// Element-wise arctangent of input / other with consideration of the quadrant.
        /// </summary>
        /// <returns></returns>
        public static Tensor arctan_(Tensor input, Tensor other) => input.arctan_(other);

        // https://pytorch.org/docs/stable/generated/torch.bitwise_not
        /// <summary>
        /// Element-wise bitwise NOT
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor bitwise_not(Tensor input) => input.bitwise_not();

        // https://pytorch.org/docs/stable/generated/torch.bitwise_not
        /// <summary>
        /// Element-wise bitwise NOT, in place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor bitwise_not_(Tensor input) => input.bitwise_not_();

        // https://pytorch.org/docs/stable/generated/torch.bitwise_and
        /// <summary>
        /// Element-wise bitwise AND
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        [Pure]public static Tensor bitwise_and(Tensor left, Tensor right) => left.bitwise_and(right);

        // https://pytorch.org/docs/stable/generated/torch.bitwise_and
        /// <summary>
        /// Element-wise bitwise AND, in place.
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_and_(Tensor left, Tensor right) => left.bitwise_and_(right);

        // https://pytorch.org/docs/stable/generated/torch.bitwise_or
        /// <summary>
        /// Element-wise bitwise OR
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        [Pure]public static Tensor bitwise_or(Tensor left, Tensor right) => left.bitwise_or(right);

        // https://pytorch.org/docs/stable/generated/torch.bitwise_or
        /// <summary>
        /// Element-wise bitwiseXOR, in place.
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_or_(Tensor left, Tensor right) => left.bitwise_or_(right);

        // https://pytorch.org/docs/stable/generated/torch.bitwise_xor
        /// <summary>
        /// Element-wise bitwise XOR
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        [Pure]public static Tensor bitwise_xor(Tensor left, Tensor right) => left.bitwise_xor(right);

        // https://pytorch.org/docs/stable/generated/torch.bitwise_xor
        /// <summary>
        /// Element-wise bitwise XOR, in place.
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_xor_(Tensor left, Tensor right) => left.bitwise_xor_(right);

        // https://pytorch.org/docs/stable/generated/torch.bitwise_left_shift
        /// <summary>
        /// Element-wise bitwise left shift
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_left_shift(Tensor left, Tensor right) => left.bitwise_left_shift(right);

        // https://pytorch.org/docs/stable/generated/torch.bitwise_left_shift
        /// <summary>
        /// Element-wise bitwise left shift, in place.
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_left_shift_(Tensor left, Tensor right) => left.bitwise_left_shift_(right);

        // https://pytorch.org/docs/stable/generated/torch.bitwise_right_shift
        /// <summary>
        /// Element-wise bitwise right shift
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        [Pure]public static Tensor bitwise_right_shift(Tensor left, Tensor right) => left.bitwise_right_shift(right);

        // https://pytorch.org/docs/stable/generated/torch.bitwise_right_shift
        /// <summary>
        /// Element-wise bitwise right shift, in place.
        /// </summary>
        /// <param name="left">Left-hand operand.</param>
        /// <param name="right">Right-hand operand.</param>
        public static Tensor bitwise_right_shift_(Tensor left, Tensor right) => left.bitwise_right_shift_(right);

        // https://pytorch.org/docs/stable/generated/torch.ceil
        /// <summary>
        /// Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor ceil(Tensor input) => input.ceil();

        // https://pytorch.org/docs/stable/generated/torch.ceil
        /// <summary>
        /// Replaces each element of the input with the smallest integer greater than or equal to the element.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor ceil_(Tensor input) => input.ceil_();

        // https://pytorch.org/docs/stable/generated/torch.clamp
        /// <summary>
        /// Clamps all elements in input into the range [ min, max ].
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="min">The minimum value</param>
        /// <param name="max">The maximum value</param>
        public static Tensor clamp(Tensor input, Scalar? min = null, Scalar? max = null) => input.clamp(min, max);

        // https://pytorch.org/docs/stable/generated/torch.clamp
        /// <summary>
        /// Clamps all elements in input into the range [ min, max ] in place.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="min">The minimum value</param>
        /// <param name="max">The maximum value</param>
        public static Tensor clamp_(Tensor input, Scalar? min = null, Scalar? max = null) => input.clamp_(min, max);

        // https://pytorch.org/docs/stable/generated/torch.clamp
        /// <summary>
        /// Clamps all elements in input into the range [ min, max ].
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="min">The minimum value</param>
        /// <param name="max">The maximum value</param>
        public static Tensor clamp(Tensor input, Tensor? min = null, Tensor? max = null) => input.clamp(min, max);

        // https://pytorch.org/docs/stable/generated/torch.clamp
        /// <summary>
        /// Clamps all elements in input into the range [ min, max ] in place.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="min">The minimum value</param>
        /// <param name="max">The maximum value</param>
        public static Tensor clamp_(Tensor input, Tensor? min = null, Tensor? max = null) => input.clamp_(min, max);

        // https://pytorch.org/docs/stable/generated/torch.clip
        public static Tensor clip(Tensor input, Scalar? min = null, Scalar? max = null) => input.clip(min, max);

        // https://pytorch.org/docs/stable/generated/torch.conj_physical
        /// <summary>
        /// Computes the element-wise conjugate of the given input tensor. If input has a non-complex <see cref="DeviceType">dtype</see>, this function just returns input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor conj_physical(Tensor input) => input.conj_physical();

        // https://pytorch.org/docs/stable/generated/torch.conj_physical
        /// <summary>
        /// In-place version of conj_physical
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor conj_physical_(Tensor input) => input.conj_physical_();

        // https://pytorch.org/docs/stable/generated/torch.copysign
        /// <summary>
        /// Create a new floating-point tensor with the magnitude of input and the sign of other, elementwise.
        /// Supports broadcasting to a common shape, and integer and float inputs.
        /// </summary>
        /// <param name="input">magnitudes</param>
        /// <param name="other">contains value(s) whose signbit(s) are applied to the magnitudes in <paramref name="input"/>.</param>
        /// <returns>the output tensor</returns>
        [Pure]public static Tensor copysign(Tensor input, Tensor other) => input.copysign(other);

        // https://pytorch.org/docs/stable/generated/torch.copysign
        /// <summary>
        /// Copies the signs of other to input, in-place.
        /// </summary>
        public static Tensor copysign_(Tensor input, Tensor other) => input.copysign_(other);

        // https://pytorch.org/docs/stable/generated/torch.cos
        /// <summary>
        /// Computes the cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor cos(Tensor input) => input.cos();

        /// <summary>
        /// Computes the cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor cos_(Tensor input) => input.cos_();

        // https://pytorch.org/docs/stable/generated/torch.cosh
        /// <summary>
        /// Computes the hyperbolic cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor cosh(Tensor input) => input.cosh();

        // https://pytorch.org/docs/stable/generated/torch.cosh
        /// <summary>
        /// Computes the hyperbolic cosine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor cosh_(Tensor input) => input.cosh_();

        // https://pytorch.org/docs/stable/generated/torch.deg2rad
        [Pure]public static Tensor deg2rad(Tensor input) => input.deg2rad();

        // https://pytorch.org/docs/stable/generated/torch.deg2rad
        /// <summary>
        /// Convert each element from degrees to radians, in-place.
        /// </summary>
        public static Tensor deg2rad_(Tensor input) => input.deg2rad_();

        // https://pytorch.org/docs/stable/generated/torch.div
        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor div(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        // https://pytorch.org/docs/stable/generated/torch.div
        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor div(Tensor left, Scalar right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        // https://pytorch.org/docs/stable/generated/torch.div
        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor div_(Tensor left, Scalar right, RoundingMode rounding_mode = RoundingMode.None) => left.div_(right, rounding_mode);

        // https://pytorch.org/docs/stable/generated/torch.div
        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor div_(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div_(right);

        // https://pytorch.org/docs/stable/generated/torch.divide
        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor divide(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        // https://pytorch.org/docs/stable/generated/torch.divide
        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor divide(Tensor left, Scalar right, RoundingMode rounding_mode = RoundingMode.None) => left.div(right);

        // https://pytorch.org/docs/stable/generated/torch.divide
        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        /// <param name="rounding_mode">Rounding mode.</param>
        public static Tensor divide_(Tensor left, Tensor right, RoundingMode rounding_mode = RoundingMode.None) => left.div_(right);

        // https://pytorch.org/docs/stable/generated/torch.divide
        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor divide_(Tensor left, Scalar right) => left.div_(right);

        // https://pytorch.org/docs/stable/generated/torch.digamma
        /// <summary>
        /// Computes the logarithmic derivative of the gamma function on input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor digamma(Tensor input) => input.digamma();

        // https://pytorch.org/docs/stable/generated/torch.digamma
        /// <summary>
        /// Computes the logarithmic derivative of the gamma function on input, in place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor digamma_(Tensor input) => input.digamma_();

        // https://pytorch.org/docs/stable/generated/torch.erf
        /// <summary>
        /// Computes the error function of the input.
        /// </summary>
        public static Tensor erf(Tensor input) => input.erf();

        // https://pytorch.org/docs/stable/generated/torch.erf
        /// <summary>
        /// Computes the error function of the input in place.
        /// </summary>
        public static Tensor erf_(Tensor input) => input.erf_();

        // https://pytorch.org/docs/stable/generated/torch.erfc
        /// <summary>
        /// Computes the error function of the input.
        /// </summary>
        public static Tensor erfc(Tensor input) => input.erfc();

        // https://pytorch.org/docs/stable/generated/torch.erfc
        /// <summary>
        /// Computes the error function of the input in place.
        /// </summary>
        public static Tensor erfc_(Tensor input) => input.erfc_();

        // https://pytorch.org/docs/stable/generated/torch.erfinv
        /// <summary>
        /// Computes the error function of the input.
        /// </summary>
        public static Tensor erfinv(Tensor input) => input.erf();

        // https://pytorch.org/docs/stable/generated/torch.erfinv
        /// <summary>
        /// Computes the error function of the input in place.
        /// </summary>
        public static Tensor erfinv_(Tensor input) => input.erfinv_();

        // https://pytorch.org/docs/stable/generated/torch.exp
        /// <summary>
        /// Returns a new tensor with the exponential of the elements of the input tensor input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor exp(Tensor input) => input.exp();

        // https://pytorch.org/docs/stable/generated/torch.exp
        /// <summary>
        /// Replaces each element of the input with the exponential of the elements of the input tensor input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor exp_(Tensor input) => input.exp_();

        // https://pytorch.org/docs/stable/generated/torch.exp2
        /// <summary>
        /// Computes the base 2 exponential function of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor exp2(Tensor input) => input.exp2();

        // https://pytorch.org/docs/stable/generated/torch.exp2
        /// <summary>
        /// Computes the base 2 exponential function of input, in-place.
        /// </summary>
        public static Tensor exp2_(Tensor input) => input.exp2_();

        // https://pytorch.org/docs/stable/generated/torch.expm1
        /// <summary>
        /// Returns a new tensor with the exponential of the elements minus 1 of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor expm1(Tensor input) => input.expm1();

        // https://pytorch.org/docs/stable/generated/torch.expm1
        /// <summary>
        /// Replaces each element with the exponential of the element minus 1 of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor expm1_(Tensor input) => input.expm1_();

        // https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_channel_affine
        /// <summary>
        /// Returns a new tensor with the data in <paramref name="input"/> fake quantized per channel using
        /// <paramref name="scale"/>, <paramref name="zero_point"/>, <paramref name="quant_min"/> and <paramref name="quant_max"/>,
        /// across the channel specified by <paramref name="axis"/>.
        /// </summary>
        /// <param name="input">the input value(s) (float32)</param>
        /// <param name="scale">quantization scale, per channel (float32)</param>
        /// <param name="zero_point">quantization zero_point, per channel (torch.int32, torch.half, or torch.float32)</param>
        /// <param name="axis">channel axis</param>
        /// <param name="quant_min">lower bound of the quantized domain</param>
        /// <param name="quant_max">upper bound of the quantized domain</param>
        /// <returns>A newly fake_quantized per channel torch.float32 tensor</returns>
        [Pure, Obsolete("not implemented", true)]
        public static Tensor fake_quantize_per_channel_affine(Tensor input, Tensor scale, Tensor zero_point, int axis, long quant_min, long quant_max)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_tensor_affine
        /// <summary>
        /// Returns a new tensor with the data in <paramref name="input"/> fake quantized per channel using
        /// <paramref name="scale"/>, <paramref name="zero_point"/>, <paramref name="quant_min"/> and <paramref name="quant_max"/>,
        /// across the channel specified by axis.
        /// </summary>
        /// <param name="input">the input value(s) (float32)</param>
        /// <param name="scale">quantization scale, per channel (float32)</param>
        /// <param name="zero_point">quantization zero_point, per channel (torch.int32, torch.half, or torch.float32)</param>
        /// <param name="quant_min">lower bound of the quantized domain</param>
        /// <param name="quant_max">upper bound of the quantized domain</param>
        /// <returns>A newly fake_quantized per channel torch.float32 tensor</returns>
        [Pure, Obsolete("not implemented", true)]
        public static Tensor fake_quantize_per_tensor_affine(Tensor input, Tensor scale, Tensor zero_point, long quant_min, long quant_max)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor
        /// <summary>
        /// Converts a float tensor to a quantized tensor with given scale and zero point.
        /// </summary>
        /// <param name="input">Float tensor to quantize</param>
        /// <param name="scale">Scale to apply in quantization formula</param>
        /// <param name="zero_point">Offset in integer value that maps to float zero</param>
        /// <param name="dtype">The desired data type of returned tensor. Must be a quantized type (torch.qint8, torch.quint8, or torch.qint32).</param>
        /// <returns>A newly quantized tensor</returns>
        public static Tensor quantize_per_tensor(Tensor input, double scale, long zero_point, ScalarType dtype)
        {
            if (!is_quantized(dtype))
                throw new ArgumentException("dtype must be a quantized type (QInt8, QUInt8, or QInt32)", nameof(dtype));
            return input._quantize_per_tensor(scale, zero_point, dtype);
        }

        // https://pytorch.org/docs/stable/generated/torch.quantize_per_channel
        /// <summary>
        /// Converts a float tensor to a per-channel quantized tensor with given scales and zero points.
        /// </summary>
        /// <param name="input">Float tensor to quantize</param>
        /// <param name="scales">Float 1D tensor of scales to use, size should match input.size(axis)</param>
        /// <param name="zero_points">Integer 1D tensor of offsets to use, size should match input.size(axis)</param>
        /// <param name="axis">Dimension on which to apply per-channel quantization</param>
        /// <param name="dtype">The desired data type of returned tensor. Must be a quantized type (torch.qint8, torch.quint8, or torch.qint32).</param>
        /// <returns>A newly quantized tensor</returns>
        public static Tensor quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, long axis, ScalarType dtype)
        {
            if (!is_quantized(dtype))
                throw new ArgumentException("dtype must be a quantized type (QInt8, QUInt8, or QInt32)", nameof(dtype));
            return input._quantize_per_channel(scales, zero_points, axis, dtype);
        }

        // https://pytorch.org/docs/stable/generated/torch.dequantize
        /// <summary>
        /// Returns an fp32 Tensor by dequantizing a quantized Tensor.
        /// </summary>
        /// <param name="input">A quantized tensor</param>
        /// <returns>A dequantized (float) tensor</returns>
        public static Tensor dequantize(Tensor input) => input.dequantize();

        // https://pytorch.org/docs/stable/generated/torch.fix
        /// <summary>
        /// Returns a new tensor with the truncated integer values of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor fix(Tensor input) => input.fix();

        // https://pytorch.org/docs/stable/generated/torch.fix
        /// <summary>
        /// Replaces each element with the truncated integer values of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor fix_(Tensor input) => input.fix_();

        // https://pytorch.org/docs/stable/generated/torch.float_power
        /// <summary>
        /// Raises input to the power of exponent, elementwise, in double precision.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="target">The exponent.</param>
        /// <remarks> If neither input is complex returns a torch.float64 tensor, and if one or more inputs is complex returns a torch.complex128 tensor.</remarks>
        [Pure]public static Tensor float_power(Tensor input, Tensor target) => input.float_power(target);

        // https://pytorch.org/docs/stable/generated/torch.float_power
        /// <summary>
        /// Raises input to the power of exponent, elementwise, in double precision, in-place.
        /// </summary>
        public static Tensor float_power_(Tensor input, Tensor target) => input.float_power_(target);

        // https://pytorch.org/docs/stable/generated/torch.floor
        /// <summary>
        /// Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor floor(Tensor input) => input.floor();

        // https://pytorch.org/docs/stable/generated/torch.floor
        /// <summary>
        /// Replaces each element with the floor of the input, the largest integer less than or equal to each element.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor floor_(Tensor input) => input.floor_();

        // https://pytorch.org/docs/stable/generated/torch.floor_divide
        /// <summary>
        /// Computes input divided by other, elementwise, and floors the result.
        /// Supports broadcasting to a common shape, type promotion, and integer and float inputs.
        /// </summary>
        /// <param name="input">the dividend</param>
        /// <param name="other">the divisor</param>
        /// <returns>the output tensor</returns>
        [Pure]
        public static Tensor floor_divide(Tensor input, Tensor other) => input.floor_divide(other);

        // https://pytorch.org/docs/stable/generated/torch.floor_divide
        /// <summary>
        /// Computes input divided by other, elementwise, and floors the result.
        /// Supports broadcasting to a common shape, type promotion, and integer and float inputs.
        /// </summary>
        /// <param name="input">the dividend</param>
        /// <param name="other">the divisor</param>
        /// <returns>the output tensor</returns>
        public static Tensor floor_divide_(Tensor input, Tensor other) => input.floor_divide_(other);

        // https://pytorch.org/docs/stable/generated/torch.fmod
        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        [Pure]public static Tensor fmod(Tensor left, Tensor right) => left.fmod(right);

        // https://pytorch.org/docs/stable/generated/torch.fmod
        /// <summary>
        /// Computes the element-wise remainder of division, in place.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor fmod_(Tensor left, Tensor right) => left.fmod_(right);

        // https://pytorch.org/docs/stable/generated/torch.fmod
        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        [Pure]public static Tensor fmod(Tensor left, Scalar right) => left.fmod(right);

        // https://pytorch.org/docs/stable/generated/torch.fmod
        /// <summary>
        /// Computes the element-wise remainder of division, in place.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor fmod_(Tensor left, Scalar right) => left.fmod_(right);

        // https://pytorch.org/docs/stable/generated/torch.frac
        /// <summary>
        /// Computes the fractional portion of each element in input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor frac(Tensor input) => input.frac();

        // https://pytorch.org/docs/stable/generated/torch.frac
        /// <summary>
        /// Computes the fractional portion of each element in input, in-place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor frac_(Tensor input) => input.frac_();

        // https://pytorch.org/docs/stable/generated/torch.frexp
        /// <summary>
        /// Decomposes input into mantissa and exponent tensors
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static (Tensor Mantissa, Tensor Exponent) frexp(Tensor input) => input.frexp();

        // https://pytorch.org/docs/stable/generated/torch.gradient
        [Pure, Obsolete("not implemented", true)]
        public static ICollection<Tensor> gradient(Tensor input, int spacing = 1, long? dim = null, int edge_order = 1)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.gradient
        [Pure, Obsolete("not implemented", true)]
        public static ICollection<Tensor> gradient(Tensor input, int spacing = 1, long[]? dims = null, int edge_order = 1)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.heaviside
        /// <summary>
        /// Computes the Heaviside step function for each element in input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="values">The values to use where input is zero.</param>
        [Pure]public static Tensor heaviside(Tensor input, Tensor values) => input.heaviside(values);

        // https://pytorch.org/docs/stable/generated/torch.heaviside
        /// <summary>
        /// Computes the Heaviside step function for each element in input, in-place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="values">The values to use where input is zero.</param>
        public static Tensor heaviside_(Tensor input, Tensor values) => input.heaviside_(values);

        // https://pytorch.org/docs/stable/generated/torch.imag
        [Pure]public static Tensor imag(Tensor input) => input.imag;

        // https://pytorch.org/docs/stable/generated/torch.ldexp
        /// <summary>
        /// Multiplies input by pow(2,other).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="other">A tensor of exponents, typically integers</param>
        /// <remarks>Typically this function is used to construct floating point numbers by multiplying mantissas in input with integral powers of two created from the exponents in other.</remarks>
        [Pure]public static Tensor ldexp(Tensor input, Tensor other) => input.ldexp(other);

        // https://pytorch.org/docs/stable/generated/torch.ldexp
        /// <summary>
        /// Multiplies input by pow(2,other) in place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="other">A tensor of exponents, typically integers</param>
        /// <remarks>Typically this function is used to construct floating point numbers by multiplying mantissas in input with integral powers of two created from the exponents in other.</remarks>
        public static Tensor ldexp_(Tensor input, Tensor other) => input.ldexp_(other);

        // https://pytorch.org/docs/stable/generated/torch.lerp
        /// <summary>
        /// Does a linear interpolation of two tensors start (given by input)
        /// and end based on a scalar or tensor weight and returns the resulting out tensor.
        /// </summary>
        /// <remarks>
        /// The shapes of start and end must be broadcastable.
        /// If weight is a tensor, then the shapes of weight, start, and end must be broadcastable.
        /// </remarks>
        /// <param name="input">the tensor with the starting points</param>
        /// <param name="end">the tensor with the ending points</param>
        /// <param name="weight">the weight for the interpolation formula</param>
        /// <returns>the output tensor</returns>
        [Pure]public static Tensor lerp(Tensor input, Tensor end, Tensor weight) => input.lerp(end, weight);

        // https://pytorch.org/docs/stable/generated/torch.lgamma
        /// <summary>
        /// Computes the logarithm of the gamma function on input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor lgamma(Tensor input) => input.lgamma();

        // https://pytorch.org/docs/stable/generated/torch.lgamma
        /// <summary>
        /// Computes the logarithm of the gamma function on input, in place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor lgamma_(Tensor input) => input.lgamma();

        // https://pytorch.org/docs/stable/generated/torch.log
        /// <summary>
        /// Returns a new tensor with the natural logarithm of the input elements.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor log(Tensor input) => input.log();

        // https://pytorch.org/docs/stable/generated/torch.log
        /// <summary>
        /// Replaces each elements with the natural logarithm of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log_(Tensor input) => input.log_();

        // https://pytorch.org/docs/stable/generated/torch.log10
        /// <summary>
        /// Returns a new tensor with the logarithm to the base 10 of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor log10(Tensor input) => input.log10();

        // https://pytorch.org/docs/stable/generated/torch.log10
        /// <summary>
        /// Replaces each elements with the logarithm to the base 10 of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log10_(Tensor input) => input.log10_();

        // https://pytorch.org/docs/stable/generated/torch.log1p
        /// <summary>
        /// Returns a new tensor with the natural logarithm of (1 + input).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor log1p(Tensor input) => input.log1p();

        // https://pytorch.org/docs/stable/generated/torch.log1p
        /// <summary>
        /// Replaces each elements with the natural logarithm of (1 + input), in place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log1p_(Tensor input) => input.log1p_();

        // https://pytorch.org/docs/stable/generated/torch.log2
        /// <summary>
        /// Returns a new tensor with the logarithm to the base 10 of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor log2(Tensor input) => input.log2();

        // https://pytorch.org/docs/stable/generated/torch.log2
        /// <summary>
        /// Replaces each elements with the logarithm to the base 10 of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor log2_(Tensor input) => input.log2_();

        // https://pytorch.org/docs/stable/generated/torch.logaddexp
        /// <summary>
        /// Logarithm of the sum of exponentiations of the inputs.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]public static Tensor logaddexp(Tensor left, Tensor right) => left.logaddexp(right);

        // https://pytorch.org/docs/stable/generated/torch.logaddexp2
        /// <summary>
        /// Logarithm of the sum of exponentiations of the inputs in base-2.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]public static Tensor logaddexp2(Tensor left, Tensor right) => left.logaddexp2(right);

        // https://pytorch.org/docs/stable/generated/torch.logical_and
        /// <summary>
        /// Element-wise logical AND
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]public static Tensor logical_and(Tensor left, Tensor right) => left.logical_and(right);

        // https://pytorch.org/docs/stable/generated/torch.logical_and
        /// <summary>
        /// Element-wise logical AND, in place.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor logical_and_(Tensor left, Tensor right) => left.logical_and_(right);

        // https://pytorch.org/docs/stable/generated/torch.logical_not
        /// <summary>
        /// Element-wise logical NOT
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor logical_not(Tensor input) => input.logical_not();

        // https://pytorch.org/docs/stable/generated/torch.logical_or
        /// <summary>
        /// Element-wise logical OR
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]public static Tensor logical_or(Tensor left, Tensor right) => left.logical_or(right);

        // https://pytorch.org/docs/stable/generated/torch.logical_or
        /// <summary>
        /// Element-wise logicalXOR, in place.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor logical_or_(Tensor left, Tensor right) => left.logical_or_(right);

        // https://pytorch.org/docs/stable/generated/torch.logical_xor
        /// <summary>
        /// Element-wise logical XOR
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]public static Tensor logical_xor(Tensor left, Tensor right) => left.logical_xor(right);

        // https://pytorch.org/docs/stable/generated/torch.logical_xor
        /// <summary>
        /// Element-wise logical XOR
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor logical_xor_(Tensor left, Tensor right) => left.logical_xor_(right);

        // https://pytorch.org/docs/stable/generated/torch.logit
        /// <summary>
        /// Returns a new tensor with the logit of the elements of input.
        /// input is clamped to [eps, 1 - eps] when eps is not null
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="eps">The epsilon for input clamp bound.</param>
        [Pure]public static Tensor logit(Tensor input, double? eps = null) => input.logit(eps);

        // https://pytorch.org/docs/stable/generated/torch.logit
        /// <summary>
        /// Returns the logit of the elements of input, in-place.
        /// input is clamped to [eps, 1 - eps] when eps is not null.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="eps">The epsilon for input clamp bound.</param>
        public static Tensor logit_(Tensor input, double? eps = null) => input.logit_(eps);

        // https://pytorch.org/docs/stable/generated/torch.hypot
        /// <summary>
        /// Element-wise: given the legs of a right triangle, return its hypotenuse.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]public static Tensor hypot(Tensor left, Tensor right) => left.hypot(right);

        // https://pytorch.org/docs/stable/generated/torch.i0
        /// <summary>
        /// Alias for <see cref="torch.special.i0"/>.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        [Pure]public static Tensor i0(Tensor input) => special.i0(input);

        // https://pytorch.org/docs/stable/generated/torch.i0
        /// <summary>
        /// Computes the zeroth order modified Bessel function of the first kind for each element of input, in-place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor i0_(Tensor input) => input.i0_();

        // https://pytorch.org/docs/stable/generated/torch.igamma
        /// <summary>
        /// Alias for <see cref="torch.special.gammainc"/>.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="other"></param>
        /// <returns></returns>
        [Pure]public static Tensor igamma(Tensor input, Tensor other) => special.gammainc(input, other);

        // https://pytorch.org/docs/stable/generated/torch.igammac
        /// <summary>
        /// Alias for <see cref="torch.special.gammaincc"/>".
        /// </summary>
        /// <param name="input"></param>
        /// <param name="other"></param>
        /// <returns></returns>
        [Pure]public static Tensor igammac(Tensor input, Tensor other) => special.gammaincc(input, other);

        // https://pytorch.org/docs/stable/generated/torch.mul
        /// <summary>
        /// Multiplies each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor mul(Tensor left, Tensor right) => left.mul(right);

        // https://pytorch.org/docs/stable/generated/torch.mul
        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor mul(Tensor left, Scalar right) => left.mul(right);

        // https://pytorch.org/docs/stable/generated/torch.mul
        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor mul_(Tensor left, Tensor right) => left.mul_(right);

        // https://pytorch.org/docs/stable/generated/torch.mul
        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor mul_(Tensor left, Scalar right) => left.mul_(right);

        // https://pytorch.org/docs/stable/generated/torch.multiply
        /// <summary>
        /// Multiplies each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor multiply(Tensor left, Tensor right) => left.mul(right);

        // https://pytorch.org/docs/stable/generated/torch.multiply
        /// <summary>
        /// Divides each element of the input by a scalar value.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor multiply(Tensor left, Scalar right) => left.mul(right);

        // https://pytorch.org/docs/stable/generated/torch.multiply
        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor multiply_(Tensor left, Tensor right) => left.mul_(right);

        // https://pytorch.org/docs/stable/generated/torch.multiply
        /// <summary>
        /// Divides each element of the input by the corresponding element of other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor multiply_(Tensor left, Scalar right) => left.mul_(right);

        // https://pytorch.org/docs/stable/generated/torch.mvlgamma
        /// <summary>
        /// Computes the multivariate log-gamma function) with dimension pp element-wise
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="p">The number of dimensions</param>
        /// <returns></returns>
        public static Tensor mvlgamma(Tensor input, long p) => input.mvlgamma(p);

        // https://pytorch.org/docs/stable/generated/torch.mvlgamma
        /// <summary>
        /// Computes the multivariate log-gamma function) with dimension pp element-wise, in place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="p">The number of dimensions</param>
        /// <returns></returns>
        public static Tensor mvlgamma_(Tensor input, long p) => input.mvlgamma_(p);

        // https://pytorch.org/docs/stable/generated/torch.nan_to_num
        /// <summary>
        /// Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan,
        /// posinf, and neginf, respectively. By default, NaNs are replaced with zero,
        /// positive infinity is replaced with the greatest finite value representable by input’s dtype,
        /// and negative infinity is replaced with the least finite value representable by input’s dtype.
        /// </summary>
        /// <param name="input">the input tensor</param>
        /// <param name="nan">the value to replace NaNs with. Default is zero.</param>
        /// <param name="posinf">
        /// if a Number, the value to replace positive infinity values with.
        /// If None, positive infinity values are replaced with the greatest finite value representable by input’s dtype.
        /// Default is null.
        /// </param>
        /// <param name="neginf">
        /// if a Number, the value to replace negative infinity values with.
        /// If None, negative infinity values are replaced with the lowest finite value representable by input’s dtype.
        /// Default is null.
        /// </param>
        /// <returns></returns>
        public static Tensor nan_to_num(Tensor input, double nan = 0d, double? posinf = null, double? neginf = null)
            => input.nan_to_num(nan, posinf, neginf);

        // https://pytorch.org/docs/stable/generated/torch.nan_to_num
        /// <summary>
        /// Replaces NaN, positive infinity, and negative infinity values in input, in-place.
        /// </summary>
        public static Tensor nan_to_num_(Tensor input, double nan = 0d, double? posinf = null, double? neginf = null)
            => input.nan_to_num_(nan, posinf, neginf);

        // https://pytorch.org/docs/stable/generated/torch.neg
        /// <summary>
        /// Negation
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor neg(Tensor input) => input.neg();

        // https://pytorch.org/docs/stable/generated/torch.neg
        /// <summary>
        /// In-place negation
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor neg_(Tensor input) => input.neg_();

        // https://pytorch.org/docs/stable/generated/torch.negative
        /// <summary>
        /// Negation
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor negative(Tensor input) => input.neg();

        // https://pytorch.org/docs/stable/generated/torch.negative
        /// <summary>
        /// In-place negation
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor negative_(Tensor input) => input.neg_();


        // https://pytorch.org/docs/stable/generated/torch.nextafter
        public static Tensor nextafter(Tensor input, Tensor other) => input.nextafter(other);

        // https://pytorch.org/docs/stable/generated/torch.nextafter
        /// <summary>
        /// Return the next floating-point value after input towards other, elementwise, in-place.
        /// </summary>
        public static Tensor nextafter_(Tensor input, Tensor other) => input.nextafter_(other);

        // https://pytorch.org/docs/stable/generated/torch.polygamma
        /// <summary>
        /// Computes the Nth derivative of the digamma function on input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="p">The number of dimensions</param>
        /// <returns></returns>
        public static Tensor polygamma(Tensor input, long p) => input.polygamma(p);

        // https://pytorch.org/docs/stable/generated/torch.polygamma
        /// <summary>
        /// Computes the Nth derivative of the digamma function on input, in-place.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="p">The number of dimensions</param>
        public static Tensor polygamma_(Tensor input, long p) => input.polygamma_(p);

        // https://pytorch.org/docs/stable/generated/torch.positive
        public static Tensor positive(Tensor input) => input.positive();

        // https://pytorch.org/docs/stable/generated/torch.pow
        /// <summary>
        /// Takes the power of each element in input with exponent and returns a tensor with the result.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="exponent">The right-hand operand.</param>
        [Pure]public static Tensor pow(Tensor left, Tensor exponent) => left.pow(exponent);

        // https://pytorch.org/docs/stable/generated/torch.pow
        /// <summary>
        /// Takes the power of each element in input with exponent and returns a tensor with the result.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="exponent">The right-hand operand.</param>
        [Pure]public static Tensor pow(Tensor left, Scalar exponent) => left.pow(exponent);

        // https://pytorch.org/docs/stable/generated/torch.pow
        /// <summary>
        /// Replaces each element in input with the power of the element and the exponent.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="exponent">The right-hand operand.</param>
        public static Tensor pow_(Tensor left, Tensor exponent) => left.pow_(exponent);

        // https://pytorch.org/docs/stable/generated/torch.pow
        /// <summary>
        /// Replaces each element in input with the power of the element and the exponent.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="exponent">The right-hand operand.</param>
        public static Tensor pow_(Tensor left, Scalar exponent) => left.pow_(exponent);

        // https://pytorch.org/docs/stable/generated/torch.quantized_batch_norm
        /// <summary>
        /// Applies batch normalization on a 4D (NCHW) quantized tensor.
        /// </summary>
        /// <param name="input">quantized tensor</param>
        /// <param name="weight">float tensor that corresponds to the gamma, size C</param>
        /// <param name="bias">float tensor that corresponds to the beta, size C</param>
        /// <param name="mean">float mean value in batch normalization, size C</param>
        /// <param name="var">float tensor for variance, size C</param>
        /// <param name="eps">a value added to the denominator for numerical stability.</param>
        /// <param name="output_scale">output quantized tensor scale</param>
        /// <param name="output_zero_point">output quantized tensor zero_point</param>
        /// <returns>A quantized tensor with batch normalization applied.</returns>
        [Pure, Obsolete("not implemented", true)]
        public static Tensor quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor @var, double eps, double output_scale, long output_zero_point)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.quantized_max_pool1d
        /// <summary>
        /// Applies a 1D max pooling over an input quantized tensor composed of several input planes.
        /// </summary>
        /// <param name="input">quantized tensor</param>
        /// <param name="kernel_size">the size of the sliding window</param>
        /// <param name="stride">the stride of the sliding window</param>
        /// <param name="padding">padding to be added on both sides, must be &gt;= 0 and &lt;= kernel_size / 2</param>
        /// <param name="dilation">the stride between elements within a sliding window, must be &gt; 0. Default 1</param>
        /// <param name="ceil_mode">If <value>true</value>, will use ceil instead of floor to compute the output shape. Defaults to <value>false</value>.</param>
        /// <returns>A quantized tensor with max_pool1d applied.</returns>
        [Pure, Obsolete("not implemented", true)]
        public static Tensor quantized_max_pool1d(Tensor input, long[] kernel_size, long[]? stride, long[]? padding, long dilation=1, bool ceil_mode=false)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.quantized_max_pool2d
        /// <summary>
        /// Applies a 2D max pooling over an input quantized tensor composed of several input planes.
        /// </summary>
        /// <param name="input">quantized tensor</param>
        /// <param name="kernel_size">the size of the sliding window</param>
        /// <param name="stride">the stride of the sliding window</param>
        /// <param name="padding">padding to be added on both sides, must be &gt;= 0 and &lt;= kernel_size / 2</param>
        /// <param name="dilation">The stride between elements within a sliding window, must be > 0. Default 1</param>
        /// <param name="ceil_mode">If <value>true</value>, will use ceil instead of floor to compute the output shape. Defaults to <value>false</value>.</param>
        /// <returns>A quantized tensor with max_pool2d applied.</returns>
        [Pure, Obsolete("not implemented", true)]
        public static Tensor quantized_max_pool2d(Tensor input, long[] kernel_size, long[]? stride, long[]? padding, long dilation=1, bool ceil_mode=false)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.rad2deg
        /// <summary>
        /// Returns a new tensor with each of the elements of input converted from angles in radians to degrees.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>tensor with angles in radians</returns>
        public static Tensor rad2deg(Tensor input) => input.rad2deg();

        // https://pytorch.org/docs/stable/generated/torch.rad2deg
        /// <summary>
        /// Convert each element from radians to degrees, in-place.
        /// </summary>
        public static Tensor rad2deg_(Tensor input) => input.rad2deg_();

        // https://pytorch.org/docs/stable/generated/torch.real
        /// <summary>
        /// Returns a new tensor containing real values of the self tensor.
        /// The returned tensor and self share the same underlying storage.
        /// </summary>
        /// <param name="input">the input tensor.</param>
        /// <returns>tensor containing real values</returns>
        [Pure]public static Tensor real(Tensor input) => input.real;

        // https://pytorch.org/docs/stable/generated/torch.reciprocal
        /// <summary>
        /// Returns a new tensor with the reciprocal of the elements of input
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor reciprocal(Tensor input) => input.reciprocal();

        // https://pytorch.org/docs/stable/generated/torch.reciprocal
        /// <summary>
        /// Replaces each element with the reciprocal of the input
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor reciprocal_(Tensor input) => input.reciprocal_();

        // https://pytorch.org/docs/stable/generated/torch.remainder
        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        [Pure]public static Tensor remainder(Tensor left, Tensor right) => left.remainder(right);

        // https://pytorch.org/docs/stable/generated/torch.remainder
        /// <summary>
        /// Computes the element-wise remainder of division.
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        [Pure]public static Tensor remainder(Tensor left, Scalar right) => left.remainder(right);

        // https://pytorch.org/docs/stable/generated/torch.remainder
        /// <summary>
        /// Computes the element-wise remainder of division, in place
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor remainder_(Tensor left, Tensor right) => left.remainder_(right);

        // https://pytorch.org/docs/stable/generated/torch.remainder
        /// <summary>
        /// Computes the element-wise remainder of division, in place
        /// </summary>
        /// <param name="left">Numerator</param>
        /// <param name="right">Denominator</param>
        public static Tensor remainder_(Tensor left, Scalar right) => left.remainder_(right);

        // https://pytorch.org/docs/stable/generated/torch.round
        /// <summary>
        /// Returns a new tensor with each of the elements of input rounded to the closest value with the given number of decimals.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="decimals">Number of decimal places to round to (default: 0). If decimals is negative, it specifies the number of positions to the left of the decimal point.</param>
        [Pure]public static Tensor round(Tensor input, long decimals = 0L) => input.round(decimals);

        // https://pytorch.org/docs/stable/generated/torch.round
        /// <summary>
        /// Replaces each of the elements of input with the element rounded to the closest  value with the given number of decimals.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="decimals">Number of decimal places to round to (default: 0). If decimals is negative, it specifies the number of positions to the left of the decimal point.</param>
        public static Tensor round_(Tensor input, long decimals = 0L) => input.round_(decimals);

        // https://pytorch.org/docs/stable/generated/torch.rsqrt
        /// <summary>
        /// Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor rsqrt(Tensor input) => input.rsqrt();

        // https://pytorch.org/docs/stable/generated/torch.rsqrt
        /// <summary>
        /// Replaces each of the elements of input with  the reciprocal of the square-root of each of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor rsqrt_(Tensor input) => input.rsqrt_();

        // https://pytorch.org/docs/stable/generated/torch.sigmoid
        /// <summary>
        /// Computes the logistic sigmoid function of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor sigmoid(Tensor input) => input.sigmoid();

        // https://pytorch.org/docs/stable/generated/torch.sigmoid
        /// <summary>
        /// Computes the logistic sigmoid function of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor sigmoid_(Tensor input) => input.sigmoid_();

        // https://pytorch.org/docs/stable/generated/torch.sign
        /// <summary>
        /// Returns a new tensor with the signs (-1, 0, 1) of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor sign(Tensor input) => input.sign();

        // https://pytorch.org/docs/stable/generated/torch.sign
        /// <summary>
        /// Returns a new tensor with the signs (-1, 0, 1) of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure] public static Tensor sign_(Tensor input) => input.sign_();

        // https://pytorch.org/docs/stable/generated/torch.sgn
        /// <summary>
        /// This function is an extension of torch.sign() to complex tensors.
        /// It computes a new tensor whose elements have the same angles as the corresponding
        /// elements of input and absolute values (i.e. magnitudes) of one for complex tensors
        /// and is equivalent to torch.sign() for non-complex tensors.
        /// </summary>
        /// <param name="input">the input tensor.</param>
        /// <returns>the output tensor.</returns>
        [Pure]
        public static Tensor sgn(Tensor input) => input.sgn();

        // https://pytorch.org/docs/stable/generated/torch.sgn
        /// <summary>
        /// This function is an extension of torch.sign() to complex tensors.
        /// It computes a new tensor whose elements have the same angles as the corresponding
        /// elements of input and absolute values (i.e. magnitudes) of one for complex tensors
        /// and is equivalent to torch.sign() for non-complex tensors.
        /// </summary>
        /// <param name="input">the input tensor.</param>
        /// <returns>the output tensor.</returns>
        [Pure]
        public static Tensor sgn_(Tensor input) => input.sgn_();

        // https://pytorch.org/docs/stable/generated/torch.signbit
        /// <summary>
        /// Tests whether each element of input has its sign bit set (is less than zero) or not.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>A boolean tensor of the same shape as the input.</returns>
        [Pure]public static Tensor signbit(Tensor input) => input.signbit();

        // https://pytorch.org/docs/stable/generated/torch.sin
        /// <summary>
        /// Computes the sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor sin(Tensor input) => input.sin();

        // https://pytorch.org/docs/stable/generated/torch.sin
        /// <summary>
        /// Computes the sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor sin_(Tensor input) => input.sin_();

        // https://pytorch.org/docs/stable/generated/torch.sinc
        /// <summary>
        /// Computes the normalized sinc of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor sinc(Tensor input) => input.sinc();

        // https://pytorch.org/docs/stable/generated/torch.sinc
        /// <summary>
        /// Computes the normalized sinc of input, in place.
        /// </summary>
        /// <returns></returns>
        public static Tensor sinc_(Tensor input) => input.sinc_();

        // https://pytorch.org/docs/stable/generated/torch.sinh
        /// <summary>
        /// Computes the hyperbolic sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor sinh(Tensor input) => input.sinh();

        // https://pytorch.org/docs/stable/generated/torch.sinh
        /// <summary>
        /// Computes the hyperbolic sine of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor sinh_(Tensor input) => input.sinh_();

        // https://pytorch.org/docs/stable/generated/torch.sqrt
        /// <summary>
        /// Computes the element-wise square root
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor sqrt(Tensor input) => input.sqrt();

        // https://pytorch.org/docs/stable/generated/torch.sqrt
        /// <summary>
        /// Computes the element-wise square root, in place
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor sqrt_(Tensor input) => input.sqrt_();

        // https://pytorch.org/docs/stable/generated/torch.square
        /// <summary>
        /// Computes the element-wise square
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor square(Tensor input) => input.square();

        // https://pytorch.org/docs/stable/generated/torch.sub
        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]public static Tensor sub(Tensor left, Tensor right) => left.sub(right);

        // https://pytorch.org/docs/stable/generated/torch.sub
        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]public static Tensor sub(Tensor left, Scalar right) => left.sub(right);

        // https://pytorch.org/docs/stable/generated/torch.sub
        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor sub_(Tensor left, Tensor right) => left.sub_(right);

        // https://pytorch.org/docs/stable/generated/torch.sub
        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor sub_(Tensor left, Scalar right) => left.sub_(right);

        // https://pytorch.org/docs/stable/generated/torch.subtract
        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]public static Tensor subtract(Tensor left, Tensor right) => left.subtract(right);

        // https://pytorch.org/docs/stable/generated/torch.subtract
        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]public static Tensor subtract(Tensor left, Scalar right) => left.subtract(right);

        // https://pytorch.org/docs/stable/generated/torch.subtract
        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <returns></returns>
        public static Tensor subtract_(Tensor left, Tensor right) => left.subtract_(right);

        // https://pytorch.org/docs/stable/generated/torch.subtract
        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor subtract_(Tensor left, Scalar right) => left.subtract_(right);

        // https://pytorch.org/docs/stable/generated/torch.tan
        /// <summary>
        /// Computes the tangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]public static Tensor tan(Tensor input) => input.tan();

        // https://pytorch.org/docs/stable/generated/torch.tan
        /// <summary>
        /// Computes the tangent of the elements of input. In-place version.
        /// </summary>
        /// <returns></returns>
        public static Tensor tan_(Tensor input) => input.tan_();

        // https://pytorch.org/docs/stable/generated/torch.tanh
        /// <summary>
        /// Computes the hyperbolic tangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        [Pure]
        public static Tensor tanh(Tensor input) => input.tanh();

        // https://pytorch.org/docs/stable/generated/torch.tanh
        /// <summary>
        /// Computes the hyperbolic tangent of the elements of input.
        /// </summary>
        /// <returns></returns>
        public static Tensor tanh_(Tensor input) => input.tanh_();

        // https://pytorch.org/docs/stable/generated/torch.true_divide
        /// <summary>
        /// Alias for torch.div() with rounding_mode=None.
        /// </summary>
        [Pure]
        public static Tensor true_divide(Tensor dividend, Tensor divisor) => dividend.true_divide(divisor);

        // https://pytorch.org/docs/stable/generated/torch.true_divide
        /// <summary>
        /// Alias for torch.div_() with rounding_mode=None.
        /// </summary>
        public static Tensor true_divide_(Tensor dividend, Tensor divisor) => dividend.true_divide_(divisor);

        // https://pytorch.org/docs/stable/generated/torch.trunc
        /// <summary>
        /// Returns a new tensor with the truncated integer values of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]
        public static Tensor trunc(Tensor input) => input.trunc();

        // https://pytorch.org/docs/stable/generated/torch.trunc
        /// <summary>
        /// Replaces each element with the truncated integer values of the elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor trunc_(Tensor input) => input.trunc_();

        // https://pytorch.org/docs/stable/generated/torch.xlogy
        /// <summary>
        /// Computes x * log(y)
        /// </summary>
        /// <param name="x">The 'x' operand.</param>
        /// <param name="y">The 'y' operand.</param>
        [Pure]
        public static Tensor xlogy(Tensor x, Tensor y) => x.xlogy(y);

        // https://pytorch.org/docs/stable/generated/torch.xlogy
        /// <summary>
        /// Computes x * log(y) in place
        /// </summary>
        /// <param name="x">The 'x' operand.</param>
        /// <param name="y">The 'y' operand.</param>
        public static Tensor xlogy_(Tensor x, Tensor y) => x.xlogy_(y);

        // https://pytorch.org/docs/stable/generated/torch.xlogy
        /// <summary>
        /// Computes x * log(y)
        /// </summary>
        /// <param name="x">The 'x' operand.</param>
        /// <param name="y">The 'y' operand.</param>
        [Pure]
        public static Tensor xlogy(Tensor x, Scalar y) => x.xlogy(y);

        // https://pytorch.org/docs/stable/generated/torch.xlogy
        /// <summary>
        /// Computes x * log(y) in place
        /// </summary>
        /// <param name="x">The 'x' operand.</param>
        /// <param name="y">The 'y' operand.</param>
        public static Tensor xlogy_(Tensor x, Scalar y) => x.xlogy_(y);
    }
}