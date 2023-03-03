// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics.Contracts;

namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#comparison-ops
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.allclose
        /// <summary>
        /// This function checks if all input and other lie within a certain distance from each other
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="target"></param>
        /// <param name="rtol">Relative tolerance</param>
        /// <param name="atol">Absolute tolerance</param>
        /// <param name="equal_nan">If true, then two NaN s will be considered equal</param>
        [Pure]
        public static bool allclose(Tensor tensor, Tensor target, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false)
            => tensor.allclose(target, rtol, atol, equal_nan);

        // https://pytorch.org/docs/stable/generated/torch.argsort
        /// <summary>
        /// Returns the indices that sort a tensor along a given dimension in ascending order by value.
        /// </summary>
        [Pure]
        public static Tensor argsort(Tensor input, long dim = -1, bool descending = false)
            => input.argsort(dim, descending);

        // https://pytorch.org/docs/stable/generated/torch.eq
        /// <summary>
        /// Element-wise equal comparison
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]
        public static Tensor eq(Tensor left, Tensor right)
            => left.eq(right);

        // https://pytorch.org/docs/stable/generated/torch.equal
        [Pure]
        public static Tensor equal(Tensor tensor, Tensor target)
            => tensor.equal(target);

        // https://pytorch.org/docs/stable/generated/torch.ge
        /// <summary>
        /// Element-wise greater-than-or-equal comparison
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]
        public static Tensor ge(Tensor left, Tensor right)
            => left.ge(right);

        // https://pytorch.org/docs/stable/generated/torch.greater_equal
        [Pure]
        public static Tensor greater_equal(Tensor tensor, Tensor target)
            => tensor.greater_equal(target);

        // https://pytorch.org/docs/stable/generated/torch.gt
        /// <summary>
        /// Element-wise greater-than comparison
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]
        public static Tensor gt(Tensor left, Tensor right)
            => left.gt(right);

        // https://pytorch.org/docs/stable/generated/torch.greater
        [Pure]
        public static Tensor greater(Tensor tensor, Tensor target)
            => tensor.greater(target);

        // https://pytorch.org/docs/stable/generated/torch.isclose
        [Pure]
        public static Tensor isclose(Tensor tensor, Tensor other, double rtol = 1e-05, double atol = 1e-08, bool nanEqual = false)
            => tensor.isclose(other, rtol, atol, nanEqual);

        // https://pytorch.org/docs/stable/generated/torch.isfinite
        [Pure]
        public static Tensor isfinite(Tensor tensor)
            => tensor.isfinite();

        // https://pytorch.org/docs/stable/generated/torch.isin
        [Pure]
        public static Tensor isin(Tensor tensor, Tensor test_elements, bool assumeUnique = false, bool invert = false)
            => tensor.isin(test_elements, assumeUnique, invert);

        // https://pytorch.org/docs/stable/generated/torch.isinf
        [Pure]
        public static Tensor isinf(Tensor tensor)
            => tensor.isinf();

        // https://pytorch.org/docs/stable/generated/torch.isposinf
        [Pure]
        public static Tensor isposinf(Tensor tensor)
            => tensor.isposinf();

        // https://pytorch.org/docs/stable/generated/torch.isneginf
        [Pure]
        public static Tensor isneginf(Tensor tensor)
            => tensor.isneginf();

        // https://pytorch.org/docs/stable/generated/torch.isnan
        /// <summary>
        /// Returns a new tensor with boolean elements representing if each element of input is <value>NaN</value> or not.
        /// Complex values are considered <value>NaN</value> when either their real and/or imaginary part is <value>NaN</value>.
        /// </summary>
        /// <param name="input">the input tensor</param>
        /// <returns>A boolean tensor that is <value>True</value> where input is <value>NaN</value> and <value>False</value> elsewhere</returns>
        [Pure]
        public static Tensor isnan(Tensor input)
            => input.isnan();

        // https://pytorch.org/docs/stable/generated/torch.isreal
        [Pure]
        public static Tensor isreal(Tensor tensor)
            => tensor.isreal();

        // https://pytorch.org/docs/stable/generated/torch.kthvalue
        /// <summary>
        /// Returns a named tuple (values, indices) where values is the k th smallest element of each row of the input tensor in the given dimension dim. And indices is the index location of each element found.
        /// If dim is not given, the last dimension of the input is chosen.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="k">k for the k-th smallest element</param>
        /// <param name="dim">The dimension to find the kth value along</param>
        /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
        [Pure]
        public static (Tensor values, Tensor indices) kthvalue(Tensor input, long k, long? dim, bool keepdim = false)
            => input.kthvalue(k, dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.le
        /// <summary>
        /// Element-wise less-than-or-equal comparison
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]
        public static Tensor le(Tensor left, Tensor right)
            => left.le(right);

        // https://pytorch.org/docs/stable/generated/torch.less_equal
        [Pure]
        public static Tensor less_equal(Tensor tensor, Tensor target)
            => tensor.less_equal(target);

        // https://pytorch.org/docs/stable/generated/torch.lt
        /// <summary>
        /// Element-wise less-than comparison
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]
        public static Tensor lt(Tensor left, Tensor right)
            => left.lt(right);

        // https://pytorch.org/docs/stable/generated/torch.less
        [Pure]
        public static Tensor less(Tensor tensor, Tensor target)
            => tensor.less(target);

        // https://pytorch.org/docs/stable/generated/torch.maximum
        /// <summary>
        /// Computes the element-wise maximum of input and other.
        /// </summary>
        /// <param name="input">The first input tensor</param>
        /// <param name="other">The second input tensor</param>
        [Pure]
        public static Tensor maximum(Tensor input, Tensor other)
            => input.maximum(other);

        // https://pytorch.org/docs/stable/generated/torch.minimum
        /// <summary>
        /// Computes the element-wise minimum of input and other.
        /// </summary>
        /// <param name="input">The first input tensor</param>
        /// <param name="other">The second input tensor</param>
        [Pure]
        public static Tensor minimum(Tensor input, Tensor other)
            => input.minimum(other);

        // https://pytorch.org/docs/stable/generated/torch.fmax
        /// <summary>
        /// Computes the element-wise maximum of input and other.
        ///
        /// This is like torch.maximum() except it handles NaNs differently: if exactly one of the two elements being compared is a NaN
        /// then the non-NaN element is taken as the maximum.
        /// Only if both elements are NaN is NaN propagated.
        /// </summary>
        /// <param name="input">The first input tensor</param>
        /// <param name="other">The second input tensor</param>
        [Pure]
        public static Tensor fmax(Tensor input, Tensor other)
            => input.fmax(other);

        // https://pytorch.org/docs/stable/generated/torch.fmin
        /// <summary>
        /// Computes the element-wise minimum of input and other.
        ///
        /// This is like torch.minimum() except it handles NaNs differently: if exactly one of the two elements being compared is a NaN
        /// then the non-NaN element is taken as the minimum.
        /// Only if both elements are NaN is NaN propagated.
        /// </summary>
        /// <param name="input">The first input tensor</param>
        /// <param name="other">The second input tensor</param>
        [Pure]
        public static Tensor fmin(Tensor input, Tensor other)
            => input.fmin(other);

        // https://pytorch.org/docs/stable/generated/torch.ne
        /// <summary>
        /// Element-wise not-equal comparison
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        [Pure]
        public static Tensor ne(Tensor left, Tensor right)
            => left.ne(right);

        // https://pytorch.org/docs/stable/generated/torch.not_equal
        [Pure]
        public static Tensor not_equal(Tensor tensor, Tensor target)
            => tensor.not_equal(target);

        // https://pytorch.org/docs/stable/generated/torch.sort
        /// <summary>
        /// Sorts the elements of the input tensor along a given dimension in ascending order by value.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">The dimension to sort along. If dim is not given, the last dimension of the input is chosen.</param>
        /// <param name="descending">Controls the sorting order (ascending or descending)</param>
        /// <param name="stable">Makes the sorting routine stable, which guarantees that the order of equivalent elements is preserved.</param>
        /// <returns>A named tuple of (values, indices) is returned, where the values are the sorted values and indices are the indices of the elements in the original input tensor.</returns>
        [Pure]
        public static (Tensor values, Tensor indices) sort(Tensor input, long dim = -1, bool descending = false, bool stable = false)
            => input.sort(dim, descending, stable);

        // https://pytorch.org/docs/stable/generated/torch.searchsorted.html
        /// <summary>
        /// Find the indices from the innermost dimension of sorted_sequence such that, if the corresponding values in values were inserted before the indices,
        /// when sorted, the order of the corresponding innermost dimension within sorted_sequence would be preserved.
        /// Return a new tensor with the same size as values.
        /// If right is false, then the left boundary of sorted_sequence is closed. 
        /// </summary>
        /// <param name="sorted_sequence">N-D or 1-D tensor, containing monotonically increasing sequence on the innermost dimension unless sorter is provided, in which case the sequence does not need to be sorted</param>
        /// <param name="values">N-D tensor or a Scalar containing the search value(s).</param>
        /// <param name="out_int32">Indicates the output data type. torch.int32 if true, torch.int64 otherwise. Default value is false, i.e. default output data type is torch.int64.</param>
        /// <param name="right">Indicates the output data type. torch.int32 if true, torch.int64 otherwise. Default value is false, i.e. default output data type is torch.int64.</param>
        /// <param name="sorter">If provided, a tensor matching the shape of the unsorted sorted_sequence containing a sequence of indices that sort it in the ascending order on the innermost dimension</param>
        public static Tensor searchsorted(Tensor sorted_sequence, Tensor values, bool out_int32 = false, bool right = false, Tensor sorter = null)
        {
            var res = PInvoke.LibTorchSharp.THSTensor_searchsorted_t(sorted_sequence.Handle, values.Handle, out_int32, right, sorter is null ? IntPtr.Zero : sorter.Handle);
            if (res == IntPtr.Zero) CheckForErrors();
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.searchsorted.html
        /// <summary>
        /// Find the indices from the innermost dimension of sorted_sequence such that, if the corresponding values in values were inserted before the indices,
        /// when sorted, the order of the corresponding innermost dimension within sorted_sequence would be preserved.
        /// Return a new tensor with the same size as values.
        /// If right is false, then the left boundary of sorted_sequence is closed. 
        /// </summary>
        /// <param name="sorted_sequence">N-D or 1-D tensor, containing monotonically increasing sequence on the innermost dimension unless sorter is provided, in which case the sequence does not need to be sorted</param>
        /// <param name="values">A Scalar containing the search value.</param>
        /// <param name="out_int32">Indicates the output data type. torch.int32 if true, torch.int64 otherwise. Default value is false, i.e. default output data type is torch.int64.</param>
        /// <param name="right">Indicates the output data type. torch.int32 if true, torch.int64 otherwise. Default value is false, i.e. default output data type is torch.int64.</param>
        /// <param name="sorter">If provided, a tensor matching the shape of the unsorted sorted_sequence containing a sequence of indices that sort it in the ascending order on the innermost dimension</param>
        public static Tensor searchsorted(Tensor sorted_sequence, Scalar values, bool out_int32, bool right, Tensor sorter)
        {
            var res = PInvoke.LibTorchSharp.THSTensor_searchsorted_s(sorted_sequence.Handle, values.Handle, out_int32, right, sorter is null ? IntPtr.Zero : sorter.Handle);
            if (res == IntPtr.Zero) CheckForErrors();
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.topk
        /// <summary>
        /// Returns the k largest elements of the given input tensor along a given dimension.
        /// </summary>
        /// <param name="tensor">The input tensor</param>
        /// <param name="k">The 'k' in 'top-k'.</param>
        /// <param name="dim">The dimension to sort along. If dim is not given, the last dimension of the input is chosen.</param>
        /// <param name="largest">Controls whether to return largest or smallest elements</param>
        /// <param name="sorted">Controls whether to return the elements in sorted order</param>
        [Pure]
        public static (Tensor values, Tensor indices) topk(Tensor tensor, int k, int dim = -1, bool largest = true, bool sorted = true)
            => tensor.topk(k, dim, largest, sorted);

        // https://pytorch.org/docs/stable/generated/torch.msort
        [Pure]
        public static Tensor msort(Tensor tensor)
            => tensor.msort();
    }
}