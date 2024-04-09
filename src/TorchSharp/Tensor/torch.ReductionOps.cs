// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;

using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#reduction-ops
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.argmax
        /// <summary>
        /// Returns the indices of the maximum value of all elements in the input tensor.
        /// </summary>
        [Pure]public static Tensor argmax(Tensor input)
            => input.argmax();

        // https://pytorch.org/docs/stable/generated/torch.argmax
        /// <summary>
        /// Returns the indices of the maximum value of all elements in the input tensor.
        /// </summary>
        [Pure]public static Tensor argmax(Tensor input, long dim, bool keepdim = false)
            => input.argmax(dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.argmin
        /// <summary>
        /// Returns the indices of the minimum value of all elements in the input tensor.
        /// </summary>
        [Pure]public static Tensor argmin(Tensor input)
            => input.argmin();

        // https://pytorch.org/docs/stable/generated/torch.argmin
        /// <summary>
        /// Returns the indices of the minimum value of all elements in the input tensor.
        /// </summary>
        [Pure]public static Tensor argmin(Tensor input, long dim, bool keepdim = false)
            => input.argmin(dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.amax
        /// <summary>
        /// Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dims">The dimension or dimensions to reduce.</param>
        /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
        /// <param name="out">The output tensor -- optional.</param>
        [Pure]public static Tensor amax(Tensor input, long[] dims, bool keepdim = false, Tensor? @out = null)
            => input.amax(dims, keepdim, @out);

        [Pure]public static Tensor amax(Tensor input, ReadOnlySpan<long> dims, bool keepdim = false, Tensor? @out = null)
            => input.amax(dims, keepdim, @out);

        [Pure]public static Tensor amax(Tensor input, long[] dims)
            => input.amax(dims);

        // https://pytorch.org/docs/stable/generated/torch.amin
        /// <summary>
        /// Returns the minimum value of each slice of the input tensor in the given dimension(s) dim.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dims">The dimension or dimensions to reduce.</param>
        /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
        /// <param name="out">The output tensor -- optional.</param>
        [Pure]public static Tensor amin(Tensor input, long[] dims, bool keepdim = false, Tensor? @out = null)
            => input.amin(dims, keepdim, @out);

        // https://pytorch.org/docs/stable/generated/torch.amin
        /// <summary>
        /// Returns the minimum value of each slice of the input tensor in the given dimension(s) dim.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dims">The dimension or dimensions to reduce.</param>
        /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
        /// <param name="out">The output tensor -- optional.</param>
        [Pure]public static Tensor amin(Tensor input, ReadOnlySpan<long> dims, bool keepdim = false, Tensor? @out = null)
            => input.amin(dims, keepdim, @out);

        // https://pytorch.org/docs/stable/generated/torch.aminmax
        [Pure]public static (Tensor min, Tensor max) aminmax(Tensor input, long? dim = null, bool keepdim = false)
            => input.aminmax(dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.all
        /// <summary>
        /// Tests if all elements in input evaluate to true.
        /// <param name="input">The input tensor</param>
        /// </summary>
        [Pure]public static Tensor all(Tensor input) => input.all();

        // https://pytorch.org/docs/stable/generated/torch.all
        /// <summary>
        /// Tests if all elements in input evaluate to true.
        /// <param name="input">The input tensor</param>
        /// <param name="dim">The dimension to reduce</param>
        /// <param name="keepdim">Keep the dimension to reduce</param>
        /// </summary>
        [Pure]public static Tensor all(Tensor input, long dim, bool keepdim = false) => input.all(dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.any
        /// <summary>
        /// Tests if all elements in input evaluate to true.
        /// <param name="input">The input tensor</param>
        /// </summary>
        [Pure]public static Tensor any(Tensor input) => input.any();

        // https://pytorch.org/docs/stable/generated/torch.any
        /// <summary>
        /// Tests if any element in input evaluate to true.
        /// <param name="input">The input tensor</param>
        /// <param name="dim">The dimension to reduce</param>
        /// <param name="keepdim">Keep the dimension to reduce</param>
        /// </summary>
        [Pure]public static Tensor any(Tensor input, long dim, bool keepdim = false) => input.any(dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.max
        /// <summary>
        /// Returns the maximum value of all elements in the input tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor max(Tensor input) => input.max();

        /// <summary>
        /// Computes the element-wise maximum of input and other.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="other">The second tensor.</param>
        [Pure] public static Tensor max(Tensor input, Tensor other) => torch.maximum(input, other);

        // https://pytorch.org/docs/stable/generated/torch.max
        /// <summary>
        /// Returns a named tuple (values, indexes) where values is the maximum value of each row of the input tensor in the given dimension dim.
        /// And indices is the index location of each maximum value found (argmax).
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">the dimension to reduce.</param>
        /// <param name="keepdim">whether the output tensor has dim retained or not. Default: false.</param>
        /// <remarks>If keepdim is true, the output tensors are of the same size as input except in the dimension dim where they are of size 1.
        /// Otherwise, dim is squeezed(see torch.squeeze()), resulting in the output tensors having 1 fewer dimension than input.</remarks>
        [Pure]public static (Tensor values, Tensor indexes) max(Tensor input, long dim, bool keepdim = false) => input.max(dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.min
        /// <summary>
        /// Returns the minimum value of all elements in the input tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        [Pure]public static Tensor min(Tensor input) => input.min();

        /// <summary>
        /// Computes the element-wise minimum of input and other.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="other">The second tensor.</param>
        [Pure] public static Tensor min(Tensor input, Tensor other) => torch.minimum(input, other);

        // https://pytorch.org/docs/stable/generated/torch.min
        /// <summary>
        /// Returns a named tuple (values, indexes) where values is the minimum value of each row of the input tensor in the given dimension dim.
        /// And indices is the index location of each minimum value found (argmin).
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">the dimension to reduce.</param>
        /// <param name="keepdim">whether the output tensor has dim retained or not. Default: false.</param>
        /// <remarks>If keepdim is true, the output tensors are of the same size as input except in the dimension dim where they are of size 1.
        /// Otherwise, dim is squeezed(see torch.squeeze()), resulting in the output tensors having 1 fewer dimension than input.</remarks>
        [Pure]public static (Tensor values, Tensor indexes) min(Tensor input, long dim, bool keepdim = false) => input.min(dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.dist
        /// <summary>
        /// Returns the p-norm of (input - other).
        /// The shapes of input and other must be broadcastable.
        /// </summary>
        /// <param name="input">Left-hand side input tensor.</param>
        /// <param name="other">Right-hand side input tensor</param>
        /// <param name="p">The norm to be computed.</param>
        [Pure]public static Tensor dist(Tensor input, Tensor other, float p = 2.0f) => input.dist(other, p);

        // https://pytorch.org/docs/stable/generated/torch.logsumexp
        /// <summary>
        /// Returns the log of summed exponentials of each row of the input tensor in the given dimension dim.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to do the operation over</param>
        /// <param name="keepdim">Thether the output tensor has dim retained or not.</param>
        [Pure]public static Tensor logsumexp(Tensor input, long dim, bool keepdim = false)
            => input.logsumexp(dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.mean
        /// <summary>
        /// Returns the mean value of all elements in the input tensor.
        /// </summary>
        [Pure]public static Tensor mean(Tensor input) => input.mean();

        // https://pytorch.org/docs/stable/generated/torch.
        /// <summary>
        /// Returns the mean value of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dimensions">The dimension or dimensions to reduce.</param>
        /// <param name="keepdim">Whether the output tensor has dim retained or not.</param>
        /// <param name="type">The desired data type of returned tensor. If specified, the input tensor is cast to dtype before the operation is performed. This is useful for preventing data type overflows.</param>
        /// <remarks>
        /// If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1.
        /// Otherwise, dim is squeezed(see torch.squeeze()), resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
        /// </remarks>
        [Pure]public static Tensor mean(Tensor input, long[] dimensions, bool keepdim = false, ScalarType? type = null) => input.mean(dimensions, keepdim, type);

        // https://pytorch.org/docs/stable/generated/torch.nanmean
        [Pure]public static Tensor nanmean(Tensor input, int? dim = null, bool keepdim = false, ScalarType? dtype = null)
            => input.nanmean(dim, keepdim, dtype);

        // https://pytorch.org/docs/stable/generated/torch.median
        [Pure]public static Tensor median(Tensor input) => input.median();

        // https://pytorch.org/docs/stable/generated/torch.nanmedian
        [Pure]public static Tensor nanmedian(Tensor input)
            => input.nanmedian();

        // https://pytorch.org/docs/stable/generated/torch.mode
        [Pure]public static (Tensor values, Tensor indices) mode(Tensor input, long dim = -1L, bool keepdim = false)
            => input.mode(dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.norm
        [Pure]public static Tensor norm(Tensor input, float p = 2.0f)
            => input.norm(p);

        [Pure]public static Tensor norm(Tensor input, int dimension, bool keepdim = false, float p = 2.0f)
            => input.norm(dimension, keepdim, p);

        // https://pytorch.org/docs/stable/generated/torch.nansum
        [Pure]public static Tensor nansum(Tensor input) => input.nansum();

        // https://pytorch.org/docs/stable/generated/torch.prod
        // TODO: torch.prod
        // static Tensor prod(Tensor input, ScalarType dtype = null) => input.prod(dtype);

        // https://pytorch.org/docs/stable/generated/torch.quantile
        [Pure]public static Tensor quantile(Tensor input, Tensor q, long dim = -1, bool keepdim = false)
            => input.quantile(q, dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.nanquantile
        [Pure]public static Tensor nanquantile(Tensor input, Tensor q, long dim = -1, bool keepdim = false)
            => input.nanquantile(q, dim, keepdim);

        // https://pytorch.org/docs/stable/generated/torch.prod
        /// <summary>
        ///  Returns the product of each row of the input tensor in the given dimensions.
        /// </summary>
        [Pure] public static Tensor prod(Tensor input, ScalarType? type = null) => input.prod(type);

        // https://pytorch.org/docs/stable/generated/torch.prod
        /// <summary>
        /// Returns the product of each row of the input tensor in the given dimension.
        /// </summary>
        [Pure] public static Tensor prod(Tensor input, long dim, bool keepdim = false, ScalarType? type = null) => input.prod(dim, keepdim, type);

        // https://pytorch.org/docs/stable/generated/torch.std
        /// <summary>
        /// Calculates the standard deviation of all elements in the tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        [Pure]public static Tensor std(Tensor input, bool unbiased = true) => input.std(unbiased);

        /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        [Pure]public static Tensor std(Tensor input, long[] dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.std(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        [Pure]public static Tensor std(Tensor input, long dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.std(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        [Pure]public static Tensor std(Tensor input, (long, long) dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.std(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the standard deviation of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        [Pure]public static Tensor std(Tensor input, (long, long, long) dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.std(dimensions, unbiased, keepdim, type);

        // https://pytorch.org/docs/stable/generated/torch.std_mean
        /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
        [Pure]public static (Tensor std, Tensor mean) std_mean(Tensor input, long[] dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.std_mean(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
        [Pure]public static (Tensor std, Tensor mean) std_mean(Tensor input, ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.std_mean(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
        [Pure]public static (Tensor std, Tensor mean) std_mean(Tensor input, long dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.std_mean(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
        [Pure]public static (Tensor std, Tensor mean) std_mean(Tensor input, (long, long) dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.std_mean(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the standard deviation and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample deviation is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>A <see cref="Tensor">tensor</see> tuple of the standard deviation and the mean.</returns>
        [Pure]public static (Tensor std, Tensor mean) std_mean(Tensor input, (long, long, long) dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.std_mean(dimensions, unbiased, keepdim, type);

        // https://pytorch.org/docs/stable/generated/torch.sum
        /// <summary>
        ///  Returns the sum of each row of the input tensor in the given dimensions.
        /// </summary>
        [Pure]public static Tensor sum(Tensor input, ScalarType? type = null) => input.sum(type);

        // https://pytorch.org/docs/stable/generated/torch.sum
        /// <summary>
        /// Returns the sum of each row of the input tensor in the given dimensions.
        /// </summary>
        [Pure]public static Tensor sum(Tensor input, ReadOnlySpan<long> dim, bool keepdim = false, ScalarType? type = null) => input.sum(dim, keepdim, type);

        // https://pytorch.org/docs/stable/generated/torch.sum
        /// <summary>
        /// Returns the sum of each row of the input tensor in the given dimension.
        /// </summary>
        [Pure]public static Tensor sum(Tensor input, long dim, bool keepdim = false, ScalarType? type = null) => input.sum(dim, keepdim, type);

        // https://pytorch.org/docs/stable/generated/torch.unique
        /// <summary>
        /// Returns the unique elements of the input tensor.
        /// </summary>
        [Pure]public static (Tensor output, Tensor? inverse_indices, Tensor? counts) unique(
            Tensor input, bool sorted = true, bool return_inverse = false, bool return_counts = false, int? dim = null)
            => input.unique(sorted, return_inverse, return_counts, dim);

        // https://pytorch.org/docs/stable/generated/torch.unique_consecutive
        /// <summary>
        /// Eliminates all but the first element from every consecutive group of equivalent elements.
        /// </summary>
        [Pure]public static (Tensor output, Tensor? inverse_indices, Tensor? counts) unique_consecutive(
            Tensor input, bool return_inverse = false, bool return_counts = false, int? dim = null)
            => input.unique_consecutive(return_inverse, return_counts, dim);

        // https://pytorch.org/docs/stable/generated/torch.var
        /// <summary>
        /// Calculates the variance of all elements in the tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        [Pure]public static Tensor var(Tensor input, bool unbiased = true) => input.var(unbiased);

        /// <summary>Calculates the variance of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample variance is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        [Pure]public static Tensor var(Tensor input, long[] dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.var(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the variance of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample variance is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        [Pure]public static Tensor var(Tensor input, long dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.var(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the variance of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample variance is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        [Pure]public static Tensor var(Tensor input, (long, long) dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.var(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the variance of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample variance is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>The <see cref="Tensor">output tensor</see>.</returns>
        [Pure]public static Tensor var(Tensor input, (long, long, long) dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.var(dimensions, unbiased, keepdim, type);

        // https://pytorch.org/docs/stable/generated/torch.var_mean
        /// <summary>
        /// Calculates the variance and mean of all elements in the tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        [Pure]public static (Tensor @var, Tensor mean) var_mean(Tensor input, bool unbiased = true) => input.std_mean(unbiased);

        /// <summary>Calculates the variance and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample variance is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>A <see cref="Tensor">tensor</see> tuple of the variance and the mean.</returns>
        [Pure]public static (Tensor @var, Tensor mean) var_mean(Tensor input, long[] dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.var_mean(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the variance and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample variance is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>A <see cref="Tensor">tensor</see> tuple of the variance and the mean.</returns>
        [Pure]public static (Tensor @var, Tensor mean) var_mean(Tensor input, ReadOnlySpan<long> dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.var_mean(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the variance and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample variance is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>A <see cref="Tensor">tensor</see> tuple of the variance and the mean.</returns>
        [Pure]public static (Tensor @var, Tensor mean) var_mean(Tensor input, long dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.var_mean(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the variance and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample variance is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>A <see cref="Tensor">tensor</see> tuple of the variance and the mean.</returns>
        [Pure]public static (Tensor @var, Tensor mean) var_mean(Tensor input, (long, long) dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.var_mean(dimensions, unbiased, keepdim, type);

        /// <summary>Calculates the variance and mean of all elements in the tensor.</summary>
        /// <remarks>
        /// If <paramref name="unbiased" /> is <value>true</value>, Bessel’s correction will be used.
        /// Otherwise, the sample variance is calculated, without any correction.
        /// </remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="dimensions">The dimensions to reduce.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        /// <param name="keepdim">Whether the <see cref="Tensor">output tensor</see> has dim retained or not.</param>
        /// <param name="type"></param>
        /// <returns>A <see cref="Tensor">tensor</see> tuple of the variance and the mean.</returns>
        [Pure]public static (Tensor @var, Tensor mean) var_mean(Tensor input, (long, long, long) dimensions, bool unbiased = true, bool keepdim = false, ScalarType? type = null)
            => input.var_mean(dimensions, unbiased, keepdim, type);

        // https://pytorch.org/docs/stable/generated/torch.count_nonzero
        /// <summary>
        /// Counts the number of non-zero values in the tensor input along the given dim. If no dim is specified then all non-zeros in the tensor are counted.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dims">List of dims along which to count non-zeros.</param>
        [Pure]public static Tensor count_nonzero(Tensor input, long[]? dims = null)
            => input.count_nonzero(dims);
    }
}