// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    // This file contains methods that are manipulating Tensors, and are not also defined on the Tensor class.
    public static partial class torch
    {
        /// <summary>
        /// Computes the Cholesky decomposition of a symmetric positive-definite matrix 'input' or for batches of symmetric positive-definite matrices.
        /// </summary>
        /// <param name="input">The input matrix</param>
        /// <param name="upper">If upper is true, the returned matrix U is upper-triangular. If upper is false, the returned matrix L is lower-triangular</param>
        /// <returns></returns>
        [Obsolete("torch.cholesky is deprecated in favor of torch.linalg.cholesky and will be removed in a future release. Use torch.linalg.cholesky instead.", false)]
#pragma warning disable CS0618 // Obsolete
        public static Tensor cholesky(Tensor input, bool upper) => input.cholesky(upper);
#pragma warning restore CS0618

        /// <summary>
        /// Returns the matrix norm or vector norm of a given tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor norm(Tensor input) => input.norm();

        /// <summary>
        /// Concatenates the given sequence of tensors along the given axis (dimension).
        /// </summary>
        /// <param name="tensors">A sequence of tensors of the same type. Non-empty tensors provided must have the same shape, except in the cat dimension.</param>
        /// <param name="axis">The dimension over which the tensors are concatenated</param>
        /// <returns>A tensor resulting from concatenating the input tensors along <paramref name="axis"/>.</returns>
        /// <remarks> All tensors must either have the same shape (except in the concatenating dimension) or be empty.</remarks>
        public static Tensor concatenate(IList<Tensor> tensors, long axis = 0) => torch.cat(tensors, axis);

        /// <summary>
        /// Concatenates the given sequence of tensors along the given axis (dimension).
        /// </summary>
        /// <param name="tensors">A sequence of tensors of the same type. Non-empty tensors provided must have the same shape, except in the cat dimension.</param>
        /// <param name="axis">The dimension over which the tensors are concatenated</param>
        /// <returns>A tensor resulting from concatenating the input tensors along <paramref name="axis"/>.</returns>
        /// <remarks> All tensors must either have the same shape (except in the concatenating dimension) or be empty.</remarks>
        public static Tensor concatenate(Tensor[] tensors, long axis = 0) => torch.cat(tensors, axis);

        /// <summary>
        /// Concatenates the given sequence of tensors along the given axis (dimension).
        /// </summary>
        /// <param name="tensors">A sequence of tensors of the same type. Non-empty tensors provided must have the same shape, except in the cat dimension.</param>
        /// <param name="axis">The dimension over which the tensors are concatenated</param>
        /// <returns>A tensor resulting from concatenating the input tensors along <paramref name="axis"/>.</returns>
        /// <remarks> All tensors must either have the same shape (except in the concatenating dimension) or be empty.</remarks>
        public static Tensor concatenate(ReadOnlySpan<Tensor> tensors, long axis = 0) => torch.cat(tensors, axis);

        /// <summary>
        /// Returns a tensor with all the dimensions of input of size 1 removed. When dim is given, a squeeze operation is done only in the given dimension.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">If given, the input will be squeezed only in this dimension</param>
        /// <returns></returns>
        public static Tensor squeeze(Tensor input, long? dim = null) => input.squeeze(dim);

        /// <summary>
        /// Modifies (in-place) a tensor with all the dimensions of input of size 1 removed. When dim is given, a squeeze operation is done only in the given dimension.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">If given, the input will be squeezed only in this dimension</param>
        /// <returns></returns>
        public static Tensor squeeze_(Tensor input, long? dim = null) => input.squeeze_(dim);

        /// <summary>
        /// Creates a new tensor by horizontally stacking the input tensors.
        /// </summary>
        /// <param name="tensors">A list of input tensors.</param>
        /// <returns>A tensor formed by horizontally stacking the inputs. Zero- or one-dimensional tensors are first reshaped into (numel, 1) columns.</returns>
        /// <remarks>Equivalent to torch.hstack(tensors), except each zero or one dimensional tensor t in tensors is first reshaped into a (t.numel(), 1) column before being stacked horizontally.</remarks>
        public static Tensor column_stack(IList<Tensor> tensors) => column_stack(tensors.ToHandleArray());

        /// <summary>
        /// Creates a new tensor by horizontally stacking the input tensors.
        /// </summary>
        /// <param name="tensors">An array of input tensors.</param>
        /// <returns>A tensor formed by horizontally stacking the inputs. Zero- or one-dimensional tensors are first reshaped into (numel, 1) columns.</returns>
        /// <remarks>Equivalent to torch.hstack(tensors), except each zero or one dimensional tensor t in tensors is first reshaped into a (t.numel(), 1) column before being stacked horizontally.</remarks>
        public static Tensor column_stack(params Tensor[] tensors) => column_stack(tensors.ToHandleArray());

        /// <summary>
        /// Creates a new tensor by horizontally stacking the input tensors.
        /// </summary>
        /// <param name="tensors">A span of input tensors.</param>
        /// <returns>A tensor formed by horizontally stacking the inputs. Zero- or one-dimensional tensors are first reshaped into (numel, 1) columns.</returns>
        /// <remarks>Equivalent to torch.hstack(tensors), except each zero or one dimensional tensor t in tensors is first reshaped into a (t.numel(), 1) column before being stacked horizontally.</remarks>
        public static Tensor column_stack(ReadOnlySpan<Tensor> tensors) => column_stack(tensors.ToHandleArray());

        static Tensor column_stack(IntPtr[] tensors)
        {
            using var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors);

            var res = THSTensor_column_stack(tensorsRef, parray.Array.Length);
            if (res == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(res);
        }

        /// <summary>
        /// Stack tensors in sequence vertically (row wise).
        /// </summary>
        /// <param name="tensors">A list of input tensors.</param>
        /// <returns>A tensor formed by stacking the inputs row-wise (vertically).</returns>
        public static Tensor row_stack(IList<Tensor> tensors) => row_stack(tensors.ToHandleArray());

        /// <summary>
        /// Stack tensors in sequence vertically (row wise).
        /// </summary>
        /// <param name="tensors">An array of input tensors.</param>
        /// <returns>A tensor formed by stacking the inputs row-wise (vertically).</returns>
        public static Tensor row_stack(params Tensor[] tensors) => row_stack(tensors.ToHandleArray());

        /// <summary>
        /// Stack tensors in sequence vertically (row wise).
        /// </summary>
        /// <param name="tensors">A span of input tensors.</param>
        /// <returns>A tensor formed by stacking the inputs row-wise (vertically).</returns>
        public static Tensor row_stack(ReadOnlySpan<Tensor> tensors) => row_stack(tensors.ToHandleArray());

        static Tensor row_stack(IntPtr[] tensors)
        {
            using var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors);

            var res = THSTensor_row_stack(tensorsRef, parray.Array.Length);
            if (res == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(res);
        }

        /// <summary>
        /// Removes a tensor dimension.
        /// </summary>
        /// <param name="tensor">The input tensor</param>
        /// <param name="dim">The dimension to remove.</param>
        /// <returns>An array of all slices along a given dimension, already without it.</returns>
        public static Tensor[] unbind(Tensor tensor, int dim = 0) => tensor.unbind(dim);

        /// <summary>
        /// Adds all values from the tensor other into input at the indices specified in the index tensor in a similar fashion as scatter_().
        /// For each value in src, it is added to an index in self which is specified by its index in src for dimension != dim and by the
        /// corresponding value in index for dimension = dim.
        /// </summary>
        public static Tensor scatter_add_(Tensor input, long dim, Tensor index, Tensor src) => input.scatter_add_(dim, index, src);

        public static Tensor clamp_max(Tensor input, Scalar max) => input.clamp_max(max);

        public static Tensor clamp_max_(Tensor input, Scalar max) => input.clamp_max(max);

        public static Tensor clamp_min(Tensor input, Scalar min) => input.clamp_min(min);

        public static Tensor clamp_min_(Tensor input, Scalar min) => input.clamp_min(min);

        /// <summary>
        /// Expands the dimension dim of the self tensor over multiple dimensions of sizes given by sizes.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">Dimension to unflatten.</param>
        /// <param name="sizes">New shape of the unflattened dimension.</param>
        public static Tensor unflatten(Tensor input, long dim, params long[] sizes) => input.unflatten(dim, sizes);

        /// <summary>
        /// Expands the dimension dim of the self tensor over multiple dimensions of sizes given by sizes.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">Dimension to unflatten.</param>
        /// <param name="sizes">New shape of the unflattened dimension.</param>
        public static Tensor unflatten(Tensor input, long dim, Size sizes) => input.unflatten(dim, sizes.Shape);

        public static Tensor _standard_gamma(Tensor input, Generator? generator = null)
        {
            var res = THSTensor_standard_gamma_(input.Handle, generator is null ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(res);
        }

        public static Tensor _sample_dirichlet(Tensor input, Generator? generator = null)
        {
            var res = THSTensor_sample_dirichlet_(input.Handle, generator is null ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(res);
        }

        /// <summary>
        /// Fills the elements of the input tensor with value value by selecting the indices in the order given in index.
        ///
        /// For example, if dim == 0, index[i] == j, and alpha=-1, then the ith row of source is subtracted from the jth row of the input tensor.
        /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match self, or an error will be raised.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">Dimension along which to index</param>
        /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
        /// <param name="value">The scalar multiplier for source</param>
        /// <returns></returns>
        public static Tensor index_fill(Tensor input, long dim, Tensor index, Scalar value) => input.index_fill(dim, index, value);

        /// <summary>
        /// Fills, in place, the elements of the input tensor with value value by selecting the indices in the order given in index.
        ///
        /// For example, if dim == 0, index[i] == j, and alpha=-1, then the ith row of source is subtracted from the jth row of the input tensor.
        /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match self, or an error will be raised.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">Dimension along which to index</param>
        /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
        /// <param name="value">The scalar multiplier for source</param>
        /// <returns></returns>
        public static Tensor index_fill_(Tensor input, long dim, Tensor index, Scalar value) => input.index_fill_(dim, index, value);

        /// <summary>
        /// Returns true if the input is a conjugated tensor, i.e. its conjugate bit is set to True.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static bool is_conj(Tensor input) => input.is_conj();

        /// <summary>
        /// Calculates the standard deviation and mean of all elements in the tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="unbiased">Whether to use Bessel’s correction (δN=1).</param>
        public static (Tensor std, Tensor mean) std_mean(Tensor input, bool unbiased = true) => input.std_mean(unbiased);

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public static Tensor rand_out(Tensor input, params long[] sizes) => input.randn_out(sizes);

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randint_out(Tensor input, long high, long[] sizes) => input.randint_out(high, sizes);

        /// <summary>
        /// Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public static Tensor randn_like(Tensor input, ScalarType? dtype = null, Device? device = null, bool requires_grad = false) => input.randn_like(dtype, device, requires_grad);

        /// <summary>
        /// Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly in the range [low,high).
        /// </summary>
        public static Tensor randint_like(Tensor input, long low, long high, ScalarType? dtype = null, Device? device = null, bool requires_grad = false) => input.randint_like(low, high, dtype, device, requires_grad);

        /// <summary>
        ///  Mutates the tensor to be a 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        [Obsolete("This doesn't exist in PyTorch.")]
        public static Tensor randperm_out(Tensor input, long n) => torch.randperm(n, input);

        /// <summary>
        /// Draws a binomial distribution given a trial count and probabilities.
        /// </summary>
        /// <param name="count">Trial count</param>
        /// <param name="probs">Probability vector</param>
        /// <param name="generator">Optional random number generator</param>
        /// <returns></returns>
        public static Tensor binomial(Tensor count, Tensor probs, Generator? generator = null) => count.binomial(probs, generator);
    }
}
