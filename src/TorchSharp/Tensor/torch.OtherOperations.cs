// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.PInvoke;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#other-operations
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.atleast_1d
        /// <summary>
        /// Returns a 1-dimensional view of each input tensor with zero dimensions. Input tensors with one or more dimensions are returned as-is.
        /// </summary>
        public static IEnumerable<Tensor> atleast_1d(params Tensor[] input) => input.Select(t => t.atleast_1d());

        // https://pytorch.org/docs/stable/generated/torch.atleast_2d
        /// <summary>
        /// Returns a 2-dimensional view of each input tensor with zero or one dimensions. Input tensors with two or more dimensions are returned as-is.
        /// </summary>
        public static IEnumerable<Tensor> atleast_2d(params Tensor[] input) => input.Select(t => t.atleast_2d());

        // https://pytorch.org/docs/stable/generated/torch.atleast_3d
        /// <summary>
        /// Returns a 1-dimensional view of each input tensor with fewer than three dimensions. Input tensors with three or more dimensions are returned as-is.
        /// </summary>
        public static IEnumerable<Tensor> atleast_3d(params Tensor[] input) => input.Select(t => t.atleast_3d());

        // https://pytorch.org/docs/stable/generated/torch.bincount
        /// <summary>
        /// Count the frequency of each value in an array of non-negative ints.
        /// </summary>
        public static Tensor bincount(Tensor input, Tensor? weights = null, long minlength = 0) => input.bincount(weights, minlength);

        // https://pytorch.org/docs/stable/generated/torch.block_diag
        /// <summary>
        /// Create a block diagonal matrix from provided tensors.
        /// </summary>
        /// <param name="tensors">One or more tensors with 0, 1, or 2 dimensions.</param>
        /// <returns></returns>
        public static Tensor block_diag(params Tensor[] tensors)
        {
            using var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            var res = THSTensor_block_diag(tensorsRef, parray.Array.Length);
            if (res == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.broadcast_tensors
        /// <summary>
        /// Broadcasts the given tensors according to Torch broadcasting semantics.
        /// </summary>
        /// <param name="tensors">Any number of tensors of the same type</param>
        public static IList<Tensor> broadcast_tensors(params Tensor[] tensors)
        {
            if (tensors.Length == 0) {
                throw new ArgumentException(nameof(tensors));
            }
            if (tensors.Length == 1) {
                return tensors;
            }

            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            using (var parray = new PinnedArray<IntPtr>()) {

                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                THSTensor_broadcast_tensors(tensorsRef, tensors.Length, pa.CreateArray);
                CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new Tensor(x)).ToList();
        }

        // https://pytorch.org/docs/stable/generated/torch.broadcast_to
        public static Tensor broadcast_to(Tensor input, params long[] shape) => input.broadcast_to(shape);

        // https://pytorch.org/docs/stable/generated/torch.broadcast_shapes
        /// <summary>
        /// This is equivalent to <code>torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape</code>
        /// but avoids the need create to intermediate tensors.
        /// This is useful for broadcasting tensors of common batch shape but different rightmost shape,
        /// e.g. to broadcast mean vectors with covariance matrices.
        /// </summary>
        /// <param name="shapes">Shapes of tensors</param>
        /// <returns>A shape compatible with all input shapes</returns>
        /// <exception cref="ArgumentException">If shapes are incompatible.</exception>
        public static Size broadcast_shapes(params long[][] shapes)
        {
            var max_len = 0;
            foreach (var shape in shapes) {
                var s = shape.Length;
                if (s > max_len) max_len = s;
            }

            var result = Enumerable.Repeat<long>(1, max_len).ToArray();

            foreach (var shape in shapes) {
                for (var i = shape.Length - 1; i >= 0; i--) {
                    if (shape.Length == 0 || shape[i] == 1 || shape[i] == result[i])
                        continue;
                    if (result[i] != 1)
                        throw new ArgumentException("Shape mismatch: objects cannot be broadcast to a single shape");
                    result[i] = shape[i];
                }
            }
            return result;
        }

        // https://pytorch.org/docs/stable/generated/torch.bucketize
        public static Tensor bucketize(Tensor input, Tensor boundaries, bool outInt32 = false, bool right = false)
            => input.bucketize(boundaries, outInt32, right);

        // https://pytorch.org/docs/stable/generated/torch.cartesian_prod
        /// <summary>
        /// Do cartesian product of the given sequence of tensors.
        /// </summary>
        /// <param name="tensors"></param>
        public static Tensor cartesian_prod(IList<Tensor> tensors)
        {
            using var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            var res = THSTensor_cartesian_prod(tensorsRef, parray.Array.Length);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.cartesian_prod
        /// <summary>
        /// Do cartesian product of the given sequence of tensors.
        /// </summary>
        /// <param name="tensors"></param>
        public static Tensor cartesian_prod(params Tensor[] tensors) => cartesian_prod((IList<Tensor>)tensors);

        // https://pytorch.org/docs/stable/generated/torch.cdist
        /// <summary>
        /// Computes batched the p-norm distance between each pair of the two collections of row vectors.
        /// </summary>
        /// <param name="x1">Input tensor of shape BxPxM</param>
        /// <param name="x2">Input tensor of shape BxRxM</param>
        /// <param name="p">p value for the p-norm distance to calculate between each vector (p > 0)</param>
        /// <param name="compute_mode">
        /// use_mm_for_euclid_dist_if_necessary - will use matrix multiplication approach to calculate euclidean distance (p = 2) if P > 25 or R > 25
        /// use_mm_for_euclid_dist - will always use matrix multiplication approach to calculate euclidean distance (p = 2)
        /// donot_use_mm_for_euclid_dist - will never use matrix multiplication approach to calculate euclidean distance (p = 2)
        /// </param>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor cdist(
            Tensor x1,
            Tensor x2,
            double p = 2.0,
            compute_mode compute_mode = compute_mode.use_mm_for_euclid_dist_if_necessary)
        {
            if (p < 0)
                throw new ArgumentException($"p must be non-negative");

            var res = THSTensor_cdist(x1.Handle, x2.Handle, p, (long)compute_mode);
            if (res == IntPtr.Zero)
                CheckForErrors();
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.clone
        public static Tensor clone(Tensor input) => input.clone();

        // https://pytorch.org/docs/stable/generated/torch.combinations
        /// <summary>
        /// Compute combinations of length r of the given tensor
        /// </summary>
        /// <param name="input">1D vector.</param>
        /// <param name="r">Number of elements to combine</param>
        /// <param name="with_replacement">Whether to allow duplication in combination</param>
        /// <returns></returns>
        public static Tensor combinations(Tensor input, int r = 2, bool with_replacement = false)
        {
            if (input.ndim != 1)
                throw new ArgumentException($"Expected a 1D vector, but got one with {input.ndim} dimensions.");
            if (r < 0)
                throw new ArgumentException($"r must be non-negative");

            var res = THSTensor_combinations(input.Handle, r, with_replacement);
            if (res == IntPtr.Zero)
                CheckForErrors();
            return new Tensor(res);
        }



        // https://pytorch.org/docs/stable/generated/torch.corrcoef
        public static Tensor corrcoef(Tensor input) => input.corrcoef();

        // https://pytorch.org/docs/stable/generated/torch.cov
        /// <summary>
        /// Estimates the covariance matrix of the variables given by the input matrix, where rows are the variables and columns are the observations.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="correction">
        /// Difference between the sample size and sample degrees of freedom.
        /// Defaults to Bessel’s correction, correction = 1 which returns the unbiased estimate,
        /// even if both fweights and aweights are specified.
        /// Correction = 0 will return the simple average.
        /// </param>
        /// <param name="fweights">
        /// A Scalar or 1D tensor of observation vector frequencies representing the number of times each observation should be repeated.
        /// Its numel must equal the number of columns of input.
        /// Must have integral dtype.</param>
        /// <param name="aweights">A Scalar or 1D array of observation vector weights.
        /// These relative weights are typically large for observations considered “important” and smaller for
        /// observations considered less “important”.
        /// Its numel must equal the number of columns of input.
        /// Must have floating point dtype.</param>
        public static Tensor cov(Tensor input, long correction = 1, Tensor? fweights = null, Tensor? aweights = null)
            => input.cov(correction, fweights, aweights);

        // https://pytorch.org/docs/stable/generated/torch.cross
        /// <summary>
        /// Returns the cross product of vectors in dimension dim of input and other.
        /// input and other must have the same size, and the size of their dim dimension should be 3.
        /// </summary>
        public static Tensor cross(Tensor input, Scalar other, long dim = 0L) => input.cross(other, dim);

        // https://pytorch.org/docs/stable/generated/torch.cummax
        public static (Tensor values, Tensor indices) cummax(Tensor input, long dim) => input.cummax(dim);

        // https://pytorch.org/docs/stable/generated/torch.cummin
        public static (Tensor values, Tensor indices) cummin(Tensor input, long dim) => input.cummin(dim);

        // https://pytorch.org/docs/stable/generated/torch.cumprod
        public static Tensor cumprod(Tensor input, long dim, ScalarType? dtype = null) => input.cumprod(dim, dtype);

        // https://pytorch.org/docs/stable/generated/torch.cumsum
        /// <summary>
        /// Returns the cumulative sum of elements of input in the dimension dim.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to do the operation over</param>
        /// <param name="type">The desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed.
        /// This is useful for preventing data type overflows.</param>
        public static Tensor cumsum(Tensor input, long dim, ScalarType? type = null) => input.cumsum(dim, type);

        // https://pytorch.org/docs/stable/generated/torch.diag
        /// <summary>
        /// If input is a vector (1-D tensor), then returns a 2-D square tensor with the elements of input as the diagonal.
        /// If input is a matrix (2-D tensor), then returns a 1-D tensor with the diagonal elements of input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="diagonal">
        /// The argument diagonal controls which diagonal to consider:
        /// If diagonal is 0, it is the main diagonal.
        /// If diagonal is greater than 0, it is above the main diagonal.
        /// If diagonal is less than 0, it is below the main diagonal.
        /// </param>
        public static Tensor diag(Tensor input, long diagonal = 0) => input.diag(diagonal);

        // https://pytorch.org/docs/stable/generated/torch.diag_embed
        /// <summary>
        /// Creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2) are filled by input.
        /// To facilitate creating batched diagonal matrices, the 2D planes formed by the last two dimensions of the returned tensor are chosen by default.
        ///
        /// The argument offset controls which diagonal to consider:
        ///   If offset is equal to 0, it is the main diagonal.
        ///   If offset is greater than 0, it is above the main diagonal.
        ///   If offset is less than 0, it is below the main diagonal.
        ///
        /// The size of the new matrix will be calculated to make the specified diagonal of the size of the last input dimension.Note that for offset other than 0,
        ///
        /// the order of dim1 and dim2 matters.Exchanging them is equivalent to changing the sign of offset.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="offset">Which diagonal to consider.</param>
        /// <param name="dim1">First dimension with respect to which to take diagonal. </param>
        /// <param name="dim2">Second dimension with respect to which to take diagonal</param>
        public static Tensor diag_embed(Tensor input, long offset = 0L, long dim1 = -2L, long dim2 = -1L)
            => input.diag_embed(offset, dim1, dim2);

        // https://pytorch.org/docs/stable/generated/torch.diagflat
        /// <summary>
        /// If input is a vector (1-D tensor), then returns a 2-D square tensor with the elements of input as the diagonal.
        /// If input is a matrix (2-D tensor), then returns a 2-D tensor with diagonal elements equal to a flattened input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="offset">
        /// The argument diagonal controls which diagonal to consider:
        /// If diagonal is 0, it is the main diagonal.
        /// If diagonal is greater than 0, it is above the main diagonal.
        /// If diagonal is less than 0, it is below the main diagonal.
        /// </param>
        public static Tensor diagflat(Tensor input, long offset = 0) => input.diagflat(offset);

        // https://pytorch.org/docs/stable/generated/torch.diagonal
        /// <summary>
        /// Returns a partial view of input with the its diagonal elements with respect to dim1 and dim2 appended as a dimension at the end of the shape.
        /// The argument offset controls which diagonal to consider:
        ///
        ///     If offset == 0, it is the main diagonal.
        ///     If offset &gt; 0, it is above the main diagonal.
        ///     If offset &lt; 0, it is below the main diagonal.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="offset">Which diagonal to consider. Default: 0 (main diagonal).</param>
        /// <param name="dim1">First dimension with respect to which to take diagonal. Default: 0.</param>
        /// <param name="dim2">Second dimension with respect to which to take diagonal. Default: 1.</param>
        /// <remarks>
        /// Applying torch.diag_embed() to the output of this function with the same arguments yields a diagonal matrix with the diagonal entries of the input.
        /// However, torch.diag_embed() has different default dimensions, so those need to be explicitly specified.
        /// </remarks>
        public static Tensor diagonal(Tensor input, long offset = 0, long dim1 = 0, long dim2 = 0) => input.diagonal(offset, dim1, dim2);

        // https://pytorch.org/docs/stable/generated/torch.diff
        /// <summary>
        /// Computes the n-th forward difference along the given dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="n">The number of times to recursively compute the difference</param>
        /// <param name="dim">The dimension to compute the difference along. Default is the last dimension.</param>
        /// <param name="prepend">
        /// Values to prepend or append to input along dim before computing the difference.
        /// Their dimensions must be equivalent to that of input, and their shapes must match input’s shape except on dim.
        /// </param>
        /// <param name="append">
        /// Values to prepend or append to input along dim before computing the difference.
        /// Their dimensions must be equivalent to that of input, and their shapes must match input’s shape except on dim.
        /// </param>
        public static Tensor diff(Tensor input, long n = 1, long dim = -1, Tensor? prepend = null, Tensor? append = null) => input.diff(n, dim, prepend, append);

        // https://pytorch.org/docs/stable/generated/torch.einsum
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
            using var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            var res = THSTensor_einsum(equation, tensorsRef, parray.Array.Length);
            if (res == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.flatten
        /// <summary>
        /// Flattens input by reshaping it into a one-dimensional tensor.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="start_dim">The first dim to flatten</param>
        /// <param name="end_dim">The last dim to flatten.</param>
        /// <remarks>Flattening a zero-dimensional tensor will return a one-dimensional view.</remarks>
        public static Tensor flatten(Tensor input, long start_dim = 0, long end_dim = -1) => input.flatten(start_dim, end_dim);

        // https://pytorch.org/docs/stable/generated/torch.flip
        public static Tensor flip(Tensor input, params long[] dims) => input.flip(dims);

        // https://pytorch.org/docs/stable/generated/torch.fliplr
        public static Tensor fliplr(Tensor input) => input.fliplr();

        // https://pytorch.org/docs/stable/generated/torch.flipud
        public static Tensor flipud(Tensor input) => input.flipud();

        // https://pytorch.org/docs/stable/generated/torch.kron
        /// <summary>
        /// Computes the Kronecker product of input and other.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="other">The second tensor</param>
        public static Tensor kron(Tensor input, Tensor other) => input.kron(other);

        // https://pytorch.org/docs/stable/generated/torch.rot90
        /// <summary>
        /// Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.
        /// Rotation direction is from the first towards the second axis if k is greater than 0,
        /// and from the second towards the first for k less than 0.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="k">The number of times to rotate.</param>
        /// <param name="dims">Axes to rotate</param>
        public static Tensor rot90(Tensor input, long k = 1, (long, long)? dims = null) => input.rot90(k, dims);

        // https://pytorch.org/docs/stable/generated/torch.gcd
        /// <summary>
        /// Computes the element-wise greatest common divisor (GCD) of input and other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor gcd(Tensor left, Tensor right) => left.gcd(right);

        // https://pytorch.org/docs/stable/generated/torch.gcd
        /// <summary>
        /// Computes the element-wise greatest common divisor (GCD) of input and other.
        /// </summary>
        /// <param name="left">The left-hand operand.</param>
        /// <param name="right">The right-hand operand.</param>
        public static Tensor gcd_(Tensor left, Tensor right) => left.gcd_(right);

        // https://pytorch.org/docs/stable/generated/torch.histc
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

        // https://pytorch.org/docs/stable/generated/torch.histogram
        [Obsolete("not implemented", true)]
        static Tensor histogram(
            Tensor input,
            long bins,
            (float min, float max)? range = null,
            Tensor? weight = null,
            bool density = false)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.histogram
        [Obsolete("not implemented", true)]
        static Tensor histogram(
            Tensor input,
            long[] bins,
            (float min, float max)? range = null,
            Tensor? weight = null,
            bool density = false)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.histogram
        [Obsolete("not implemented", true)]
        static Tensor histogram(
            Tensor input,
            Tensor[] bins,
            (float min, float max)? range = null,
            Tensor? weight = null,
            bool density = false)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.histogramdd
        [Obsolete("not implemented", true)]
        static (Tensor hist_values, Tensor[] edges) histogramdd(
            Tensor input,
            long bins,
            (float min, float max)? range = null,
            Tensor? weight = null,
            bool density = false)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.histogramdd
        [Obsolete("not implemented", true)]
        static (Tensor hist_values, Tensor[] edges) histogramdd(
            Tensor input,
            long[] bins,
            (float min, float max)? range = null,
            Tensor? weight = null,
            bool density = false)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.histogramdd
        [Obsolete("not implemented", true)]
        static (Tensor hist_values, Tensor[] edges) histogramdd(
            Tensor input,
            Tensor[] bins,
            (float min, float max)? range = null,
            Tensor? weight = null,
            bool density = false)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.meshgrid
        /// <summary>
        /// Creates grids of coordinates specified by the 1D inputs in tensors.
        /// This is helpful when you want to visualize data over some range of inputs.
        /// </summary>
        /// <returns></returns>
        /// <remarks>All tensors need to be of the same size.</remarks>
        static IEnumerable<Tensor> meshgrid(IEnumerable<Tensor> tensors, indexing indexing = indexing.ij)
        {
            var idx = indexing switch {
                indexing.ij => "ij",
                indexing.xy => "xy",
                _ => throw new ArgumentOutOfRangeException()
            };
            return meshgrid(tensors, idx);
        }

        // https://pytorch.org/docs/stable/generated/torch.meshgrid
        /// <summary>
        /// Creates grids of coordinates specified by the 1D inputs in tensors.
        /// This is helpful when you want to visualize data over some range of inputs.
        /// </summary>
        /// <returns></returns>
        /// <remarks>All tensors need to be of the same size.</remarks>
        public static Tensor[] meshgrid(IEnumerable<Tensor> tensors, string indexing = "ij")
        {
            IntPtr[] ptrArray;

            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());
                _ = THSTensor_meshgrid(tensorsRef, parray.Array.Length, indexing, parray.CreateArray);
                CheckForErrors();
                ptrArray = parray.Array;
            }
            return ptrArray.Select(x => new Tensor(x)).ToArray();
        }

        // https://pytorch.org/docs/stable/generated/torch.lcm
        /// <summary>
        /// Computes the element-wise least common multiple (LCM) of input and other.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <remarks>Both input and other must have integer types.</remarks>
        public static Tensor lcm(Tensor input, Tensor other) => input.lcm(other);

        // https://pytorch.org/docs/stable/generated/torch.lcm
        /// <summary>
        /// Computes the element-wise least common multiple (LCM) of input and other in place.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <remarks>Both input and other must have integer types.</remarks>
        public static Tensor lcm_(Tensor input, Tensor other) => input.lcm_(other);

        // https://pytorch.org/docs/stable/generated/torch.logcumsumexp
        /// <summary>
        /// Returns the logarithm of the cumulative summation of the exponentiation of elements of input in the dimension dim.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to do the operation over</param>
        public static Tensor logcumsumexp(Tensor input, long dim) => input.logcumsumexp(dim);

        // https://pytorch.org/docs/stable/generated/torch.ravel
        public static Tensor ravel(Tensor input) => input.ravel();

        // https://pytorch.org/docs/stable/generated/torch.renorm
        /// <summary>
        /// Returns a tensor where each sub-tensor of input along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the value maxnorm
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="p">The power for the norm computation</param>
        /// <param name="dim">The dimension to slice over to get the sub-tensors</param>
        /// <param name="maxnorm">The maximum norm to keep each sub-tensor under</param>
        public static Tensor renorm(Tensor input, float p, long dim, float maxnorm) => input.renorm(p, dim, maxnorm);

        // https://pytorch.org/docs/stable/generated/torch.repeat_interleave
        /// <summary>
        /// Repeat elements of a tensor.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="repeats">The number of repeats</param>
        /// <param name="dim">The dimension to repeat</param>
        /// <param name="output_size">The size of output</param>
        /// <returns></returns>
        public static Tensor repeat_interleave(Tensor input, long repeats, long? dim = null, long? output_size = null) => input.repeat_interleave(repeats, dim, output_size);

        // https://pytorch.org/docs/stable/generated/torch.repeat_interleave
        /// <summary>
        /// Repeat elements of a tensor.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="repeats">The number of repeats</param>
        /// <param name="dim">The dimension to repeat</param>
        /// <param name="output_size">The size of output</param>
        /// <returns></returns>
        public static Tensor repeat_interleave(Tensor input, Tensor repeats, long? dim = null, long? output_size = null) => input.repeat_interleave(repeats, dim, output_size);

        // https://pytorch.org/docs/stable/generated/torch.roll
        /// <summary>
        /// Roll the tensor along the given dimension(s).
        /// Elements that are shifted beyond the last position are re-introduced at the first position.
        /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
        /// </summary>
        public static Tensor roll(Tensor input, long shifts, long? dims = null) => input.roll(shifts, dims);

        // https://pytorch.org/docs/stable/generated/torch.roll
        /// <summary>
        /// Roll the tensor along the given dimension(s).
        /// Elements that are shifted beyond the last position are re-introduced at the first position.
        /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
        /// </summary>
        public static Tensor roll(Tensor input, (long, long) shifts, (long, long) dims) => input.roll(shifts, dims);

        // https://pytorch.org/docs/stable/generated/torch.roll
        /// <summary>
        /// Roll the tensor along the given dimension(s).
        /// Elements that are shifted beyond the last position are re-introduced at the first position.
        /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
        /// </summary>
        public static Tensor roll(Tensor input, (long, long, long) shifts, (long, long, long) dims) => input.roll(shifts, dims);

        // https://pytorch.org/docs/stable/generated/torch.roll
        /// <summary>
        /// Roll the tensor along the given dimension(s).
        /// Elements that are shifted beyond the last position are re-introduced at the first position.
        /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
        /// </summary>
        public static Tensor roll(Tensor input, long[] shifts, long[] dims) => input.roll(shifts, dims);

        // https://pytorch.org/docs/stable/generated/torch.roll
        /// <summary>
        /// Roll the tensor along the given dimension(s).
        /// Elements that are shifted beyond the last position are re-introduced at the first position.
        /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
        /// </summary>
        public static Tensor roll(Tensor input, long[] shifts) => input.roll(shifts);

        // https://pytorch.org/docs/stable/generated/torch.roll
        /// <summary>
        /// Roll the tensor along the given dimension(s).
        /// Elements that are shifted beyond the last position are re-introduced at the first position.
        /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
        /// </summary>
        public static Tensor roll(Tensor input, ReadOnlySpan<long> shifts, ReadOnlySpan<long> dims = default) => input.roll(shifts, dims);

        // https://pytorch.org/docs/stable/generated/torch.searchsorted
        [Obsolete("not implemented", true)]
        static Tensor searchsorted(
            Tensor sorted_sequence,
            Tensor values,
            bool out_int32 = false,
            bool right = false,
            side side = side.left,
            Tensor? sorter = null)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.tensordot
        /// <summary>
        /// Returns a contraction of <paramref name="a"/> and <paramref name="b"/> over multiple dimensions.
        /// tensordot implements a generalized matrix product.
        /// </summary>
        /// <param name="a">Left tensor to contract</param>
        /// <param name="b">Right tensor to contract</param>
        /// <param name="dims">number of dimensions to contract for <paramref name="a"/> and <paramref name="b"/></param>
        /// <returns>contraction</returns>
        public static Tensor tensordot(Tensor a, Tensor b, long dims = 2) => a.tensordot(b, dims);

        // https://pytorch.org/docs/stable/generated/torch.tensordot
        /// <summary>
        /// Returns a contraction of <paramref name="a"/> and <paramref name="b"/> over multiple dimensions.
        /// tensordot implements a generalized matrix product.
        /// </summary>
        /// <param name="a">Left tensor to contract</param>
        /// <param name="b">Right tensor to contract</param>
        /// <param name="dims1">dimensions to contract for <paramref name="a"/></param>
        /// <param name="dims2">dimensions to contract for <paramref name="b"/></param>
        /// <returns>contraction</returns>
        public static Tensor tensordot(Tensor a, Tensor b, long[] dims1, long[] dims2) => a.tensordot(b, dims1, dims2);

        // https://pytorch.org/docs/stable/generated/torch.tensordot
        /// <summary>
        /// Returns a contraction of <paramref name="a"/> and <paramref name="b"/> over multiple dimensions.
        /// tensordot implements a generalized matrix product.
        /// </summary>
        /// <param name="a">Left tensor to contract</param>
        /// <param name="b">Right tensor to contract</param>
        /// <param name="dims">dimensions to contract for <paramref name="a"/> and <paramref name="b"/> respectively</param>
        /// <returns>contraction</returns>
        public static Tensor tensordot(Tensor a, Tensor b, (long, long)[] dims) => a.tensordot(b, dims);

        // https://pytorch.org/docs/stable/generated/torch.trace
        /// <summary>
        /// Returns the sum of the elements of the diagonal of the input 2-D matrix.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <returns></returns>
        public static Tensor trace(Tensor input) => input.trace();

        // https://pytorch.org/docs/stable/generated/torch.tril
        public static Tensor tril(Tensor input, long diagonal = 0) => input.tril(diagonal);

        // https://pytorch.org/docs/stable/generated/torch.tril_indices
        public static Tensor tril_indices(
            long row,
            long col,
            long offset = 0L,
            ScalarType dtype = ScalarType.Int64,
            Device? device = null)
        {
            if (!torch.is_integral(dtype))
                throw new ArgumentException("dtype must be integral.");

            if (device == null) {
                device = torch.CPU;
            }

            var res = LibTorchSharp.THSTensor_tril_indices(row, col, offset, (sbyte)dtype, (int)device.type, device.index);
            if (res == IntPtr.Zero)
                CheckForErrors();
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.triu
        public static Tensor triu(Tensor input, long diagonal = 0L) => input.triu(diagonal);

        // https://pytorch.org/docs/stable/generated/torch.triu_indices
        public static Tensor triu_indices(
            long row,
            long col,
            long offset = 0L,
            ScalarType dtype = ScalarType.Int64,
            Device? device = null)
        {
            if (!torch.is_integral(dtype))
                throw new ArgumentException("dtype must be integral.");

            if (device == null) {
                device = torch.CPU;
            }

            var res = LibTorchSharp.THSTensor_triu_indices(row, col, offset, (sbyte)dtype, (int)device.type, device.index);
            if (res == IntPtr.Zero)
                CheckForErrors();
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.vander
        public static Tensor vander(Tensor x, long N = -1, bool increasing = false) => x.vander(N, increasing);

        // https://pytorch.org/docs/stable/generated/torch.view_as_real
        /// <summary>
        /// Returns a view of input as a real tensor.
        /// For an input complex tensor of size m1, m2, …, mi, this function returns a new real tensor of size m1, m2, …, mi, 2, where the last dimension of size 2 represents the real and imaginary components of complex numbers.
        /// </summary>
        /// <param name="input">The input tensor</param>
        public static Tensor view_as_real(Tensor input) => input.view_as_real();

        // https://pytorch.org/docs/stable/generated/torch.view_as_complex
        /// <summary>
        /// Returns a view of input as a complex tensor.
        /// For an input complex tensor of size m1, m2, …, mi, 2, this function returns a new complex tensor of size m1, m2, …, mi where the last dimension of the input tensor is expected to represent the real and imaginary components of complex numbers.
        /// </summary>
        /// <param name="input">The input tensor</param>
        public static Tensor view_as_complex(Tensor input) => input.view_as_complex();

        // https://pytorch.org/docs/stable/generated/torch.resolve_conj
        /// <summary>
        /// Returns a new tensor with materialized conjugation if input’s conjugate bit is set to True, else returns input.
        /// The output tensor will always have its conjugate bit set to False.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor resolve_conj(Tensor input) => input.resolve_conj();

        // https://pytorch.org/docs/stable/generated/torch.resolve_neg
        /// <summary>
        /// Returns a new tensor with materialized negation if input’s negative bit is set to True, else returns input.
        /// The output tensor will always have its negative bit set to False.
        /// </summary>
        public static Tensor resolve_neg(Tensor input) => input.resolve_neg();

        /// <summary>
        /// Returns true if the input's negative bit is set to True.
        /// </summary>
        public static Tensor is_neg(Tensor input) => input.is_neg();
    }
}