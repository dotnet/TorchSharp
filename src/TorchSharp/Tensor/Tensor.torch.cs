// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;

namespace TorchSharp
{
    // This file contains methods that are manipulating Tensors, and are not also defined on the Tensor class.

    public static partial class torch

    {
        /// <summary>
        /// Returns a 1-dimensional view of each input tensor with zero dimensions. Input tensors with one or more dimensions are returned as-is.
        /// </summary>
        public static IEnumerable<Tensor> atleast_1d(params Tensor[] input) => input.Select(t => t.atleast_1d());

        /// <summary>
        /// Returns a 2-dimensional view of each input tensor with zero or one dimensions. Input tensors with two or more dimensions are returned as-is.
        /// </summary>
        public static IEnumerable<Tensor> atleast_2d(params Tensor[] input) => input.Select(t => t.atleast_2d());

        /// <summary>
        /// Returns a 1-dimensional view of each input tensor with fewer than three dimensions. Input tensors with three or more dimensions are returned as-is.
        /// </summary>
        public static IEnumerable<Tensor> atleast_3d(params Tensor[] input) => input.Select(t => t.atleast_3d());

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_block_diag(IntPtr tensor, int len);

        /// <summary>
        /// Create a block diagonal matrix from provided tensors.
        /// </summary>
        /// <param name="tensors">One or more tensors with 0, 1, or 2 dimensions.</param>
        /// <returns></returns>
        public static Tensor block_diag(params Tensor[] tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_block_diag(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_broadcast_tensors(IntPtr tensor, long length, AllocatePinnedArray allocator);

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
                torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new Tensor(x)).ToList();
        }

        /// <summary>
        /// Returns a tensor containing the indices of all non-zero elements of input.
        /// Each row in the result contains the indices of a non-zero element in input.
        /// The result is sorted lexicographically, with the last index changing the fastest (C-style).
        /// </summary>
        public static Tensor nonzero(Tensor input) => input.nonzero();


        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_cat(IntPtr tensor, int len, long dim);

        /// <summary>
        /// Concatenates the given sequence of seq tensors in the given dimension.
        /// </summary>
        /// <param name="tensors"></param>
        /// <param name="dimension"></param>
        /// <returns></returns>
        /// <remarks> All tensors must either have the same shape (except in the concatenating dimension) or be empty.</remarks>
        public static Tensor cat(IList<Tensor> tensors, long dimension)
        {
            if (tensors.Count == 0) {
                throw new ArgumentException(nameof(tensors));
            }
            if (tensors.Count == 1) {
                return tensors[0];
            }

            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_cat(tensorsRef, parray.Array.Length, dimension);
                if (res == IntPtr.Zero)
                    torch.CheckForErrors();
                return new Tensor(res);
            }
        }

        /// <summary>
        /// Roll the tensor along the given dimension(s).
        /// Elements that are shifted beyond the last position are re-introduced at the first position.
        /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
        /// </summary>
        public static Tensor roll(Tensor input, long shifts, long? dims = null) => input.roll(shifts, dims);

        /// <summary>
        /// Roll the tensor along the given dimension(s).
        /// Elements that are shifted beyond the last position are re-introduced at the first position.
        /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
        /// </summary>
        public static Tensor roll(Tensor input, (long, long) shifts, (long, long) dims) => input.roll(shifts, dims);

        /// <summary>
        /// Roll the tensor along the given dimension(s).
        /// Elements that are shifted beyond the last position are re-introduced at the first position.
        /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
        /// </summary>
        public static Tensor roll(Tensor input, (long, long, long) shifts, (long, long, long) dims) => input.roll(shifts, dims);

        /// <summary>
        /// Roll the tensor along the given dimension(s).
        /// Elements that are shifted beyond the last position are re-introduced at the first position.
        /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
        /// </summary>
        public static Tensor roll(Tensor input, long[] shifts, long[] dims = null) => input.roll(shifts, dims);

        /// <summary>
        /// Roll the tensor along the given dimension(s).
        /// Elements that are shifted beyond the last position are re-introduced at the first position.
        /// If a dimension is not specified, the tensor will be flattened before rolling and then restored to the original shape.
        /// </summary>
        public static Tensor roll(Tensor input, ReadOnlySpan<long> shifts, ReadOnlySpan<long> dims = default) => input.roll(shifts, dims);

        /// <summary>
        /// Returns a tensor with all the dimensions of input of size 1 removed. When dim is given, a squeeze operation is done only in the given dimension.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">If given, the input will be squeezed only in this dimension</param>
        /// <returns></returns>
        public static Tensor squeeze(Tensor input, long? dim = null) => input.squeeze(dim);

        /// <summary>
        /// Sorts the elements of the input tensor along a given dimension in ascending order by value.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">The dimension to sort along. If dim is not given, the last dimension of the input is chosen.</param>
        /// <param name="descending">Controls the sorting order (ascending or descending)</param>
        /// <param name="stable">Makes the sorting routine stable, which guarantees that the order of equivalent elements is preserved.</param>
        /// <returns>A named tuple of (values, indices) is returned, where the values are the sorted values and indices are the indices of the elements in the original input tensor.</returns>
        public static (Tensor Values, Tensor Indices) sort(Tensor input, long dim = -1, bool descending = false, bool stable = false) => input.sort(dim, descending, stable);

        /// <summary>
        ///  Returns a new tensor with a dimension of size one inserted at the specified position.
        ///  The returned tensor shares the same underlying data with this tensor.
        /// </summary>
        public static Tensor unsqueeze(Tensor input, long dim) => input.unsqueeze(dim);

        /// <summary>
        ///  Returns a new tensor with a dimension of size one inserted at the specified position.
        ///  The returned tensor shares the same underlying data with this tensor.
        /// </summary>
        public static Tensor unsqueeze_(Tensor input, long dim) => input.unsqueeze_(dim);

        /// <summary>
        /// Return a tensor of elements selected from either x or y, depending on condition.
        /// </summary>
        /// <param name="condition">When true, yield x, otherwise yield y.</param>
        /// <param name="x">Values selected at indices where condition is true</param>
        /// <param name="y">Values selected at indices where condition is false</param>
        /// <returns></returns>
        public static Tensor where(Tensor condition, Tensor x, Tensor y) => x.where(condition, y);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_stack(IntPtr tensor, int len, long dim);

        /// <summary>
        /// Concatenates a sequence of tensors along a new dimension.
        /// </summary>
        /// <returns></returns>
        /// <remarks>All tensors need to be of the same size.</remarks>
        public static Tensor stack(IEnumerable<Tensor> tensors, long dimension = 0)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_stack(tensorsRef, parray.Array.Length, dimension);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        /// <summary>
        ///  Returns a view of the original tensor with its dimensions permuted.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="permutation">The desired ordering of dimensions</param>
        static public Tensor permute(Tensor input, params long[] permutation) => input.permute(permutation);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_hstack(IntPtr tensor, int len);

        /// <summary>
        /// Stack tensors in sequence horizontally (column wise).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public static Tensor hstack(IList<Tensor> tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_hstack(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_vstack(IntPtr tensor, int len);

        /// <summary>
        /// Stack tensors in sequence vertically (row wise).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public static Tensor vstack(IList<Tensor> tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_vstack(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_column_stack(IntPtr tensors, int len);

        /// <summary>
        /// Creates a new tensor by horizontally stacking the input tensors.
        /// </summary>
        /// <param name="tensors">A list of input tensors.</param>
        /// <returns></returns>
        /// <remarks>Equivalent to torch.hstack(tensors), except each zero or one dimensional tensor t in tensors is first reshaped into a (t.numel(), 1) column before being stacked horizontally.</remarks>
        public static Tensor column_stack(IList<Tensor> tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_column_stack(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_row_stack(IntPtr tensor, int len);

        /// <summary>
        /// Stack tensors in sequence vertically (row wise).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public static Tensor row_stack(IList<Tensor> tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_row_stack(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_dstack(IntPtr tensor, int len);

        /// <summary>
        /// Stack tensors in sequence depthwise (along third axis).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        /// <remarks>This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped by torch.atleast_3d().</remarks>
        public static Tensor dstack(IList<Tensor> tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_dstack(tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_meshgrid(IntPtr tensor, long len, [MarshalAs(UnmanagedType.LPStr)] string indexing, AllocatePinnedArray allocator);

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

                var res = THSTensor_meshgrid(tensorsRef, parray.Array.LongLength, indexing, parray.CreateArray);
                torch.CheckForErrors();
                ptrArray = parray.Array;
            }
            return ptrArray.Select(x => new Tensor(x)).ToArray();
        }

        /// <summary>
        /// Returns the k largest elements of the given input tensor along a given dimension.
        /// </summary>
        /// <param name="tensor">The input tensor</param>
        /// <param name="k">The 'k' in 'top-k'.</param>
        /// <param name="dimension">The dimension to sort along. If dim is not given, the last dimension of the input is chosen.</param>
        /// <param name="largest">Controls whether to return largest or smallest elements</param>
        /// <param name="sorted">Controls whether to return the elements in sorted order</param>
        public static (Tensor values, Tensor indexes) topk(Tensor tensor, int k, int dimension = -1, bool largest = true, bool sorted = true) => tensor.topk(k, dimension, largest, sorted);

        /// <summary>
        /// Removes a tensor dimension.
        /// </summary>
        /// <param name="tensor">The input tensor</param>
        /// <param name="dimension">The dimension to remove.</param>
        /// <returns>An array of all slices along a given dimension, already without it.</returns>
        public static Tensor[] unbind(Tensor tensor, int dimension = 0) => tensor.unbind(dimension);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTorch_lstsq(IntPtr input, IntPtr A, out IntPtr pQR);

        /// <summary>
        /// Computes the solution to the least squares and least norm problems for a full rank matrix A of size m×n and a matrix B of size m×k.
        /// </summary>
        /// <param name="A">the m by n matrix AA</param>
        /// <param name="B">the matrix BB</param>
        /// <returns></returns>
        public static (Tensor Solution, Tensor QR) lstsq(Tensor B, Tensor A)
        {
            var solution = THSTorch_lstsq(B.Handle, A.Handle, out var qr);
            if (solution == IntPtr.Zero || qr == IntPtr.Zero)
                torch.CheckForErrors();
            return (new Tensor(solution), new Tensor(qr));
        }

        /// <summary>
        ///  Writes all values from the tensor src into input at the indices specified in the index tensor. For each
        ///  value in src, its output index is specified by its index in src for dimension != dim and by the #
        ///  corresponding value in index for dimension = dim.
        /// </summary>
        public static Tensor scatter(Tensor input, long dimension, Tensor index, Tensor src) => input.scatter(dimension, index, src);

        /// <summary>
        ///  Writes all values from the tensor src into input at the indices specified in the index tensor. For each
        ///  value in src, its output index is specified by its index in src for dimension != dim and by the #
        ///  corresponding value in index for dimension = dim.
        /// </summary>
        public static Tensor scatter_(Tensor input, long dimension, Tensor index, Tensor src) => input.scatter_(dimension, index, src);

        /// <summary>
        /// Adds all values from the tensor other into input at the indices specified in the index tensor in a similar fashion as scatter_().
        /// For each value in src, it is added to an index in self which is specified by its index in src for dimension != dim and by the
        /// corresponding value in index for dimension = dim.
        /// </summary>
        public static Tensor scatter_add(Tensor input, long dimension, Tensor index, Tensor src) => input.scatter_add(dimension, index, src);

        /// <summary>
        /// Adds all values from the tensor other into input at the indices specified in the index tensor in a similar fashion as scatter_().
        /// For each value in src, it is added to an index in self which is specified by its index in src for dimension != dim and by the
        /// corresponding value in index for dimension = dim.
        /// </summary>
        public static Tensor scatter_add_(Tensor input, long dimension, Tensor index, Tensor src) => input.scatter_add_(dimension, index, src);

        /// <summary>
        /// Clamps all elements in input into the range [ min, max ].
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="min">The minimum value</param>
        /// <param name="max">The maximum value</param>
        static public Tensor clamp(Tensor input, Scalar min = null, Scalar max = null) => input.clamp(min, max);

        /// <summary>
        /// Clamps all elements in input into the range [ min, max ] in place.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="min">The minimum value</param>
        /// <param name="max">The maximum value</param>
        static public Tensor clamp_(Tensor input, Scalar min = null, Scalar max = null) => input.clamp_(min, max);

        /// <summary>
        /// Clamps all elements in input into the range [ min, max ].
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="min">The minimum value</param>
        /// <param name="max">The maximum value</param>
        static public Tensor clamp(Tensor input, Tensor min = null, Tensor max = null) => input.clamp(min, max);

        /// <summary>
        /// Clamps all elements in input into the range [ min, max ] in place.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="min">The minimum value</param>
        /// <param name="max">The maximum value</param>
        static public Tensor clamp_(Tensor input, Tensor min = null, Tensor max = null) => input.clamp_(min, max);

        static public Tensor clamp_max(Tensor input, Scalar max) => input.clamp_max(max);

        static public Tensor clamp_max_(Tensor input, Scalar max) => input.clamp_max(max);

        static public Tensor clamp_min(Tensor input, Scalar min) => input.clamp_min(min);

        static public Tensor clamp_min_(Tensor input, Scalar min) => input.clamp_min(min);

        /// <summary>
        /// Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dims">The dimension or dimensions to reduce.</param>
        /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
        /// <param name="out">The output tensor -- optional.</param>
        static public Tensor amax(Tensor input, long[] dims, bool keepDim = false, Tensor @out = null) => input.amax(dims, keepDim, @out);

        /// <summary>
        /// Returns the minimum value of each slice of the input tensor in the given dimension(s) dim.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dims">The dimension or dimensions to reduce.</param>
        /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
        /// <param name="out">The output tensor -- optional.</param>
        static public Tensor amin(Tensor input, long[] dims, bool keepDim = false, Tensor @out = null) => input.amin(dims, keepDim, @out);

        /// <summary>
        /// Returns a tensor with the same data and number of elements as the input but with the specified shape.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="shape">The new tensor shape.</param>
        static public Tensor reshape(Tensor input, params long[] shape) => input.reshape(shape);

        /// <summary>
        /// Flattens input by reshaping it into a one-dimensional tensor.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="start_dim">The first dim to flatten</param>
        /// <param name="end_dim">The last dim to flatten.</param>
        /// <remarks>Flattening a zero-dimensional tensor will return a one-dimensional view.</remarks>
        static public Tensor flatten(Tensor input, long start_dim = 0, long end_dim = -1) => input.flatten(start_dim, end_dim);

        /// <summary>
        /// Expands the dimension dim of the self tensor over multiple dimensions of sizes given by sizes.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">Dimension to unflatten.</param>
        /// <param name="sizes">New shape of the unflattened dimension.</param>
        static public Tensor unflatten(Tensor input, long dim, params long[] sizes) => input.unflatten(dim, sizes);

        /// <summary>
        /// Expands the dimension dim of the self tensor over multiple dimensions of sizes given by sizes.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">Dimension to unflatten.</param>
        /// <param name="sizes">New shape of the unflattened dimension.</param>
        static public Tensor unflatten(Tensor input, long dim, torch.Size sizes) => input.unflatten(dim, sizes.Shape);

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_standard_gamma_(IntPtr tensor, IntPtr gen);

        public static Tensor _standard_gamma(Tensor input, torch.Generator generator = null)
        {
            var res = THSTensor_standard_gamma_(input.Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_sample_dirichlet_(IntPtr tensor, IntPtr gen);

        public static Tensor _sample_dirichlet(Tensor input, torch.Generator generator = null)
        {
            var res = THSTensor_sample_dirichlet_(input.Handle, (generator is null) ? IntPtr.Zero : generator.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(res);
        }


        /// <summary>
        /// Accumulate the elements of alpha times source into the input tensor by adding to the indices in the order given in index.
        /// 
        /// For example, if dim == 0, index[i] == j, and alpha=-1, then the ith row of source is subtracted from the jth row of the input tensor.
        /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match self, or an error will be raised.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">Dimension along which to index</param>
        /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
        /// <param name="source">The tensor containing values to add</param>
        /// <param name="alpha">The scalar multiplier for source</param>
        /// <returns></returns>
        public static Tensor index_add(Tensor input, long dim, Tensor index, Tensor source, Scalar alpha) => input.index_add(dim, index, source, alpha);

        /// <summary>
        /// Accumulate, in place, the elements of alpha times source into the input tensor by adding to the indices in the order given in index.
        /// 
        /// For example, if dim == 0, index[i] == j, and alpha=-1, then the ith row of source is subtracted from the jth row of the input tensor.
        /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match self, or an error will be raised.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">Dimension along which to index</param>
        /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
        /// <param name="source">The tensor containing values to add</param>
        /// <param name="alpha">The scalar multiplier for source</param>
        /// <returns></returns>
        public static Tensor index_add_(Tensor input, long dim, Tensor index, Tensor source, Scalar alpha) => input.index_add_(dim, index, source, alpha);

        /// <summary>
        /// Copies the elements of the source tensor into the input tensor by selecting the indices in the order given in index.
        ///
        /// For example, if dim == 0 and index[i] == j, then the ith row of tensor is copied to the jth row of the input tensor.
        /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match self, or an error will be raised.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">Dimension along which to index</param>
        /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
        /// <param name="source">The tensor containing values to copy</param>
        /// <returns></returns>
        public static Tensor index_copy(Tensor input, long dim, Tensor index, Tensor source) => input.index_copy(dim, index, source);

        /// <summary>
        /// Copies, in place, the elements of the source tensor into the input tensor by selecting the indices in the order given in index.
        ///
        /// For example, if dim == 0 and index[i] == j, then the ith row of tensor is copied to the jth row of the input tensor.
        /// The dimth dimension of source must have the same size as the length of index(which must be a vector), and all other dimensions must match self, or an error will be raised.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dim">Dimension along which to index</param>
        /// <param name="index">Indices of source to select from, should have dtype either torch.int64 or torch.int32</param>
        /// <param name="source">The tensor containing values to copy</param>
        /// <returns></returns>
        public static Tensor index_copy_(Tensor input, long dim, Tensor index, Tensor source) => input.index_copy_(dim, index, source);

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
        /// Repeat elements of a tensor.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="repeats">The number of repeats</param>
        /// <param name="dim">The dimension to repeat</param>
        /// <param name="output_size">The size of output</param>
        /// <returns></returns>
        public static Tensor repeat_interleave(Tensor input, long repeats, long? dim = null, long? output_size = null) => input.repeat_interleave(repeats, dim, output_size);

        /// <summary>
        /// Repeat elements of a tensor.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="repeats">The number of repeats</param>
        /// <param name="dim">The dimension to repeat</param>
        /// <param name="output_size">The size of output</param>
        /// <returns></returns>
        public static Tensor repeat_interleave(Tensor input, Tensor repeats, long? dim = null, long? output_size = null) => input.repeat_interleave(repeats, dim, output_size);

        /// <summary>
        /// Constructs a tensor by repeating the elements of input. The dims argument specifies the number of repetitions in each dimension.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dims">The number of repetitions per dimension.</param>
        public static Tensor tile(Tensor input, long[] dims) => input.tile(dims);

        /// <summary>
        /// Tests if all elements in input evaluate to true.
        /// <param name="input">The input tensor</param>
        /// </summary>
        public static Tensor all(Tensor input) => input.all();

        /// <summary>
        /// Tests if all elements in input evaluate to true.
        /// <param name="input">The input tensor</param>
        /// <param name="dim">The dimension to reduce</param>
        /// <param name="keepdim">Keep the dimension to reduce</param>
        /// </summary>
        public static Tensor all(Tensor input, long dim, bool keepdim = false) => input.all(dim, keepdim);

        /// <summary>
        /// Tests if all elements in input evaluate to true.
        /// <param name="input">The input tensor</param>
        /// </summary>
        public static Tensor any(Tensor input) => input.any();

        /// <summary>
        /// Tests if any element in input evaluate to true.
        /// <param name="input">The input tensor</param>
        /// <param name="dim">The dimension to reduce</param>
        /// <param name="keepdim">Keep the dimension to reduce</param>
        /// </summary>
        public static Tensor any(Tensor input, long dim, bool keepdim = false) => input.any(dim, keepdim);
    }
}
