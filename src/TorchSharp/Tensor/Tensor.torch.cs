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


        [DllImport("LibTorchSharp")]
        static extern void THSTensor_broadcast_tensors(IntPtr tensor, long length, AllocatePinnedArray allocator);

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
        public static Tensor roll(Tensor input, (long,long) shifts, (long,long) dims) => input.roll(shifts, dims);

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
        /// <param name="input"></param>
        /// <param name="dim">If given, the input will be squeezed only in this dimension</param>
        /// <returns></returns>
        public static Tensor squeeze(Tensor input, long? dim = null) => input.squeeze(dim);


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

        /// <summary>
        /// Creates a new tensor by horizontally stacking the tensors in tensors.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="len"></param>
        /// <returns></returns>
        /// <remarks>Equivalent to torch.hstack(tensors), except each zero or one dimensional tensor t in tensors is first reshaped into a (t.numel(), 1) column before being stacked horizontally.</remarks>
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_column_stack(IntPtr tensor, int len);

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

        /// <summary>
        /// Returns the k largest elements of the given input tensor along a given dimension.
        /// </summary>
        /// <param name="tensor">The input tensor</param>
        /// <param name="k">The 'k' in 'top-k'.</param>
        /// <param name="dimension">The dimension to sort along. If dim is not given, the last dimension of the input is chosen.</param>
        /// <param name="largest">Controls whether to return largest or smallest elements</param>
        /// <param name="sorted">Controls whether to return the elements in sorted order</param>
        /// <returns></returns>
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

        static public Tensor clamp(Tensor input, Scalar min = null, Scalar max = null) => input.clamp(min, max);

        static public Tensor clamp_(Tensor input, Scalar min = null, Scalar max = null) => input.clamp_(min, max);

        static public Tensor clamp(Tensor input, Tensor min = null, Tensor max = null) => input.clamp(min, max);

        static public Tensor clamp_(Tensor input, Tensor min = null, Tensor max = null) => input.clamp_(min, max);

        static public Tensor clamp_max(Tensor input, Scalar max) => input.clamp_max(max);

        static public Tensor clamp_max_(Tensor input, Scalar max) => input.clamp_max(max);

        static public Tensor clamp_min(Tensor input, Scalar min) => input.clamp_min(min);

        static public Tensor clamp_min_(Tensor input, Scalar min) => input.clamp_min(min);

        static public Tensor amax(Tensor input, long[] dims, bool keepDim = false, Tensor @out = null) => input.amax(dims, keepDim, @out);

        static public Tensor amin(Tensor input, long[] dims, bool keepDim = false, Tensor @out = null) => input.amin(dims, keepDim, @out);

        static public Tensor reshape(Tensor input, params long[] shape) => input.reshape(shape);

        static public Tensor flatten(Tensor input, long start_dim = 0, long end_dim = -1) => input.flatten(start_dim, end_dim);

        static public Tensor unflatten(Tensor input, long dim, params long[] sizes) => input.unflatten(dim, sizes);

        static public Tensor unflatten(Tensor input, long dim, torch.Size sizes) => input.unflatten(dim, sizes.Shape);

        /// <summary>
        /// Return a tensor of elements selected from either x or y, depending on condition.
        /// </summary>
        /// <param name="condition">When true, yield x, otherwise yield y.</param>
        /// <param name="x">Values selected at indices where condition is true</param>
        /// <param name="y">Values selected at indices where condition is false</param>
        /// <returns></returns>
        static public Tensor where(Tensor condition, Tensor x, Tensor y) => x.where(condition, y);


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
    }
}
