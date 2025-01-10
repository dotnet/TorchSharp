// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#indexing-slicing-joining-mutating-ops
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.adjoint
        /// <summary>
        /// Returns a view of the tensor conjugated and with the last two dimensions transposed.
        /// </summary>
        /// <param name="input">The input tensor</param>
        public static Tensor adjoint(Tensor input) => input.adjoint();

        // https://pytorch.org/docs/stable/generated/torch.argwhere
        /// <summary>
        /// Returns a tensor containing the indices of all non-zero elements of input.
        /// Each row in the result contains the indices of a non-zero element in input.
        /// The result is sorted lexicographically, with the last index changing the fastest (C-style).
        /// If input has n dimensions, then the resulting indices tensor out is of size (z×n), where
        /// z is the total number of non-zero elements in the input tensor.
        /// </summary>
        public static Tensor argwhere(Tensor input) => input.argwhere();

        // https://pytorch.org/docs/stable/generated/torch.cat
        /// <summary>
        /// Concatenates the given sequence of tensors in the given dimension.
        /// </summary>
        /// <param name="tensors">A sequence of tensors of the same type. Non-empty tensors provided must have the same shape, except in the cat dimension.</param>
        /// <param name="dim">The dimension over which the tensors are concatenated</param>
        /// <remarks> All tensors must either have the same shape (except in the concatenating dimension) or be empty.</remarks>
        public static Tensor cat(IList<Tensor> tensors, long dim = 0)
        {
            switch (tensors.Count)
            {
                case <=0: throw new ArgumentException(nameof(tensors));
                case 1: return tensors[0].alias();
            }

            using var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            var res = THSTensor_cat(tensorsRef, parray.Array.Length, dim);
            if (res == IntPtr.Zero) CheckForErrors();
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.concat
        /// <summary>
        /// Alias of torch.cat()
        /// </summary>
        /// <param name="tensors">A sequence of tensors of the same type. Non-empty tensors provided must have the same shape, except in the cat dimension.</param>
        /// <param name="dim">The dimension over which the tensors are concatenated</param>
        /// <remarks> All tensors must either have the same shape (except in the concatenating dimension) or be empty.</remarks>
        public static Tensor concat(IList<Tensor> tensors, long dim = 0) => torch.cat(tensors, dim);

        // https://pytorch.org/docs/stable/generated/torch.conj
        /// <summary>
        /// Returns a view of input with a flipped conjugate bit. If input has a non-complex dtype, this function just returns input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        public static Tensor conj(Tensor input) => input.conj();

        // https://pytorch.org/docs/stable/generated/torch.chunk
        public static Tensor[] chunk(Tensor input, long chunks, long dim = 0)
            => input.chunk(chunks, dim);

        // https://pytorch.org/docs/stable/generated/torch.dsplit
        public static Tensor[] dsplit(Tensor input, Tensor indices_or_sections)
            => input.dsplit(indices_or_sections);

        // https://pytorch.org/docs/stable/generated/torch.dsplit
        public static Tensor[] dsplit(Tensor input, long size)
            => input.dsplit(size);

        // https://pytorch.org/docs/stable/generated/torch.dsplit
        public static Tensor[] dsplit(Tensor input, long[] indices_or_sections)
            => input.dsplit(indices_or_sections);

        // https://pytorch.org/docs/stable/generated/torch.dsplit
        public static Tensor[] dsplit(Tensor input, (long, long) indices_or_sections)
            => input.dsplit(indices_or_sections);

        // https://pytorch.org/docs/stable/generated/torch.dsplit
        public static Tensor[] dsplit(Tensor input, (long, long, long) indices_or_sections)
            => input.dsplit(indices_or_sections);

        // https://pytorch.org/docs/stable/generated/torch.dsplit
        public static Tensor[] dsplit(Tensor input, (long, long, long, long) indices_or_sections)
            => input.dsplit(indices_or_sections);

        // https://pytorch.org/docs/stable/generated/torch.dstack
        /// <summary>
        /// Stack tensors in sequence depthwise (along third axis).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        /// <remarks>This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped by torch.atleast_3d().</remarks>
        public static Tensor dstack(params Tensor[] tensors)
            => dstack((IEnumerable<Tensor>)tensors);

        // https://pytorch.org/docs/stable/generated/torch.dstack
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
        /// Stack tensors in sequence depthwise (along third axis).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        /// <remarks>This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped by torch.atleast_3d().</remarks>
        public static Tensor dstack(IEnumerable<Tensor> tensors)
        {
            using var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());
            var res = THSTensor_dstack(tensorsRef, parray.Array.Length);
            if (res == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(res);
        }
        
        // https://pytorch.org/docs/stable/generated/torch.gather
        /// <summary>
        /// Gathers values along an axis specified by dim.
        /// </summary>
        public static Tensor gather(Tensor input, long dim, Tensor index) => input.gather(dim, index);

        // https://pytorch.org/docs/stable/generated/torch.gather
        // TODO: implement parameter sparse_grad
        public static Tensor gather(Tensor input, long dim, Tensor index, bool sparse_grad=false)
            => input.gather(dim, index);

        // https://pytorch.org/docs/stable/generated/torch.hsplit
        public static Tensor[] hsplit(Tensor input, Tensor indices_or_sections)
            => input.hsplit(indices_or_sections);

        // https://pytorch.org/docs/stable/generated/torch.hsplit
        public static Tensor[] hsplit(Tensor input, long indices_or_sections)
            => input.hsplit(indices_or_sections);

        // https://pytorch.org/docs/stable/generated/torch.hsplit
        public static Tensor[] hsplit(Tensor input, long[] indices_or_sections)
            => input.hsplit(indices_or_sections);

        // https://pytorch.org/docs/stable/generated/torch.hsplit
        public static Tensor[] hsplit(Tensor input, (long, long) indices_or_sections)
            => input.hsplit(new[]{
                indices_or_sections.Item1,
                indices_or_sections.Item2
            });

        // https://pytorch.org/docs/stable/generated/torch.hsplit
        public static Tensor[] hsplit(Tensor input, (long, long, long) indices_or_sections)
            => input.hsplit(new[]{
                indices_or_sections.Item1,
                indices_or_sections.Item2,
                indices_or_sections.Item3
            });

        // https://pytorch.org/docs/stable/generated/torch.hsplit
        public static Tensor[] hsplit(Tensor input, (long, long, long, long) indices_or_sections)
            => input.hsplit(new[]{
                indices_or_sections.Item1,
                indices_or_sections.Item2,
                indices_or_sections.Item3,
                indices_or_sections.Item4
            });

        // https://pytorch.org/docs/stable/generated/torch.hstack
        /// <summary>
        /// Stack tensors in sequence horizontally (column wise).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public static Tensor hstack(IList<Tensor> tensors)
        {
            using var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            var res = THSTensor_hstack(tensorsRef, parray.Array.Length);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.hstack
        /// <summary>
        /// Stack tensors in sequence horizontally (column wise).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public static Tensor hstack(params Tensor[] tensors)
        {
            return hstack((IEnumerable<Tensor>)tensors);
        }

        // https://pytorch.org/docs/stable/generated/torch.hstack
        /// <summary>
        /// Stack tensors in sequence horizontally (column wise).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public static Tensor hstack(IEnumerable<Tensor> tensors)
        {
            using var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            var res = THSTensor_hstack(tensorsRef, parray.Array.Length);
            if (res == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.index_add
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
        public static Tensor index_add(Tensor input, long dim, Tensor index, Tensor source, Scalar alpha)
            => input.index_add(dim, index, source, alpha);

        // https://pytorch.org/docs/stable/generated/torch.index_add
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
        public static Tensor index_add_(Tensor input, long dim, Tensor index, Tensor source, Scalar alpha)
            => input.index_add_(dim, index, source, alpha);

        // https://pytorch.org/docs/stable/generated/torch.index_copy
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
        public static Tensor index_copy(Tensor input, long dim, Tensor index, Tensor source)
            => input.index_copy(dim, index, source);

        // https://pytorch.org/docs/stable/generated/torch.index_copy
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
        public static Tensor index_copy_(Tensor input, long dim, Tensor index, Tensor source)
            => input.index_copy_(dim, index, source);

        // https://pytorch.org/docs/stable/generated/torch.index_reduce
        [Obsolete("not implemented", true)]
        public static Tensor index_reduce(Tensor input, long dim, Tensor index, Tensor source, Reduce reduce, bool include_self=true)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.index_select
        /// <summary>
        /// Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
        /// </summary>
        public static Tensor index_select(Tensor input, long dim, Tensor index)
            => input.index_select(dim, index);

        // https://pytorch.org/docs/stable/generated/torch.masked_select
        public static Tensor masked_select(Tensor input, Tensor mask)
            => input.masked_select(mask);

        // https://pytorch.org/docs/stable/generated/torch.movedim
        public static Tensor movedim(Tensor input, long source, long destination)
            => input.movedim(new[]{source}, new[]{destination});

        // https://pytorch.org/docs/stable/generated/torch.movedim
        static Tensor movedim(Tensor input, (long, long) source, (long, long) destination)
            => input.movedim(
                new[]{source.Item1, source.Item2},
                new[]{destination.Item1, destination.Item2});

        // https://pytorch.org/docs/stable/generated/torch.movedim
        static Tensor movedim(Tensor input, (long, long, long) source, (long, long, long) destination)
            => input.movedim(
                new[]{source.Item1, source.Item2, source.Item3},
                new[]{destination.Item1, destination.Item2, destination.Item3});

        // https://pytorch.org/docs/stable/generated/torch.movedim
        static Tensor movedim(Tensor input, (long, long, long, long) source, (long, long, long, long) destination)
            => input.movedim(
                new[]{source.Item1, source.Item2, source.Item3, source.Item4},
                new[]{destination.Item1, destination.Item2, destination.Item3, destination.Item4});

        // https://pytorch.org/docs/stable/generated/torch.movedim
        public static Tensor movedim(Tensor input, long[] source, long[] destination)
            => input.movedim(source, destination);

        // https://pytorch.org/docs/stable/generated/torch.moveaxis
        public static Tensor moveaxis(Tensor input, long source, long destination)
            => input.moveaxis(new[]{source}, new[]{destination});

        // https://pytorch.org/docs/stable/generated/torch.moveaxis
        public static Tensor moveaxis(Tensor input, (long, long) source, (long, long) destination)
            => input.moveaxis(
                new[]{source.Item1, source.Item2 },
                new[]{ destination.Item1, destination.Item2 });

        // https://pytorch.org/docs/stable/generated/torch.moveaxis
        public static Tensor moveaxis(Tensor input, (long, long, long) source, (long, long, long) destination)
            => input.moveaxis(
                new[]{source.Item1, source.Item2, source.Item3 },
                new[]{ destination.Item1, destination.Item2, destination.Item3 });

        public static Tensor moveaxis(Tensor input, (long, long, long, long) source, (long, long, long, long) destination)
            => input.moveaxis(
                new[]{source.Item1, source.Item2, source.Item3, source.Item4 },
                new[]{ destination.Item1, destination.Item2, destination.Item3, destination.Item4 });

        public static Tensor moveaxis(Tensor input, long[] source, long[] destination)
            => input.moveaxis(source, destination);

        // https://pytorch.org/docs/stable/generated/torch.narrow
        public static Tensor narrow(Tensor input, long dim, long start, long length)
            => input.narrow(dim, start, length);

        // https://pytorch.org/docs/stable/generated/torch.nonzero
        /// <summary>
        /// Returns a tensor containing the indices of all non-zero elements of input.
        /// Each row in the result contains the indices of a non-zero element in input.
        /// The result is sorted lexicographically, with the last index changing the fastest (C-style).
        /// </summary>
        public static Tensor nonzero(Tensor input) => input.nonzero();

        // https://pytorch.org/docs/stable/generated/torch.permute
        /// <summary>
        ///  Returns a view of the original tensor with its dimensions permuted.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="permutation">The desired ordering of dimensions</param>
        public static Tensor permute(Tensor input, params long[] permutation) => input.permute(permutation);

        // https://pytorch.org/docs/stable/generated/torch.reshape
        /// <summary>
        /// Returns a tensor with the same data and number of elements as the input but with the specified shape.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="shape">The new tensor shape.</param>
        public static Tensor reshape(Tensor input, params long[] shape) => input.reshape(shape);

        // https://pytorch.org/docs/stable/generated/torch.select
        public static Tensor select(Tensor input, long dim, long index)
            => input.select(dim, index);

        // https://pytorch.org/docs/stable/generated/torch.scatter
        /// <summary>
        ///  Writes all values from the tensor src into input at the indices specified in the index tensor. For each
        ///  value in src, its output index is specified by its index in src for dimension != dim and by the #
        ///  corresponding value in index for dimension = dim.
        /// </summary>
        public static Tensor scatter(Tensor input, long dim, Tensor index, Tensor src)
            => input.scatter(dim, index, src);

        // https://pytorch.org/docs/stable/generated/torch.scatter
        /// <summary>
        ///  Writes all values from the tensor src into input at the indices specified in the index tensor. For each
        ///  value in src, its output index is specified by its index in src for dimension != dim and by the #
        ///  corresponding value in index for dimension = dim.
        /// </summary>
        public static Tensor scatter_(Tensor input, long dim, Tensor index, Tensor src)
            => input.scatter_(dim, index, src);

        // https://pytorch.org/docs/stable/generated/torch.diagonal_scatter
        /// <summary>
        /// Embeds the values of the src tensor into input along the diagonal elements of input, with respect to dim1 and dim2.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="src">The tensor to embed into 'this'.</param>
        /// <param name="offset">Which diagonal to consider. Default: main diagonal.</param>
        /// <param name="dim1">First dimension with respect to which to take diagonal.</param>
        /// <param name="dim2">Second dimension with respect to which to take diagonal.</param>
        public static Tensor diagonal_scatter(Tensor input, Tensor src, long offset = 0L, long dim1 = 0L, long dim2 = 1L) => input.diagonal_scatter(src, offset, dim1, dim2);

        // https://pytorch.org/docs/stable/generated/torch.select_scatter
        /// <summary>
        /// Embeds the values of the src tensor into input at the given index. This function returns a tensor with fresh storage; it does not create a view.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="src">The tensor to embed into 'this'</param>
        /// <param name="dim">The dimension to insert the slice into</param>
        /// <param name="index">The index to select with</param>
        /// <remarks>This function returns a tensor with fresh storage; it does not create a view.</remarks>
        public static Tensor select_scatter(Tensor input, Tensor src, long dim, long index) => input.select_scatter(src, dim, index);

        // https://pytorch.org/docs/stable/generated/torch.slice_scatter
        /// <summary>
        /// Embeds the values of the src tensor into input at the given dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="src">The tensor to embed into 'this'.</param>
        /// <param name="dim">The dimension to insert the slice into</param>
        /// <param name="start">The start index of where to insert the slice</param>
        /// <param name="end">The end index of where to insert the slice</param>
        /// <param name="step">How many elements to skip</param>
        public static Tensor slice_scatter(Tensor input, Tensor src, long dim = 0L, long? start = null, long? end = null, long step = 1L)
            => input.slice_scatter(src, dim, start, end, step);

        // https://pytorch.org/docs/stable/generated/torch.scatter_add
        /// <summary>
        /// Adds all values from the tensor other into input at the indices specified in the index tensor in a similar fashion as scatter_().
        /// For each value in src, it is added to an index in self which is specified by its index in src for dimension != dim and by the
        /// corresponding value in index for dimension = dim.
        /// </summary>
        public static Tensor scatter_add(Tensor input, long dim, Tensor index, Tensor src)
            => input.scatter_add(dim, index, src);

        // https://pytorch.org/docs/stable/generated/torch.scatter_reduce
        [Obsolete("not implemented", true)]
        static Tensor scatter_reduce(
            Tensor input,
            long dim,
            Tensor index,
            Tensor src,
            Reduce reduce,
            bool include_self = true)
            => throw new NotImplementedException();

        // https://pytorch.org/docs/stable/generated/torch.split
        public static Tensor[] split(Tensor tensor, long[] split_size_or_sections, long dim = 0L)
            => tensor.split(split_size_or_sections, dim);

        // https://pytorch.org/docs/stable/generated/torch.stack
        /// <summary>
        /// Concatenates a sequence of tensors along a new dimension.
        /// </summary>
        /// <returns></returns>
        /// <remarks>All tensors need to be of the same size.</remarks>
        public static Tensor stack(IEnumerable<Tensor> tensors, long dim = 0)
        {
            using var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            var res = THSTensor_stack(tensorsRef, parray.Array.Length, dim);
            if (res == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.swapaxes
        public static Tensor swapaxes(Tensor input, long axis0, long axis1)
            => input.swapaxes(axis0, axis1);

        // https://pytorch.org/docs/stable/generated/torch.swapdims
        public static Tensor swapdims(Tensor input, long dim0, long dim1)
            => input.swapdims(dim0, dim1);

        // https://pytorch.org/docs/stable/generated/torch.t
        public static Tensor t(Tensor input)
            => input.t();

        // https://pytorch.org/docs/stable/generated/torch.take
        public static Tensor take(Tensor input, Tensor index)
            => input.take(index);

        // https://pytorch.org/docs/stable/generated/torch.take_along_dim
        public static Tensor take_along_dim(Tensor input, Tensor indices, long dim = 0L)
            => input.take_along_dim(indices, dim);

        // https://pytorch.org/docs/stable/generated/torch.take_along_dim
        public static Tensor take_along_dim(Tensor input, IEnumerable<long> indices, long dim = 0L)
            => input.take_along_dim(indices, dim);

        // https://pytorch.org/docs/stable/generated/torch.tensor_split
        public static Tensor[] tensor_split(Tensor input, long indices_or_sections, long dim = 0L)
            => input.tensor_split(indices_or_sections, dim);

        // https://pytorch.org/docs/stable/generated/torch.tensor_split
        public static Tensor[] tensor_split(Tensor input, long[] indices_or_sections, long dim = 0L)
            => input.tensor_split(indices_or_sections, dim);

        // https://pytorch.org/docs/stable/generated/torch.tensor_split
        public static Tensor[] tensor_split(Tensor input, Tensor indices_or_sections, long dim = 0L)
            => input.tensor_split(indices_or_sections, dim);

        // https://pytorch.org/docs/stable/generated/torch.tile
        /// <summary>
        /// Constructs a tensor by repeating the elements of input. The dims argument specifies the number of repetitions in each dimension.
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="dims">The number of repetitions per dimension.</param>
        public static Tensor tile(Tensor input, long[] dims) => input.tile(dims);

        // https://pytorch.org/docs/stable/generated/torch.transpose
        public static Tensor transpose(Tensor input, long dim0, long dim1)
            => input.transpose(dim0, dim1);

        // https://pytorch.org/docs/stable/generated/torch.unbind
        public static Tensor[] unbind(Tensor input, long dim = 0L)
            => input.unbind(dim);

        // https://pytorch.org/docs/stable/generated/torch.unsqueeze
        /// <summary>
        ///  Returns a new tensor with a dimension of size one inserted at the specified position.
        ///  The returned tensor shares the same underlying data with this tensor.
        /// </summary>
        public static Tensor unsqueeze(Tensor input, long dim)
            => input.unsqueeze(dim);

        // https://pytorch.org/docs/stable/generated/torch.unsqueeze
        /// <summary>
        ///  Returns a new tensor with a dimension of size one inserted at the specified position.
        ///  The returned tensor shares the same underlying data with this tensor.
        /// </summary>
        public static Tensor unsqueeze_(Tensor input, long dim)
            => input.unsqueeze_(dim);

        // https://pytorch.org/docs/stable/generated/torch.vsplit
        public static Tensor[] vsplit(Tensor input, long[] indices_or_sections)
            => input.vsplit(indices_or_sections);

        // https://pytorch.org/docs/stable/generated/torch.vstack
        /// <summary>
        /// Stack tensors in sequence vertically (row wise).
        /// </summary>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public static Tensor vstack(IList<Tensor> tensors)
        {
            using var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            var res = THSTensor_vstack(tensorsRef, parray.Array.Length);
            if (res == IntPtr.Zero) { CheckForErrors(); }
            return new Tensor(res);
        }

        // https://pytorch.org/docs/stable/generated/torch.where
        /// <summary>
        /// Return a tensor of elements selected from either x or y, depending on condition.
        /// </summary>
        /// <param name="condition">When true, yield x, otherwise yield y.</param>
        /// <param name="x">Values selected at indices where condition is true</param>
        /// <param name="y">Values selected at indices where condition is false</param>
        /// <returns></returns>
        public static Tensor where(Tensor condition, Tensor x, Tensor y) => x.where(condition, y);

        // https://pytorch.org/docs/stable/generated/torch.where
        /// <summary>
        /// Returns a tuple of 1-D tensors, one for each dimension in input, each containing the indices (in that dimension) of all non-zero elements of input .
        /// If input has nn dimensions, then the resulting tuple contains nn tensors of size zz, where zz is the total number of non-zero elements in the input tensor.
        /// As a special case, when input has zero dimensions and a nonzero scalar value, it is treated as a one-dimensional tensor with one element.
        /// </summary>
        /// <param name="condition">The input tensor</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static Tensor[] where(Tensor condition)
        {
            if (condition.dtype != ScalarType.Bool) throw new ArgumentException("The condition to 'where' must be a boolean tensor.");

            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_where_list(condition.Handle, pa.CreateArray);
                CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new Tensor(x)).ToArray();
        }
    }
}