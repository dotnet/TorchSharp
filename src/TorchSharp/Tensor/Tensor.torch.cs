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

                return new Tensor(THSTensor_cat(tensorsRef, parray.Array.Length, dimension));
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_stack(IntPtr tensor, int len, long dim);

        /// <summary>
        /// Concatenates a sequence of tensors along a new dimension.
        /// </summary>
        /// <returns></returns>
        /// <remarks>All tensors need to be of the same size.</remarks>
        public static Tensor stack(IList<Tensor> tensors, long dimension)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_stack(tensorsRef, parray.Array.Length, dimension);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }

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
    }
}
