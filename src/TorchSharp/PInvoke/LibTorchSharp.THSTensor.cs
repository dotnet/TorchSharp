// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
using System;
using System.Runtime.InteropServices;
using TorchSharp.Modules;

namespace TorchSharp.PInvoke
{
    internal static partial class LibTorchSharp
    {
        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_where_list(IntPtr condition, AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_squeeze_(IntPtr tensor, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_squeeze_no_dim_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_conv1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_conv2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_conv3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_conv_transpose1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_conv_transpose2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_conv_transpose3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_max_pool1d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                [MarshalAs(UnmanagedType.U1)] bool ceil_mode);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_max_pool1d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                [MarshalAs(UnmanagedType.U1)] bool ceil_mode);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_max_pool2d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                [MarshalAs(UnmanagedType.U1)] bool ceil_mode);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_max_pool2d_with_indices(IntPtr input, AllocatePinnedArray allocator,
            IntPtr kernelSize, int kernelSizeLength,
            IntPtr strides, int stridesLength,
            IntPtr padding, int paddingLength,
            IntPtr dilation, int dilationLength,
            [MarshalAs(UnmanagedType.U1)] bool ceil_mode);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_max_pool3d(IntPtr input,
            IntPtr kernelSize, int kernelSizeLength,
            IntPtr strides, int stridesLength,
            IntPtr padding, int paddingLength,
            IntPtr dilation, int dilationLength,
            [MarshalAs(UnmanagedType.U1)] bool ceil_mode);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_max_pool3d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                [MarshalAs(UnmanagedType.U1)] bool ceil_mode);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_maxunpool3d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength, IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_avg_pool1d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                [MarshalAs(UnmanagedType.U1)] bool ceil_mode,
                [MarshalAs(UnmanagedType.U1)] bool count_include_pad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_avg_pool2d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                [MarshalAs(UnmanagedType.U1)] bool ceil_mode,
                [MarshalAs(UnmanagedType.U1)] bool count_include_pad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_avg_pool3d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                [MarshalAs(UnmanagedType.U1)] bool ceil_mode,
                [MarshalAs(UnmanagedType.U1)] bool count_include_pad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_avg_pool2d_backward(IntPtr gradOutput, IntPtr originalInput,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                [MarshalAs(UnmanagedType.U1)] bool ceil_mode,
                [MarshalAs(UnmanagedType.U1)] bool count_include_pad,
                long divisorOverride);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_avg_pool3d_backward(IntPtr gradOutput, IntPtr originalInput,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                [MarshalAs(UnmanagedType.U1)] bool ceil_mode,
                [MarshalAs(UnmanagedType.U1)] bool count_include_pad,
                long divisorOverride);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_adaptive_avg_pool1d(IntPtr input,
                IntPtr outputSize, int outputSizeLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_adaptive_avg_pool2d(IntPtr input,
                IntPtr outputSize, int outputSizeLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_upsample_nearest1d(IntPtr input,
                IntPtr outputSize, int outputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_upsample_nearest1d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_upsample_nearest2d(IntPtr input,
                IntPtr outputSize, int outputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_upsample_nearest2d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_upsample_nearest3d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_upsample_nearest3d(IntPtr input,
            IntPtr outputSize, int outputSizeLength,
            IntPtr scaleFactors, int scaleFactorsLength);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string THSTensor_device_str(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_dispose(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_free(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_ndimension(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_element_size(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_numel(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_is_leaf(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_alias(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_storage_offset(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_data(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_real(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_imag(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern float THSTensor_data_idx_float16(IntPtr handle, long i);

        [DllImport("LibTorchSharp")]
        internal static extern float THSTensor_data_idx_bfloat16(IntPtr handle, long i);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_item(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fill_(IntPtr handle, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern sbyte THSTensor_type(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern int THSTensor_device_index(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern int THSTensor_device_type(IntPtr handle);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSTensor_is_sparse(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_load([MarshalAs(UnmanagedType.LPStr)] string location);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_save(IntPtr tensor, [MarshalAs(UnmanagedType.LPStr)] string location);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSTensor_requires_grad(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_set_requires_grad(IntPtr handle, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_retain_grad(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern int THSTensor_result_type(IntPtr tensor1, IntPtr tensor2);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSTensor_is_cpu(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cpu(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cuda(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_to_device(IntPtr handle, int device_type, int device_index, [MarshalAs(UnmanagedType.U1)] bool copy);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_to_type(IntPtr handle, sbyte scalar_type, [MarshalAs(UnmanagedType.U1)] bool copy);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_to_type_and_device(IntPtr handle, sbyte scalar_type, int device_type, int device_index, [MarshalAs(UnmanagedType.U1)] bool copy);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_set_(IntPtr tensor, IntPtr source);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_size(IntPtr handle, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_sizes(IntPtr handle, AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSTensor_has_names(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_names(IntPtr handle, AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rename(IntPtr tensor, IntPtr names, long nLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rename_(IntPtr tensor, IntPtr names, long nLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_refine_names(IntPtr tensor, IntPtr names, long nLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_indices(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_values(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_vander(IntPtr handle, long N, [MarshalAs(UnmanagedType.U1)] bool increasing);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_stride(IntPtr handle, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_strides(IntPtr handle, AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_as_strided(IntPtr input, IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, long storageOffset);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_backward(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_to_dense(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_clone(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_combinations(IntPtr handle, int r, [MarshalAs(UnmanagedType.U1)] bool with_replacement);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_copy_(IntPtr handle, IntPtr source, [MarshalAs(UnmanagedType.U1)] bool non_blocking);

        [DllImport("LibTorchSharp")]
        internal static extern int THSTensor_is_contiguous(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_contiguous(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_is_pinned(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_pin_memory(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_grad(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_index(IntPtr tensor, IntPtr indexStarts, IntPtr indexEnds, IntPtr indexSteps, IntPtr indexTensors, int indicesLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_index_put_scalar_(IntPtr tensor, IntPtr indexStarts, IntPtr indexEnds, IntPtr indexSteps, IntPtr indexTensors, int indicesLength, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_index_put_(IntPtr tensor, IntPtr indexStarts, IntPtr indexEnds, IntPtr indexSteps, IntPtr indexTensors, int indicesLength, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_get1(IntPtr handle, long i1);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_set1(IntPtr handle, long i1, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_get2(IntPtr handle, long i1, long i2);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_set2(IntPtr handle, long i1, long i2, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_get3(IntPtr handle, long i1, long i2, long i3);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_set3(IntPtr handle, long i1, long i2, long i3, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_get4(IntPtr handle, long i1, long i2, long i3, long i4);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_set4(IntPtr handle, long i1, long i2, long i3, long i4, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_get5(IntPtr handle, long i1, long i2, long i3, long i4, long i5);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_set5(IntPtr handle, long i1, long i2, long i3, long i4, long i5, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_get6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_set6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_index_select(IntPtr tensor, long dim, IntPtr index);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_select(IntPtr tensor, long dim, long index);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_adjoint(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_argwhere(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_take(IntPtr tensor, IntPtr index);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_take_along_dim_dflt(IntPtr tensor, IntPtr indices);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_take_along_dim(IntPtr tensor, IntPtr indices, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_index_add(IntPtr tensor, long dim, IntPtr index, IntPtr source, IntPtr alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_index_add_(IntPtr tensor, long dim, IntPtr index, IntPtr source, IntPtr alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_index_copy(IntPtr tensor, long dim, IntPtr index, IntPtr source);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_index_copy_(IntPtr tensor, long dim, IntPtr index, IntPtr source);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_index_fill(IntPtr tensor, long dim, IntPtr index, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_index_fill_(IntPtr tensor, long dim, IntPtr index, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_reshape(IntPtr tensor, IntPtr shape, int length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_flatten(IntPtr tensor, long start, long end);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_flatten_names(IntPtr tensor, IntPtr names, long nLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_unflatten(IntPtr tensor, long dim, IntPtr shape, int length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_unflatten_names(IntPtr tensor, IntPtr names, IntPtr dims, int length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_align_to(IntPtr tensor, IntPtr names, long nLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_unique(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool sorted, [MarshalAs(UnmanagedType.U1)] bool return_inverse, [MarshalAs(UnmanagedType.U1)] bool return_counts, out IntPtr inverse_indices, out IntPtr counts);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_unique_dim(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool sorted, [MarshalAs(UnmanagedType.U1)] bool return_inverse, [MarshalAs(UnmanagedType.U1)] bool return_counts, out IntPtr inverse_indices, out IntPtr counts);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_unique_consecutive(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool return_inverse, [MarshalAs(UnmanagedType.U1)] bool return_counts, out IntPtr inverse_indices, out IntPtr counts);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_unique_dim_consecutive(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool return_inverse, [MarshalAs(UnmanagedType.U1)] bool return_counts, out IntPtr inverse_indices, out IntPtr counts);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_squeeze(IntPtr tensor, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_squeeze_no_dim(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_t(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_H(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mT(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mH(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_transpose(IntPtr tensor, long dim1, long dim2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_tril(IntPtr tensor, long diagonal);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_tril_indices(long row, long col, long offset, sbyte scalar_type, int device_type, int device_index);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_triu(IntPtr tensor, long diagonal);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_triu_indices(long row, long col, long offset, sbyte scalar_type, int device_type, int device_index);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_transpose_(IntPtr tensor, long dim1, long dim2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_view(IntPtr tensor, IntPtr shape, int length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_view_as_complex(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_view_as_real(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_all(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_all_along_dimension(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool keep_dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_amax(IntPtr tensor, IntPtr dim, int dim_len, [MarshalAs(UnmanagedType.U1)] bool keep_dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_amax_out(IntPtr tensor, IntPtr dim, int dim_len, [MarshalAs(UnmanagedType.U1)] bool keep_dim, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_amin(IntPtr tensor, IntPtr dim, int dim_len, [MarshalAs(UnmanagedType.U1)] bool keep_dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_amin_out(IntPtr tensor, IntPtr dim, int dim_len, [MarshalAs(UnmanagedType.U1)] bool keep_dim, IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_aminmax(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool keep_dim, out IntPtr max);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_any(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_any_along_dimension(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool keep_dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_argmax(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_argmax_along_dimension(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool keep_dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_argmin(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_argmin_along_dimension(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool keep_dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_argsort(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool descending);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_deg2rad(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rad2deg(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_copysign(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_count_nonzero(IntPtr tensor, IntPtr dim, int dim_len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cov(IntPtr tensor, long correction, IntPtr fweights, IntPtr aweights);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_corrcoef(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_tile(IntPtr tensor, IntPtr reps, int reps_len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_digamma(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_digamma_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lgamma(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lgamma_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mvlgamma(IntPtr tensor, long p);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mvlgamma_(IntPtr tensor, long p);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_polygamma(IntPtr tensor, long p);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_polygamma_(IntPtr tensor, long p);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_positive(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_softplus(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ravel(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_relu(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_relu_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_relu6(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_relu6_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_celu(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_celu_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_elu(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_elu_(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_gelu(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hardsigmoid(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hardsigmoid_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hardswish(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hardswish_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hardtanh(IntPtr tensor, IntPtr min, IntPtr max);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hardtanh_(IntPtr tensor, IntPtr min, IntPtr max);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_heaviside(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_igamma(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_igammac(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_i0(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_isclose(IntPtr tensor, IntPtr other, double rtol, double atol, [MarshalAs(UnmanagedType.U1)] bool nanEqual);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_isin(IntPtr elements, IntPtr test_elements, [MarshalAs(UnmanagedType.U1)] bool assume_unique, [MarshalAs(UnmanagedType.U1)] bool invert);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_isinf(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_isfinite(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_isposinf(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_isneginf(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_isnan(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_is_nonzero(IntPtr handle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_isreal(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_leaky_relu(IntPtr tensor, IntPtr negative_slope);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_leaky_relu_(IntPtr tensor, IntPtr negative_slope);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_selu(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_selu_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_silu(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_silu_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_log_sigmoid(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lerp(IntPtr tensor, IntPtr end, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lerp_(IntPtr tensor, IntPtr end, IntPtr weight);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta, float alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bmm(IntPtr batch1, IntPtr batch2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cdist(IntPtr x1, IntPtr x2, double p, long compute_mode);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bucketize(IntPtr input, IntPtr boundaries, [MarshalAs(UnmanagedType.U1)] bool out_int32, [MarshalAs(UnmanagedType.U1)] bool right);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bincount(IntPtr tensor, IntPtr weights, long minlength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_channel_shuffle(IntPtr input, long groups);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_clamp(IntPtr input, IntPtr min, IntPtr max);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_clamp_tensor(IntPtr input, IntPtr min, IntPtr max);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_clamp_(IntPtr input, IntPtr min, IntPtr max);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_clamp_tensor_(IntPtr input, IntPtr min, IntPtr max);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_clamp_max(IntPtr input, IntPtr max);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_clamp_max_(IntPtr input, IntPtr max);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_clamp_min(IntPtr input, IntPtr min);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_clamp_min_(IntPtr input, IntPtr min);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_diff(IntPtr tensor, long n, long dim, IntPtr prepend, IntPtr append);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_diag(IntPtr tensor, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_trace(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_diag_embed(IntPtr tensor, long offset, long dim1, long dim2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_diagflat(IntPtr tensor, long offset);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_diagonal(IntPtr tensor, long offset, long dim1, long dim2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_erf(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_erf_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_erfc(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_erfc_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_erfinv(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_erfinv_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_eq(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_eq_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_eq_scalar(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_eq_scalar_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSTensor_equal(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool THSTensor_allclose(IntPtr tensor, IntPtr trg, double rtol, double atol, [MarshalAs(UnmanagedType.U1)] bool equal_nan);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ge(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ge_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ge_scalar(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ge_scalar_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_gt(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_gt_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_gt_scalar(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_gt_scalar_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_kron(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lcm(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lcm_(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ldexp(IntPtr right, IntPtr left);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ldexp_(IntPtr right, IntPtr left);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_le(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_le_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_le_scalar(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_le_scalar_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lt(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lt_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lt_scalar(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lt_scalar_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_masked_fill(IntPtr tensor, IntPtr mask, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_masked_fill_(IntPtr tensor, IntPtr mask, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_masked_scatter(IntPtr tensor, IntPtr mask, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_masked_scatter_(IntPtr tensor, IntPtr mask, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_masked_select(IntPtr tensor, IntPtr mask);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_topk(IntPtr tensor, AllocatePinnedArray allocator, int k, long dim, [MarshalAs(UnmanagedType.U1)] bool largest, [MarshalAs(UnmanagedType.U1)] bool sorted);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_unbind(IntPtr tensor, AllocatePinnedArray allocator, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_unfold(IntPtr tensor, long dim, long size, long step);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_split_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_split_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_tensor_split_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_tensor_split_with_tensor_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr indices, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_vsplit_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_vsplit_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_hsplit_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_hsplit_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_dsplit_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_dsplit_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_chunk(IntPtr tensor, AllocatePinnedArray allocator, long chunks, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_kthvalue(IntPtr tensor, long k, long dim, [MarshalAs(UnmanagedType.U1)] bool keepdim, out IntPtr _out);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_max(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_max_elementwise(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_max_along_dimension(IntPtr tensor, AllocatePinnedArray allocator, long dim, [MarshalAs(UnmanagedType.U1)] bool keep_dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mean(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_quantile(IntPtr tensor, IntPtr q, long dim, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_nanquantile(IntPtr tensor, IntPtr q, long dim, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mode(IntPtr tensor, AllocatePinnedArray allocator, long dim, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mean_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, [MarshalAs(UnmanagedType.U1)] bool keepdim, [MarshalAs(UnmanagedType.U1)] bool has_type, sbyte scalar_type);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_var_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, [MarshalAs(UnmanagedType.U1)] bool keepdim, [MarshalAs(UnmanagedType.U1)] bool has_type, sbyte scalar_type);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_median(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_min(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_min_elementwise(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_min_along_dimension(IntPtr tensor, AllocatePinnedArray allocator, long dim, [MarshalAs(UnmanagedType.U1)] bool keep_dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_msort(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sort(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool descending, [MarshalAs(UnmanagedType.U1)] bool stable, out IntPtr indices);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ne(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ne_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ne_scalar(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ne_scalar_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_dist(IntPtr tensor, IntPtr other, float p);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_norm(IntPtr tensor, float p);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_norm_along_dimension(IntPtr tensor, long dimension, [MarshalAs(UnmanagedType.U1)] bool keepdim, float p);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_outer(IntPtr input, IntPtr vec2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ormqr(IntPtr input, IntPtr tau, IntPtr other, [MarshalAs(UnmanagedType.U1)] bool left, [MarshalAs(UnmanagedType.U1)] bool transpose);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_inner(IntPtr input, IntPtr vec2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_inverse(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_prelu(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fmax(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fmin(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_renorm(IntPtr tensor, float p, long dim, float maxnorm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sigmoid(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sigmoid_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_std(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool unbiased);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_var(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool unbiased);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_std_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, [MarshalAs(UnmanagedType.U1)] bool unbiased, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_var_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, [MarshalAs(UnmanagedType.U1)] bool unbiased, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_std_mean(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool unbiased, out IntPtr mean);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_var_mean(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool unbiased, out IntPtr mean);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_std_mean_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, [MarshalAs(UnmanagedType.U1)] bool unbiased, [MarshalAs(UnmanagedType.U1)] bool keepdim, out IntPtr mean);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_var_mean_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, [MarshalAs(UnmanagedType.U1)] bool unbiased, [MarshalAs(UnmanagedType.U1)] bool keepdim, out IntPtr mean);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sum(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool has_type, sbyte scalar_type);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sum_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, [MarshalAs(UnmanagedType.U1)] bool keepdim, [MarshalAs(UnmanagedType.U1)] bool has_type, sbyte scalar_type);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_expand(IntPtr tensor, IntPtr psizes, int length, [MarshalAs(UnmanagedType.U1)] bool isImplicit);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_repeat(IntPtr tensor, IntPtr psizes, int length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_repeat_interleave(IntPtr tensor, IntPtr repeats, long dim, long output_size);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_repeat_interleave_int64(IntPtr tensor, long repeats, long dim, long output_size);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_broadcast_to(IntPtr tensor, IntPtr psizes, int length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_movedim(IntPtr tensor, IntPtr src, int src_len, IntPtr dst, int dst_len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_randn_out(IntPtr psizes, int length, IntPtr tensorOut);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rand_out(IntPtr psizes, int length, IntPtr tensorOut);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_randint_out(long high, IntPtr psizes, int length, IntPtr tensorOut);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rand_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_tensor_split_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_randn_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_randint_like(IntPtr input, long low, long high, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_randperm_out(long n, IntPtr tensorOut);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bernoulli(IntPtr tensor, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_multinomial(IntPtr tensor, long num_samples, [MarshalAs(UnmanagedType.U1)] bool replacement, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_poisson(IntPtr tensor, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bernoulli_0(IntPtr tensor, double p, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bernoulli_1(IntPtr tensor, IntPtr p_tensor, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_binomial(IntPtr count, IntPtr prob, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cauchy_(IntPtr tensor, double median, double sigma, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_exponential_(IntPtr tensor, double lambda, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_geometric_(IntPtr tensor, double p, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_normal_(IntPtr tensor, double mean, double std, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_log_normal_(IntPtr tensor, double mean, double std, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_random_(IntPtr tensor, double low, double high, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_uniform_(IntPtr tensor, double low, double high, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_arange_out(IntPtr start, IntPtr strp, IntPtr step, IntPtr tensorOut);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_permute(IntPtr tensor, IntPtr psizes, int length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ones_out(IntPtr psizes, int length, IntPtr tensorOut);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_zeros_out(IntPtr psizes, int length, IntPtr tensorOut);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_zeros_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ones_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_empty_out(IntPtr psizes, int length, IntPtr tensorOut);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_empty_like(IntPtr input, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_full_out(IntPtr psizes, int length, IntPtr value, IntPtr tensorOut);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_full_like(IntPtr input, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_detach(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_detach_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_eye_out(long rows, long columns, IntPtr tensorOut);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_scatter(IntPtr tensor, long dim, IntPtr index, IntPtr source);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_scatter_(IntPtr tensor, long dim, IntPtr index, IntPtr source);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_scatter_add(IntPtr tensor, long dim, IntPtr index, IntPtr source);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_diagonal_scatter(IntPtr tensor, IntPtr source, long offset, long dim1, long dim2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_select_scatter(IntPtr tensor, IntPtr source, long dim, long index);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_slice_scatter(IntPtr tensor, IntPtr source, long dim, IntPtr start, IntPtr end, long step);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_scatter_add_(IntPtr tensor, long dim, IntPtr index, IntPtr source);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_gather(IntPtr tensor, long dim, IntPtr index);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_flip(IntPtr tensor, IntPtr psizes, int length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fliplr(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_flipud(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_nanmean(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool keepdim, sbyte scalar_type);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_nanmedian(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_nansum(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_nan_to_num(IntPtr tensor, IntPtr nan, IntPtr posinf, IntPtr neginf);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_nextafter(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_narrow(IntPtr tensor, long dim, long start, long length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_nonzero(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_roll(IntPtr tensor, IntPtr shifts, int shLength, IntPtr dims, long dimLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rot90(IntPtr tensor, long k, long dim1, long dim2);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_slice(IntPtr tensor, long dim, long start, long length, long step);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_unsqueeze(IntPtr tensor, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_unsqueeze_(IntPtr tensor, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_where(IntPtr condition, IntPtr x, IntPtr y);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_atleast_1d(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_atleast_2d(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_atleast_3d(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_stft(IntPtr x, long n_fft, long hop_length, long win_length, IntPtr window, [MarshalAs(UnmanagedType.U1)] bool normalized, long onesided, [MarshalAs(UnmanagedType.U1)] bool return_complex);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_istft(IntPtr x, long n_fft, long hop_length, long win_length, IntPtr window, [MarshalAs(UnmanagedType.U1)] bool center, [MarshalAs(UnmanagedType.U1)] bool normalized, long onesided, long length, [MarshalAs(UnmanagedType.U1)] bool return_complex);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_to_type(IntPtr handle, sbyte scalar_type);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_arange(IntPtr start, IntPtr stop, IntPtr step, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requireGrad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newComplexFloat32Scalar(float real, float imaginary, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_randint(long max, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_randint(IntPtr generator, long low, long high, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sparse(IntPtr indices, IntPtr values, IntPtr sizes, int length, sbyte type, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_linspace(double start, double end, long steps, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logspace(double start, double end, long steps, double @base, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bartlett_window(long len, [MarshalAs(UnmanagedType.U1)] bool periodic, sbyte scalar_type, int device_type, int device_index, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_blackman_window(long len, [MarshalAs(UnmanagedType.U1)] bool periodic, sbyte scalar_type, int device_type, int device_index, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hamming_window(long len, [MarshalAs(UnmanagedType.U1)] bool periodic, double alpha, double beta, sbyte scalar_type, int device_type, int device_index, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hann_window(long len, [MarshalAs(UnmanagedType.U1)] bool periodic, sbyte scalar_type, int device_type, int device_index, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_kaiser_window(long len, [MarshalAs(UnmanagedType.U1)] bool periodic, double beta, sbyte scalar_type, int device_type, int device_index, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newBoolScalar([MarshalAs(UnmanagedType.U1)] bool scalar, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newByteScalar(byte scalar, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newInt8Scalar(sbyte scalar, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newInt16Scalar(short scalar, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newInt32Scalar(int scalar, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newInt64Scalar(long scalar, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newFloat16Scalar(float scalar, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newFloat32Scalar(float scalar, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newFloat64Scalar(double scalar, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newComplexFloat64Scalar(double real, double imaginary, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_frombuffer(IntPtr rawArray, GCHandleDeleter deleter, long count, long offset, sbyte type, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_new(IntPtr rawArray, GCHandleDeleter deleter, IntPtr dimensions, int numDimensions, sbyte type, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_newInt64(IntPtr rawArray, GCHandleDeleter deleter, IntPtr dimensions, int numDimensions, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rand(IntPtr generator, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_randn(IntPtr generator, IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_from_file(byte[] filename, sbyte shared, long size, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_complex(IntPtr real, IntPtr imag);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_polar(IntPtr abs, IntPtr angle);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fft(IntPtr tensor, long n, long dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ifft(IntPtr tensor, long n, long dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ifft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ifftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_irfft(IntPtr tensor, long n, long dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rfft(IntPtr tensor, long n, long dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rfft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_irfft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rfftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_irfftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hfft(IntPtr tensor, long n, long dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ihfft(IntPtr tensor, long n, long dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fftshift(IntPtr tensor, IntPtr dim, int dim_length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ifftshift(IntPtr tensor, IntPtr dim, int dim_length);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rfftfreq(long n, double d, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hfft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ihfft2(IntPtr tensor, IntPtr s, IntPtr dim, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hfftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ihfftn(IntPtr tensor, IntPtr s, int s_length, IntPtr dim, int dim_length, sbyte norm);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_angle(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_asin(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_asin_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_acos(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_acos_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_atan(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_atan_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_atan2_(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cos(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_atan2(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cos_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sin(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sin_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_tan(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_tan_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sinc(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sinc_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sinh(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sinh_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cosh(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cosh_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_tanh(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_tanh_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_arcsinh(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_arcsinh_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_arccosh(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_arccosh_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_arctanh(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_arctanh_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_block_diag(IntPtr tensor, int len);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_broadcast_tensors(IntPtr tensor, long length, AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cat(IntPtr tensor, int len, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cartesian_prod(IntPtr tensor, int len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_stack(IntPtr tensor, int len, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hstack(IntPtr tensor, int len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_vstack(IntPtr tensor, int len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_column_stack(IntPtr tensors, int len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_row_stack(IntPtr tensor, int len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_dstack(IntPtr tensor, int len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_meshgrid(IntPtr tensor, int len, [MarshalAs(UnmanagedType.LPStr)] string indexing, AllocatePinnedArray allocator);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_standard_gamma_(IntPtr tensor, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sample_dirichlet_(IntPtr tensor, IntPtr gen);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_abs(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_abs_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_add(IntPtr tensor, IntPtr trg, IntPtr alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_add_scalar(IntPtr tensor, IntPtr trg, IntPtr alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_add_(IntPtr tensor, IntPtr trg, IntPtr alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addmm_(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addmv(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta = 1, float alpha = 1);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addmv_(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addr(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addr_(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_and(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addmm(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_add_scalar_(IntPtr tensor, IntPtr trg, IntPtr alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addcdiv(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addbmm_(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addcmul(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addcmul_(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_xor(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_xor_(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_addcdiv_(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_not(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_not_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_and_(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_or(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_or_(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_left_shift_(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_right_shift(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ceil(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ceil_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_is_conj(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_resolve_conj(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern long THSTensor_is_neg(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_resolve_neg(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_left_shift(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_cummin(IntPtr tensor, AllocatePinnedArray allocator, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cumsum(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool has_type, sbyte scalar_type);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_bitwise_right_shift_(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_conj(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_conj_physical(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_div_scalar(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string? rounding_mode);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_div_(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string? rounding_mode);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_conj_physical_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_cummax(IntPtr tensor, AllocatePinnedArray allocator, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_exp(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_exp_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_exp2(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_expm1(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_floor(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_floor_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_floor_divide(IntPtr left, IntPtr right);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_floor_divide_(IntPtr left, IntPtr right);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_floor_divide_scalar(IntPtr left, IntPtr right);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_floor_divide_scalar_(IntPtr left, IntPtr right);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_frexp(IntPtr tensor, out IntPtr exponent);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_gcd(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_div(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string? rounding_mode);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fmod_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fmod_scalar(IntPtr tensor, IntPtr scalar);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_remainder_scalar_(IntPtr tensor, IntPtr scalar);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_round(IntPtr tensor, long decimals);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cumprod(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool has_type, sbyte scalar_type);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_div_scalar_(IntPtr tensor, IntPtr trg, [MarshalAs(UnmanagedType.LPStr)] string? rounding_mode);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_histc(IntPtr tensor, long bins, long min, long max);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_hypot(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_log10(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_log10_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logcumsumexp(IntPtr tensor, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logsumexp(IntPtr tensor, long dim, [MarshalAs(UnmanagedType.U1)] bool keepdim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logical_or(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logical_or_(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logical_not(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logical_not_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mul_scalar_(IntPtr tensor, IntPtr target);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_neg(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_reciprocal_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_remainder(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logical_and(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logical_and_(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_pow_scalar(IntPtr tensor, IntPtr scalar);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_pow_scalar_(IntPtr tensor, IntPtr scalar);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_remainder_scalar(IntPtr tensor, IntPtr scalar);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_reciprocal(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_remainder_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cumulative_trapezoid_x(IntPtr y, IntPtr x, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cumulative_trapezoid_dx(IntPtr y, double dx, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rsqrt_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sqrt(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_float_power(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fmod_scalar_(IntPtr tensor, IntPtr scalar);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_frac(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logit(IntPtr tensor, IntPtr eps);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mul(IntPtr tensor, IntPtr target);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_expm1_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_fmod(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_frac_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_gcd_(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_log(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_log_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logaddexp(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logaddexp2(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_log2(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_log2_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_log1p_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_log1p(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logical_xor(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logical_xor_(IntPtr tensor, IntPtr other);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mul_(IntPtr tensor, IntPtr target);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_pow(IntPtr tensor, IntPtr exponent);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_pow_(IntPtr tensor, IntPtr exponent);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mul_scalar(IntPtr tensor, IntPtr scalar);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_neg_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_round_(IntPtr tensor, long decimals);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_rsqrt(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sqrt_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sign(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sign_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sgn(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sgn_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_signbit(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sub(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sub_scalar(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sub_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_sub_scalar_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_trapezoid_x(IntPtr y, IntPtr x, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_trapezoid_dx(IntPtr y, double dx, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_true_divide(IntPtr left, IntPtr right);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_true_divide_(IntPtr left, IntPtr right);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_true_divide_scalar(IntPtr left, IntPtr right);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_true_divide_scalar_(IntPtr left, IntPtr right);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_trunc_(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_xlogy(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_xlogy_scalar(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_xlogy_scalar_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_xlogy_(IntPtr tensor, IntPtr trg);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_trunc(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_einsum([MarshalAs(UnmanagedType.LPStr)] string location, IntPtr tensors, int len);

        [DllImport("LibTorchSharp")]
        internal static extern double THSTensor_clip_grad_norm_(IntPtr tensors, int len, double max_norm, double norm_type);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_clip_grad_value_(IntPtr tensors, int len, double clip_value);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_parameters_to_vector(IntPtr tensors, int len);

        [DllImport("LibTorchSharp")]
        internal static extern void THSTensor_vector_to_parameters(IntPtr vec, IntPtr tensors, int len);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cholesky(IntPtr input, [MarshalAs(UnmanagedType.U1)] bool upper);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cholesky_inverse(IntPtr input, [MarshalAs(UnmanagedType.U1)] bool upper);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cholesky_solve(IntPtr input, IntPtr input2, [MarshalAs(UnmanagedType.U1)] bool upper);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_cross(IntPtr input, IntPtr other, long dim);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_eig(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool eigenvectors, out IntPtr pEigenvectors);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_matmul(IntPtr tensor, IntPtr target);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mm(IntPtr tensor, IntPtr target);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_mv(IntPtr tensor, IntPtr target);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_matrix_exp(IntPtr input);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_vdot(IntPtr tensor, IntPtr target);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_dot(IntPtr tensor, IntPtr target);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_logdet(IntPtr tensor);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lu(IntPtr tensor, [MarshalAs(UnmanagedType.U1)] bool pivot, [MarshalAs(UnmanagedType.U1)] bool get_infos, out IntPtr infos, out IntPtr pivots);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lu_solve(IntPtr tensor, IntPtr LU_data, IntPtr LU_pivots);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_lu_unpack(IntPtr LU_data, IntPtr LU_pivots, [MarshalAs(UnmanagedType.U1)] bool unpack_data, [MarshalAs(UnmanagedType.U1)] bool unpack_pivots, out IntPtr L, out IntPtr U);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_zeros(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requireGrad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_randperm(IntPtr generator, long n, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requireGrad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_ones(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_empty_strided(IntPtr psizes, int sz_length, IntPtr pstrides, int str_length, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_empty(IntPtr psizes, int length, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_full(IntPtr psizes, int length, IntPtr value, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_eye(long rows, long columns, sbyte scalarType, int deviceType, int deviceIndex, [MarshalAs(UnmanagedType.U1)] bool requires_grad);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_maxunpool2d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_adaptive_avg_pool3d(IntPtr input, IntPtr outputSize, int outputSizeLength);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_adaptive_avg_pool3d_backward_out(IntPtr gradInput, IntPtr gradOutput, IntPtr originalInput);

        [DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_searchsorted_t(IntPtr sorted_sequence, IntPtr values, bool out_int32, bool right, IntPtr sorter);
[DllImport("LibTorchSharp")]
        internal static extern IntPtr THSTensor_searchsorted_s(IntPtr sorted_sequence, IntPtr values, bool out_int32, bool right, IntPtr sorter);
    }
}
