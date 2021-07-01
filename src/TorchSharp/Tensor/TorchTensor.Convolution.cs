using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

#nullable enable
namespace TorchSharp.Tensor
{
    // This file contains the convolution, pooling, and padding operations on tensors.

    public sealed partial class TorchTensor
    {

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        /// <summary>
        /// Applies a 1D convolution over an input signal composed of several input planes.
        /// </summary>
        /// <param name="weight"></param>
        /// <param name="bias"></param>
        /// <param name="stride"></param>
        /// <param name="padding"></param>
        /// <param name="dilation"></param>
        /// <param name="groups"></param>
        /// <returns></returns>
        public TorchTensor conv1d(TorchTensor weight, TorchTensor? bias = null,
            long? stride = null,
            long? padding = null,
            long? dilation = null,
            long groups = 1)
        {
            var strides = new long[] { stride ?? 1 };
            var paddingArray = new long[] { padding ?? 0 };
            var dilationArray = new long[] { dilation ?? 1 };
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = paddingArray, pdilation = dilationArray) {
                    var res =
                        THSTensor_conv1d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddingArray.Length,
                            (IntPtr)pdilation, dilationArray.Length,
                            groups);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }


        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        /// <summary>
        /// Applies a 2D convolution over an input image composed of several input planes.
        /// </summary>
        /// <param name="weight"></param>
        /// <param name="bias"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <param name="dilation"></param>
        /// <param name="groups"></param>
        /// <returns></returns>
        public TorchTensor conv2d(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            padding = (padding == null) ? new long[] { 0 } : padding;
            dilation = (dilation == null) ? new long[] { 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation) {
                    var res =
                        THSTensor_conv2d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        /// <summary>
        /// Applies a 3D convolution over an input image composed of several input planes.
        /// </summary>
        /// <param name="weight"></param>
        /// <param name="bias"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <param name="dilation"></param>
        /// <param name="groups"></param>
        /// <returns></returns>
        public TorchTensor conv3d(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            padding = (padding == null) ? new long[] { 0 } : padding;
            dilation = (dilation == null) ? new long[] { 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation) {
                    var res =
                        THSTensor_conv3d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv_transpose1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        /// <summary>
        /// Applies a 1D transposed convolution operator over an input signal composed of several input planes, sometimes also called “deconvolution”.
        /// </summary>
        /// <param name="weight"></param>
        /// <param name="bias"></param>
        /// <param name="stride"></param>
        /// <param name="padding"></param>
        /// <param name="outputPadding"></param>
        /// <param name="dilation"></param>
        /// <param name="groups"></param>
        /// <returns></returns>
        public TorchTensor conv_transpose1d(TorchTensor weight, TorchTensor? bias = null,
            long? stride = null,
            long? padding = null,
            long? outputPadding = null,
            long? dilation = null,
            long groups = 1)
        {
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var outputPaddings = new long[] { outputPadding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = paddings, poutputPadding = outputPaddings, pdilation = dilations) {
                    var res =
                        THSTensor_conv_transpose1d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)poutputPadding, outputPaddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            groups);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv_transpose2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        /// <summary>
        /// Applies a 2D transposed convolution operator over an input image composed of several input planes, sometimes also called “deconvolution”.
        /// </summary>
        /// <param name="weight"></param>
        /// <param name="bias"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <param name="outputPadding"></param>
        /// <param name="dilation"></param>
        /// <param name="groups"></param>
        /// <returns></returns>
        public TorchTensor conv_transpose2d(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? outputPadding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1, 1 } : strides;
            padding = (padding == null) ? new long[] { 0, 0 } : padding;
            outputPadding = (outputPadding == null) ? new long[] { 0, 0 } : outputPadding;
            dilation = (dilation == null) ? new long[] { 1, 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = padding, poutputPadding = outputPadding, pdilation = dilation) {
                    var res =
                        THSTensor_conv_transpose2d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)poutputPadding, outputPadding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_conv_transpose3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        /// <summary>
        /// Applies a 3D transposed convolution operator over an input image composed of several input planes, sometimes also called “deconvolution”.
        /// </summary>
        /// <param name="weight"></param>
        /// <param name="bias"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <param name="outputPadding"></param>
        /// <param name="dilation"></param>
        /// <param name="groups"></param>
        /// <returns></returns>
        public TorchTensor conv_transpose3d(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? outputPadding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1, 1, 1 } : strides;
            padding = (padding == null) ? new long[] { 0, 0, 0 } : padding;
            outputPadding = (outputPadding == null) ? new long[] { 0, 0, 0 } : outputPadding;
            dilation = (dilation == null) ? new long[] { 1, 1, 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = padding, poutputPadding = outputPadding, pdilation = dilation) {
                    var res =
                        THSTensor_conv_transpose3d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)poutputPadding, outputPadding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_max_pool1d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        /// <summary>
        /// Applies a 1D max pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="kernelSize"></param>
        /// <param name="stride"></param>
        /// <param name="padding"></param>
        /// <param name="dilation"></param>
        /// <param name="ceil_mode"></param>
        /// <returns></returns>
        public TorchTensor max_pool1d(long kernelSize, long? stride = null,
            long? padding = null, long? dilation = null, bool ceil_mode = false)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings, pdilation = dilations) {
                    var res =
                        THSTensor_max_pool1d(handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            ceil_mode);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_max_pool1d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        /// <summary>
        /// Applies a 1D max pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="kernelSize"></param>
        /// <param name="stride"></param>
        /// <param name="padding"></param>
        /// <param name="dilation"></param>
        /// <param name="ceil_mode"></param>
        /// <returns></returns>
        public (TorchTensor output, TorchTensor indices) max_pool1d_with_indices(long kernelSize, long? stride = null,
            long? padding = null, long? dilation = null, bool ceil_mode = false)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings, pdilation = dilations) {
                        THSTensor_max_pool1d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            ceil_mode);
                        torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_max_pool2d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        /// <summary>
        /// Applies a 2D max pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="kernelSize"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <param name="dilation"></param>
        /// <param name="ceil_mode"></param>
        /// <returns></returns>
        public TorchTensor max_pool2d(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            unsafe {
                fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                    var res =
                        THSTensor_max_pool2d(handle,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_max_pool2d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        /// <summary>
        /// Applies a 2D max pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="kernelSize"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <param name="dilation"></param>
        /// <param name="ceil_mode"></param>
        /// <returns></returns>
        public (TorchTensor output, TorchTensor indices) max_pool2d_with_indices(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                        THSTensor_max_pool2d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                        torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_max_pool3d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        /// <summary>
        /// Applies a 3D max pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="kernelSize"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <param name="dilation"></param>
        /// <param name="ceil_mode"></param>
        /// <returns></returns>
        public TorchTensor max_pool3d(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            unsafe {
                fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                    var res =
                        THSTensor_max_pool3d(handle,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern void THSTensor_max_pool3d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        /// <summary>
        /// Applies a 3D max pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="kernelSize"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <param name="dilation"></param>
        /// <param name="ceil_mode"></param>
        /// <returns></returns>
        public (TorchTensor output, TorchTensor indices) max_pool3d_with_indices(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                        THSTensor_max_pool3d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                        torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_maxunpool2d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength);

        /// <summary>
        /// Computes a partial inverse of MaxPool2d.
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="outputSize"></param>
        /// <returns></returns>
        public TorchTensor maxunpool2d(TorchTensor indices, long[] outputSize)
        {
            unsafe {
                fixed (long* poutputSize = outputSize) {
                    var res = THSTensor_maxunpool2d(handle, indices.Handle,
                        (IntPtr)poutputSize, outputSize.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_maxunpool3d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength, IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength);

        /// <summary>
        /// Computes a partial inverse of MaxPool3d.
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="outputSize"></param>
        /// <param name="strides"></param>
        /// <param name="padding"></param>
        /// <returns></returns>
        public TorchTensor maxunpool3d(TorchTensor indices, long[] outputSize, long[] strides, long[] padding)
        {
            unsafe {
                fixed (long* poutputSize = outputSize, pstrides = strides, ppadding = padding) {
                    var res = THSTensor_maxunpool3d(handle, indices.Handle,
                        (IntPtr)poutputSize, outputSize.Length,
                        (IntPtr)pstrides, strides.Length,
                        (IntPtr)ppadding, padding.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_avg_pool1d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad);

        /// <summary>
        /// Applies a 1D average pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="kernelSize"></param>
        /// <param name="stride"></param>
        /// <param name="padding"></param>
        /// <param name="ceil_mode"></param>
        /// <param name="count_include_pad"></param>
        /// <returns></returns>
        public TorchTensor avg_pool1d(long kernelSize, long? stride = null,
            long? padding = null, bool ceil_mode = false, bool count_include_pad = true)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool1d(handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_avg_pool2d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad);

        /// <summary>
        /// Applies 2D average-pooling operation in kH × kW regions by step size sH * sW steps. The number of output features is equal to the number of input planes.
        /// </summary>
        /// <param name="kernelSizes"></param>
        /// <param name="strides"></param>
        /// <param name="paddings"></param>
        /// <param name="ceil_mode"></param>
        /// <param name="count_include_pad"></param>
        /// <returns></returns>
        public TorchTensor avg_pool2d(long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool2d(handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_avg_pool3d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad);

        /// <summary>
        /// Applies 3D average-pooling operation in kT x kH x kW regions by step size sT x sH x sW steps.
        /// </summary>
        /// <param name="kernelSizes"></param>
        /// <param name="strides"></param>
        /// <param name="paddings"></param>
        /// <param name="ceil_mode"></param>
        /// <param name="count_include_pad"></param>
        /// <returns></returns>
        public TorchTensor avg_pool3d(long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool3d(handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_avg_pool2d_backward(IntPtr gradOutput, IntPtr originalInput,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad,
                long divisorOverride);

        public TorchTensor avg_pool2d_backward(TorchTensor originalInput,
            long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true,
            long divisorOverride = 0)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool2d_backward(handle, originalInput.Handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad,
                            divisorOverride);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_avg_pool3d_backward(IntPtr gradOutput, IntPtr originalInput,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad,
                long divisorOverride);

        public TorchTensor avg_pool3d_backward(TorchTensor originalInput,
            long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true,
            long divisorOverride = 0)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool3d_backward(handle, originalInput.Handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad,
                            divisorOverride);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_adaptive_avg_pool1d(IntPtr input,
                IntPtr outputSize, int outputSizeLength);

        /// <summary>
        /// Applies a 1D adaptive average pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="outputSize"></param>
        /// <returns></returns>
        public TorchTensor adaptive_avg_pool1d(long outputSize)
        {
            var outputSizes = new long[] { outputSize };
            unsafe {
                fixed (long* poutputSize = outputSizes) {
                    var res =
                        THSTensor_adaptive_avg_pool1d(handle, (IntPtr)poutputSize, outputSizes.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_adaptive_avg_pool2d(IntPtr input,
                IntPtr outputSize, int outputSizeLength);

        /// <summary>
        /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="outputSizes"></param>
        /// <returns></returns>
        public TorchTensor adaptive_avg_pool2d(long[] outputSizes)
        {
            unsafe {
                fixed (long* poutputSize = outputSizes) {
                    var res =
                        THSTensor_adaptive_avg_pool2d(handle, (IntPtr)poutputSize, outputSizes.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_adaptive_avg_pool3d(IntPtr input, IntPtr outputSize, int outputSizeLength);

        /// <summary>
        /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
        /// </summary>
        /// <param name="outputSizes"></param>
        /// <returns></returns>
        public TorchTensor adaptive_avg_pool3d(long[] outputSizes)
        {
            unsafe {
                fixed (long* poutputSize = outputSizes) {
                    var res =
                        THSTensor_adaptive_avg_pool3d(handle, (IntPtr)poutputSize, outputSizes.Length);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_adaptive_avg_pool3d_backward_out(IntPtr gradInput, IntPtr gradOutput, IntPtr originalInput);

        public static TorchTensor adaptive_avg_pool3d_backward(TorchTensor gradInput, TorchTensor gradOutput, TorchTensor originalInput)
        {
            var res = THSTensor_adaptive_avg_pool3d_backward_out(gradInput.Handle, gradOutput.Handle, originalInput.Handle);
            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest1d(IntPtr input,
                IntPtr outputSize, int outputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        /// <summary>
        /// Upsamples the input, using nearest neighbours’ pixel values.
        /// </summary>
        /// <param name="outputSize"></param>
        /// <param name="scaleFactor"></param>
        /// <returns></returns>
        public TorchTensor upsample_nearest1d(long? outputSize, double? scaleFactor)
        {
            var outputSizes = outputSize.HasValue ? new long[] { outputSize.Value } : null;
            var outputSizesLength = outputSize.HasValue ? 1 : 0;
            var scaleFactors = scaleFactor.HasValue ? new double[] { scaleFactor.Value } : null;
            var scaleFactorsLength = scaleFactor.HasValue ? 1 : 0;
            unsafe {
                fixed (long* poutputSizes = outputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest1d(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest1d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor upsample_nearest1d_backward(long? outputSize, long inputSize, double? scaleFactor)
        {
            var outputSizes = outputSize.HasValue ? new long[] { outputSize.Value } : null;
            var outputSizesLength = outputSize.HasValue ? 1 : 0;
            var inputSizes = new long[] { inputSize };
            var scaleFactors = scaleFactor.HasValue ? new double[] { scaleFactor.Value } : null;
            var scaleFactorsLength = scaleFactor.HasValue ? 1 : 0;
            unsafe {
                fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest1d_backward(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pinputSizes, inputSizes.Length,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest2d(IntPtr input,
                IntPtr outputSize, int outputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        /// <summary>
        /// Upsamples the input, using nearest neighbours’ pixel values.
        /// </summary>
        /// <param name="outputSizes"></param>
        /// <param name="scaleFactors"></param>
        /// <returns></returns>
        public TorchTensor upsample_nearest2d(long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest2d(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest2d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor upsample_nearest2d_backward(long[] inputSizes, long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest2d_backward(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pinputSizes, inputSizes.Length,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest3d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor upsample_nearest3d_backward(long[] inputSizes, long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest3d_backward(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pinputSizes, inputSizes.Length,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        static extern IntPtr THSTensor_upsample_nearest3d(IntPtr input,
                IntPtr outputSize, int outputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        /// <summary>
        /// Upsamples the input, using nearest neighbours’ pixel values.
        /// </summary>
        /// <param name="outputSizes"></param>
        /// <param name="scaleFactors"></param>
        /// <returns></returns>
        public TorchTensor upsample_nearest3d(long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest3d(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }
    }
}
