using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class nn
        {
            public static partial class functional
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
                /// <param name="input">The input tensor.</param>
                /// <param name="weight"></param>
                /// <param name="bias"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="groups"></param>
                /// <returns></returns>
                public static Tensor conv1d(Tensor input, Tensor weight, Tensor bias = null,
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
                                THSTensor_conv1d(input.handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddingArray.Length,
                                    (IntPtr)pdilation, dilationArray.Length,
                                    groups);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="weight"></param>
                /// <param name="bias"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="groups"></param>
                /// <returns></returns>
                public static Tensor conv2d(Tensor input, Tensor weight, Tensor bias = null,
                    long[] strides = null,
                    long[] padding = null,
                    long[] dilation = null,
                    long groups = 1)
                {
                    strides = (strides == null) ? new long[] { 1 } : strides;
                    padding = (padding == null) ? new long[] { 0 } : padding;
                    dilation = (dilation == null) ? new long[] { 1 } : dilation;
                    var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
                    unsafe {
                        fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation) {
                            var res =
                                THSTensor_conv2d(input.handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    (IntPtr)pdilation, dilation.Length,
                                    groups);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="weight"></param>
                /// <param name="bias"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="groups"></param>
                /// <returns></returns>
                public static Tensor conv3d(Tensor input, Tensor weight, Tensor bias = null,
                    long[] strides = null,
                    long[] padding = null,
                    long[] dilation = null,
                    long groups = 1)
                {
                    strides = (strides == null) ? new long[] { 1 } : strides;
                    padding = (padding == null) ? new long[] { 0 } : padding;
                    dilation = (dilation == null) ? new long[] { 1 } : dilation;
                    var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
                    unsafe {
                        fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation) {
                            var res =
                                THSTensor_conv3d(input.handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    (IntPtr)pdilation, dilation.Length,
                                    groups);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="weight"></param>
                /// <param name="bias"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="outputPadding"></param>
                /// <param name="dilation"></param>
                /// <param name="groups"></param>
                /// <returns></returns>
                public static Tensor conv_transpose1d(Tensor input, Tensor weight, Tensor bias = null,
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
                                THSTensor_conv_transpose1d(input.handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    (IntPtr)poutputPadding, outputPaddings.Length,
                                    (IntPtr)pdilation, dilations.Length,
                                    groups);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="weight"></param>
                /// <param name="bias"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="outputPadding"></param>
                /// <param name="dilation"></param>
                /// <param name="groups"></param>
                /// <returns></returns>
                public static Tensor conv_transpose2d(Tensor input, Tensor weight, Tensor bias = null,
                    long[] strides = null,
                    long[] padding = null,
                    long[] outputPadding = null,
                    long[] dilation = null,
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
                                THSTensor_conv_transpose2d(input.handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    (IntPtr)poutputPadding, outputPadding.Length,
                                    (IntPtr)pdilation, dilation.Length,
                                    groups);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="weight"></param>
                /// <param name="bias"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="outputPadding"></param>
                /// <param name="dilation"></param>
                /// <param name="groups"></param>
                /// <returns></returns>
                public static Tensor conv_transpose3d(Tensor input, Tensor weight, Tensor bias = null,
                    long[] strides = null,
                    long[] padding = null,
                    long[] outputPadding = null,
                    long[] dilation = null,
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
                                THSTensor_conv_transpose3d(input.handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    (IntPtr)poutputPadding, outputPadding.Length,
                                    (IntPtr)pdilation, dilation.Length,
                                    groups);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSize"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static Tensor max_pool1d(Tensor input, long kernelSize, long? stride = null,
                    long? padding = null, long? dilation = null, bool ceil_mode = false)
                {
                    var kernelSizes = new long[] { kernelSize };
                    var strides = new long[] { stride ?? 1 };
                    var paddings = new long[] { padding ?? 0 };
                    var dilations = new long[] { dilation ?? 1 };
                    unsafe {
                        fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings, pdilation = dilations) {
                            var res =
                                THSTensor_max_pool1d(input.handle,
                                    (IntPtr)pkernelSize, kernelSizes.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    (IntPtr)pdilation, dilations.Length,
                                    ceil_mode);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSize"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static (Tensor output, Tensor indices) max_pool1d_with_indices(Tensor input, long kernelSize, long? stride = null,
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
                                THSTensor_max_pool1d_with_indices(input.handle,
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
                    return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
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
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSize"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static Tensor max_pool2d(Tensor input, long[] kernelSize, long[] strides = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    strides = strides ?? kernelSize.Select(x => 1L).ToArray();
                    padding = padding ?? kernelSize.Select(x => 0L).ToArray();
                    dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
                    unsafe {
                        fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                            var res =
                                THSTensor_max_pool2d(input.handle,
                                    (IntPtr)pkernelSize, kernelSize.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    (IntPtr)pdilation, dilation.Length,
                                    ceil_mode);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSize"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static (Tensor output, Tensor indices) max_pool2d_with_indices(Tensor input, long[] kernelSize, long[] strides = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    strides = strides ?? kernelSize.Select(x => 1L).ToArray();
                    padding = padding ?? kernelSize.Select(x => 0L).ToArray();
                    dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        unsafe {
                            fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                                THSTensor_max_pool2d_with_indices(input.handle,
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
                    return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
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
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSize"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static Tensor max_pool3d(Tensor input, long[] kernelSize, long[] strides = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    strides = strides ?? kernelSize.Select(x => 1L).ToArray();
                    padding = padding ?? kernelSize.Select(x => 0L).ToArray();
                    dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
                    unsafe {
                        fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                            var res =
                                THSTensor_max_pool3d(input.handle,
                                    (IntPtr)pkernelSize, kernelSize.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    (IntPtr)pdilation, dilation.Length,
                                    ceil_mode);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSize"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="dilation"></param>
                /// <param name="ceil_mode"></param>
                /// <returns></returns>
                public static (Tensor output, Tensor indices) max_pool3d_with_indices(Tensor input, long[] kernelSize, long[] strides = null,
                    long[] padding = null, long[] dilation = null, bool ceil_mode = false)
                {
                    strides = strides ?? kernelSize.Select(x => 1L).ToArray();
                    padding = padding ?? kernelSize.Select(x => 0L).ToArray();
                    dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
                    IntPtr[] ptrArray;

                    using (var pa = new PinnedArray<IntPtr>()) {
                        unsafe {
                            fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                                THSTensor_max_pool3d_with_indices(input.handle,
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
                    return (new Tensor(ptrArray[0]), new Tensor(ptrArray[1]));
                }

                [DllImport("LibTorchSharp")]
                static extern IntPtr THSTensor_maxunpool2d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength);

                /// <summary>
                /// Computes a partial inverse of MaxPool2d.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="indices"></param>
                /// <param name="outputSize"></param>
                /// <returns></returns>
                public static Tensor max_unpool2d(Tensor input, Tensor indices, long[] outputSize)
                {
                    unsafe {
                        fixed (long* poutputSize = outputSize) {
                            var res = THSTensor_maxunpool2d(input.handle, indices.Handle,
                                (IntPtr)poutputSize, outputSize.Length);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                [DllImport("LibTorchSharp")]
                static extern IntPtr THSTensor_maxunpool3d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength, IntPtr strides, int stridesLength,
                        IntPtr padding, int paddingLength);

                /// <summary>
                /// Computes a partial inverse of MaxPool3d.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="indices"></param>
                /// <param name="outputSize"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <returns></returns>
                public static Tensor max_unpool3d(Tensor input, Tensor indices, long[] outputSize, long[] strides, long[] padding)
                {
                    unsafe {
                        fixed (long* poutputSize = outputSize, pstrides = strides, ppadding = padding) {
                            var res = THSTensor_maxunpool3d(input.handle, indices.Handle,
                                (IntPtr)poutputSize, outputSize.Length,
                                (IntPtr)pstrides, strides.Length,
                                (IntPtr)ppadding, padding.Length);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSize"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="ceil_mode"></param>
                /// <param name="count_include_pad"></param>
                /// <returns></returns>
                public static Tensor avg_pool1d(Tensor input, long kernelSize, long? stride = null,
                    long? padding = null, bool ceil_mode = false, bool count_include_pad = true)
                {
                    var kernelSizes = new long[] { kernelSize };
                    var strides = new long[] { stride ?? 1 };
                    var paddings = new long[] { padding ?? 0 };
                    unsafe {
                        fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                            var res =
                                THSTensor_avg_pool1d(input.handle,
                                    (IntPtr)pkernelSize, kernelSizes.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    ceil_mode,
                                    count_include_pad);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSizes"></param>
                /// <param name="strides"></param>
                /// <param name="paddings"></param>
                /// <param name="ceil_mode"></param>
                /// <param name="count_include_pad"></param>
                /// <returns></returns>
                public static Tensor avg_pool2d(Tensor input, long[] kernelSizes,
                    long[] strides = null,
                    long[] paddings = null,
                    bool ceil_mode = false,
                    bool count_include_pad = true)
                {
                    strides = (strides == null) ? new long[] { 1 } : strides;
                    paddings = (paddings == null) ? new long[] { 0 } : paddings;
                    unsafe {
                        fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                            var res =
                                THSTensor_avg_pool2d(input.handle,
                                    (IntPtr)pkernelSize, kernelSizes.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    ceil_mode,
                                    count_include_pad);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="kernelSizes"></param>
                /// <param name="strides"></param>
                /// <param name="paddings"></param>
                /// <param name="ceil_mode"></param>
                /// <param name="count_include_pad"></param>
                /// <returns></returns>
                public static Tensor avg_pool3d(Tensor input, long[] kernelSizes,
                    long[] strides = null,
                    long[] paddings = null,
                    bool ceil_mode = false,
                    bool count_include_pad = true)
                {
                    strides = (strides == null) ? new long[] { 1 } : strides;
                    paddings = (paddings == null) ? new long[] { 0 } : paddings;
                    unsafe {
                        fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                            var res =
                                THSTensor_avg_pool3d(input.handle,
                                    (IntPtr)pkernelSize, kernelSizes.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    ceil_mode,
                                    count_include_pad);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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

                public static Tensor avg_pool2d_backward(Tensor input, Tensor originalInput,
                    long[] kernelSizes,
                    long[] strides = null,
                    long[] paddings = null,
                    bool ceil_mode = false,
                    bool count_include_pad = true,
                    long divisorOverride = 0)
                {
                    strides = (strides == null) ? new long[] { 1 } : strides;
                    paddings = (paddings == null) ? new long[] { 0 } : paddings;
                    unsafe {
                        fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                            var res =
                                THSTensor_avg_pool2d_backward(input.handle, originalInput.Handle,
                                    (IntPtr)pkernelSize, kernelSizes.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    ceil_mode,
                                    count_include_pad,
                                    divisorOverride);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
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

                public static Tensor avg_pool3d_backward(Tensor input, Tensor originalInput,
                    long[] kernelSizes,
                    long[] strides = null,
                    long[] paddings = null,
                    bool ceil_mode = false,
                    bool count_include_pad = true,
                    long divisorOverride = 0)
                {
                    strides = (strides == null) ? new long[] { 1 } : strides;
                    paddings = (paddings == null) ? new long[] { 0 } : paddings;
                    unsafe {
                        fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                            var res =
                                THSTensor_avg_pool3d_backward(input.handle, originalInput.Handle,
                                    (IntPtr)pkernelSize, kernelSizes.Length,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddings.Length,
                                    ceil_mode,
                                    count_include_pad,
                                    divisorOverride);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                [DllImport("LibTorchSharp")]
                static extern IntPtr THSTensor_adaptive_avg_pool1d(IntPtr input,
                        IntPtr outputSize, int outputSizeLength);

                /// <summary>
                /// Applies a 1D adaptive average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="outputSize"></param>
                /// <returns></returns>
                public static Tensor adaptive_avg_pool1d(Tensor input, long outputSize)
                {
                    var outputSizes = new long[] { outputSize };
                    unsafe {
                        fixed (long* poutputSize = outputSizes) {
                            var res =
                                THSTensor_adaptive_avg_pool1d(input.handle, (IntPtr)poutputSize, outputSizes.Length);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                [DllImport("LibTorchSharp")]
                static extern IntPtr THSTensor_adaptive_avg_pool2d(IntPtr input,
                        IntPtr outputSize, int outputSizeLength);

                /// <summary>
                /// Applies a 2D adaptive average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="outputSizes"></param>
                /// <returns></returns>
                public static Tensor adaptive_avg_pool2d(Tensor input, long[] outputSizes)
                {
                    unsafe {
                        fixed (long* poutputSize = outputSizes) {
                            var res =
                                THSTensor_adaptive_avg_pool2d(input.handle, (IntPtr)poutputSize, outputSizes.Length);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                [DllImport("LibTorchSharp")]
                static extern IntPtr THSTensor_adaptive_avg_pool3d(IntPtr input, IntPtr outputSize, int outputSizeLength);

                /// <summary>
                /// Applies a 3D adaptive average pooling over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="outputSizes"></param>
                /// <returns></returns>
                public static Tensor adaptive_avg_pool3d(Tensor input, long[] outputSizes)
                {
                    unsafe {
                        fixed (long* poutputSize = outputSizes) {
                            var res =
                                THSTensor_adaptive_avg_pool3d(input.handle, (IntPtr)poutputSize, outputSizes.Length);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                [DllImport("LibTorchSharp")]
                static extern IntPtr THSTensor_adaptive_avg_pool3d_backward_out(IntPtr gradInput, IntPtr gradOutput, IntPtr originalInput);

                public static Tensor adaptive_avg_pool3d_backward(Tensor gradInput, Tensor gradOutput, Tensor originalInput)
                {
                    var res = THSTensor_adaptive_avg_pool3d_backward_out(gradInput.Handle, gradOutput.Handle, originalInput.Handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }

                [DllImport("LibTorchSharp")]
                static extern IntPtr THSTensor_upsample_nearest1d(IntPtr input,
                        IntPtr outputSize, int outputSizeLength,
                        IntPtr scaleFactors, int scaleFactorsLength);

                /// <summary>
                /// Upsamples the input, using nearest neighbours’ pixel values.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="outputSize"></param>
                /// <param name="scaleFactor"></param>
                /// <returns></returns>
                public static Tensor upsample_nearest1d(Tensor input, long? outputSize, double? scaleFactor)
                {
                    var outputSizes = outputSize.HasValue ? new long[] { outputSize.Value } : null;
                    var outputSizesLength = outputSize.HasValue ? 1 : 0;
                    var scaleFactors = scaleFactor.HasValue ? new double[] { scaleFactor.Value } : null;
                    var scaleFactorsLength = scaleFactor.HasValue ? 1 : 0;
                    unsafe {
                        fixed (long* poutputSizes = outputSizes) {
                            fixed (double* pscaleFactors = scaleFactors) {
                                var res =
                                    THSTensor_upsample_nearest1d(input.handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
                            }
                        }
                    }
                }

                [DllImport("LibTorchSharp")]
                static extern IntPtr THSTensor_upsample_nearest1d_backward(IntPtr grad_output,
                        IntPtr outputSize, int outputSizeLength,
                        IntPtr inputSize, int inputSizeLength,
                        IntPtr scaleFactors, int scaleFactorsLength);

                public static Tensor upsample_nearest1d_backward(Tensor grad_output, long? outputSize, long inputSize, double? scaleFactor)
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
                                    THSTensor_upsample_nearest1d_backward(grad_output.handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pinputSizes, inputSizes.Length,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="outputSizes"></param>
                /// <param name="scaleFactors"></param>
                /// <returns></returns>
                public static Tensor upsample_nearest2d(Tensor input, long[] outputSizes = null, double[] scaleFactors = null)
                {
                    var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
                    var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
                    unsafe {
                        fixed (long* poutputSizes = outputSizes) {
                            fixed (double* pscaleFactors = scaleFactors) {
                                var res =
                                    THSTensor_upsample_nearest2d(input.handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
                            }
                        }
                    }
                }

                [DllImport("LibTorchSharp")]
                static extern IntPtr THSTensor_upsample_nearest2d_backward(IntPtr grad_output,
                        IntPtr outputSize, int outputSizeLength,
                        IntPtr inputSize, int inputSizeLength,
                        IntPtr scaleFactors, int scaleFactorsLength);

                public static Tensor upsample_nearest2d_backward(Tensor grad_output, long[] inputSizes, long[] outputSizes = null, double[] scaleFactors = null)
                {
                    var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
                    var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
                    unsafe {
                        fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                            fixed (double* pscaleFactors = scaleFactors) {
                                var res =
                                    THSTensor_upsample_nearest2d_backward(grad_output.handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pinputSizes, inputSizes.Length,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
                            }
                        }
                    }
                }

                [DllImport("LibTorchSharp")]
                static extern IntPtr THSTensor_upsample_nearest3d_backward(IntPtr grad_output,
                        IntPtr outputSize, int outputSizeLength,
                        IntPtr inputSize, int inputSizeLength,
                        IntPtr scaleFactors, int scaleFactorsLength);

                public static Tensor upsample_nearest3d_backward(Tensor grad_output, long[] inputSizes, long[] outputSizes = null, double[] scaleFactors = null)
                {
                    var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
                    var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
                    unsafe {
                        fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                            fixed (double* pscaleFactors = scaleFactors) {
                                var res =
                                    THSTensor_upsample_nearest3d_backward(grad_output.handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pinputSizes, inputSizes.Length,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
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
                /// <param name="input">The input tensor.</param>
                /// <param name="outputSizes"></param>
                /// <param name="scaleFactors"></param>
                /// <returns></returns>
                public static Tensor upsample_nearest3d(Tensor input, long[] outputSizes = null, double[] scaleFactors = null)
                {
                    var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
                    var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
                    unsafe {
                        fixed (long* poutputSizes = outputSizes) {
                            fixed (double* pscaleFactors = scaleFactors) {
                                var res =
                                    THSTensor_upsample_nearest3d(input.handle,
                                        (IntPtr)poutputSizes, outputSizesLength,
                                        (IntPtr)pscaleFactors, scaleFactorsLength);
                                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                                return new Tensor(res);
                            }
                        }
                    }
                }
            }
        }
    }
}