// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using TorchSharp.Amp;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class Conv2d : Convolution
        {
            
            internal Conv2d(IntPtr handle, IntPtr boxedHandle, long input_channels) : base(handle, boxedHandle, input_channels) { }

            internal Conv2d(IntPtr handle, IntPtr boxedHandle, long input_channels, long in_channels, long out_channels, long kernelSize, long padding, long stride = 1, long dilation = 1, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true)
                : base(handle, boxedHandle, input_channels)
            {
                _dimension = 2; //because is conv 2D; 2 dimension
                _in_channel = in_channels;
                _out_channel = out_channels;
                _kernel = kernelSize;
                _stride = stride;
                _padding = padding;
                _dilation = dilation;
                _paddingModes = padding_mode;
                _groups = groups;
                _bias = bias;
            }
            internal Conv2d(IntPtr handle, IntPtr boxedHandle, long input_channels, long in_channels, long out_channels, (long, long) kernelSize, Padding padding, (long, long)? stride = null, (long, long)? dilation = null, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true)
                : base(handle, boxedHandle, input_channels)
            {
                _dimension = 2; //because is conv 2D; 2 dimension
                _in_channel = in_channels;
                _out_channel = out_channels;
                _kernels = kernelSize;
                _strides = stride;
                _padding = (long)padding;
                _dilations = dilation;
                _paddingModes = padding_mode;
                _groups = groups;
                _bias = bias;
            }
            public override Tensor forward(Tensor input)
            {
                if (!ValidateShape(input, 2))
                    throw new ArgumentException($"Expected 3D (unbatched) or 4D (batched) input with {in_channels} channels to Conv2d.");

                if (padding_mode != PaddingModes.Zeros) {
                    using var paddedInput = torch.nn.functional.pad(input, _reversed_padding_repeated_twice, padding_mode);
                    return torch.nn.functional.conv2d(paddedInput, weight, bias, stride, new[] { 0L, 0L }, dilation, groups);
                }

                if (padding_type.HasValue)
                    return torch.nn.functional.conv2d_padding(input, weight, bias, stride, padding_type.Value, dilation, groups);

                return torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 2D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernel_size">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: 1</param>
            /// <param name="padding">Zero-padding added to both sides of the input. Default: 0</param>
            /// <param name="dilation">Spacing between kernel elements. Default: 1</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static Conv2d Conv2d(long in_channels, long out_channels, long kernel_size, long stride = 1, long padding = 0, long dilation = 1, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Conv2d_ctor(in_channels, out_channels, kernelSize, stride, padding, dilation, (long)padding_mode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }

                return new Conv2d(res, boxedHandle, in_channels) {
                    _in_channel = in_channels,
                    _out_channel = out_channels,
                    _kernel = kernelSize,
                    _stride = stride,
                    _padding = padding,
                    _dilation = dilation,
                    _paddingModes = padding_mode,
                    _groups = groups,
                    _bias = bias
                }.MoveModule<Conv2d>(device, dtype);
                //return conv2d.MoveModule<Conv2d>(device, dtype);
            }

            /// <summary>
            /// Applies a 2D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernel_size">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: (1,1)</param>
            /// <param name="padding">Zero-padding added to both sides of the input. Default: (0,0)</param>
            /// <param name="dilation">Spacing between kernel elements. Default: (1,1)</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static Conv2d Conv2d(long in_channels, long out_channels, (long, long) kernel_size, (long, long)? stride = null, (long, long)? padding = null, (long, long)? dilation = null, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                stride ??= (1, 1);
                padding ??= (0, 0);
                dilation ??= (1, 1);

                var res = THSNN_Conv2d_ctor_1(in_channels, out_channels, kernelSize.Item1, kernelSize.Item2, stride.Value.Item1, stride.Value.Item2, padding.Value.Item1, padding.Value.Item2, dilation.Value.Item1, dilation.Value.Item2, (long)padding_mode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv2d(res, boxedHandle, in_channels) {
                    _in_channel = in_channels,
                    _out_channel = out_channels,
                    _kernels = kernelSize,
                    _strides = stride,
                    _paddings = padding,
                    _dilations = dilation,
                    _paddingModes = padding_mode,
                    _groups = groups,
                    _bias = bias
                }.MoveModule<Conv2d>(device, dtype);
            }

            /// <summary>
            /// Applies a 2D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernel_size">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: 1</param>
            /// <param name="padding">Zero-padding added to both sides of the input. padding=Valid is the same as no padding. padding=Same pads the input so the output has the shape as the input. </param>
            /// <param name="dilation">Spacing between kernel elements. Default: 1</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static Conv2d Conv2d(long in_channels, long out_channels, long kernel_size, Padding padding, long stride = 1, long dilation = 1, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Conv2d_ctor(in_channels, out_channels, kernelSize, stride, padding == Padding.Valid ? 0 : -1, dilation, (long)padding_mode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv2d(res, boxedHandle, in_channels, in_channels, out_channels, kernelSize, (long)padding, stride, dilation, padding_mode, groups, bias).MoveModule<Conv2d>(device, dtype);
            }

            /// <summary>
            /// Applies a 2D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernel_size">Size of the convolving kernel</param>
            /// <param name="padding">Zero-padding added to both sides of the input. padding=Valid is the same as no padding. padding=Same pads the input so the output has the shape as the input. </param>
            /// <param name="stride">Stride of the convolution. Default: (1,1)</param>
            /// <param name="dilation">Spacing between kernel elements. Default: (1,1)</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static Conv2d Conv2d(long in_channels, long out_channels, (long, long) kernel_size, Padding padding, (long, long)? stride = null, (long, long)? dilation = null, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                stride ??= (1, 1);
                dilation ??= (1, 1);

                var res = THSNN_Conv2d_ctor_1(in_channels, out_channels, kernelSize.Item1, kernelSize.Item2, stride.Value.Item1, stride.Value.Item2, padding == Padding.Valid ? 0 : -1, 0, dilation.Value.Item1, dilation.Value.Item2, (long)padding_mode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                
                return new Conv2d(res, boxedHandle, in_channels, in_channels, out_channels, kernelSize, padding,stride, dilation, padding_mode ,groups,bias).MoveModule<Conv2d>(device, dtype);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 2D convolution over an input image composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="weight">weight matrix of the convolution</param>
                /// <param name="bias">Optional; bias vector of the convolution</param>
                /// <param name="strides">Stride of the convolution. Default: (1,1)</param>
                /// <param name="padding">Zero-padding added to both sides of the input. Default: (0,0)</param>
                /// <param name="dilation">Spacing between kernel elements. Default: (1,1)</param>
                /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
                /// <returns></returns>
                public static Tensor conv2d(Tensor input, Tensor weight, Tensor? bias = null,
                    long[]? strides = null,
                    long[]? padding = null,
                    long[]? dilation = null,
                    long groups = 1)
                {
                    strides ??= new long[] { 1 };
                    padding ??= new long[] { 0 };
                    dilation ??= new long[] { 1 };
                    var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
                    unsafe {
                        fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation) {
                            var res =
                                THSTensor_conv2d(input.Handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    (IntPtr)pdilation, dilation.Length,
                                    groups);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            res = AutocastMode.AutoCast(res);
                            return new Tensor(res);
                        }
                    }
                }

                /// <summary>
                /// Applies a 2D convolution over an input image composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="weight">weight matrix of the convolution</param>
                /// <param name="bias">Optional; bias vector of the convolution</param>
                /// <param name="strides">Stride of the convolution. Default: (1,1)</param>
                /// <param name="padding">Zero-padding added to both sides of the input. padding=Valid is the same as no padding. padding=Same pads the input so the output has the shape as the input. </param>
                /// <param name="dilation">Spacing between kernel elements. Default: (1,1)</param>
                /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
                /// <returns></returns>
                public static Tensor conv2d_padding(Tensor input, Tensor weight, Tensor? bias = null,
                    long[]? strides = null,
                    Padding padding = Padding.Valid,
                    long[]? dilation = null,
                    long groups = 1)
                {
                    strides ??= new long[] { 1 };
                    dilation ??= new long[] { 1 };
                    var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
                    unsafe {
                        fixed (long* pstrides = strides, pdilation = dilation) {
                            var res =
                                THSTensor_conv2d_padding(input.Handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (int)padding,
                                    (IntPtr)pdilation, dilation.Length,
                                    groups);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

            }
        }
    }
}
