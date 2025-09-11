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
        public sealed class Conv1d : Convolution
        {
            internal long _dimension, _in_channel, _out_channel, _kernel,_stride, _padding,_dilation,_groups;
            internal PaddingModes _paddingModes;
            internal (long, long)? _kernels, _strides, _paddings, _dilations;
            internal bool _bias;
            protected Convolution(IntPtr handle, IntPtr boxedHandle, long input_channels) : base(handle, boxedHandle)
            {
                this.input_channels = input_channels;
            }

            public override Tensor forward(Tensor input)
            {
                if (!ValidateShape(input, 1)) 
                    throw new ArgumentException($"Expected 2D (unbatched) or 3D (batched) input with {in_channels} channels to Conv1d.");

                if (padding_mode != PaddingModes.Zeros) {
                    using var paddedInput = torch.nn.functional.pad(input, _reversed_padding_repeated_twice, padding_mode);
                    return torch.nn.functional.conv1d(paddedInput, weight, bias, stride[0], 0, dilation[0], groups);
                }

                if (padding_type.HasValue)
                    return torch.nn.functional.conv1d_padding(input, weight, bias, stride[0], padding_type.Value, dilation[0], groups);

                return torch.nn.functional.conv1d(input, weight, bias, stride[0], padding?[0], dilation[0], groups);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 1D convolution over an input signal composed of several input planes.
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
            /// <returns>Tensor of shape (N,C_out,L_out)</returns>
            public static Conv1d Conv1d(long in_channels, long out_channels, long kernel_size, long stride = 1, long padding = 0, long dilation = 1, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Conv1d_ctor(in_channels, out_channels, kernelSize, stride, padding, dilation, (long)padding_mode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv1d(res, boxedHandle, in_channels) {
                    _in_channel = in_channels,
                    _out_channel = out_channels,
                    _kernel = kernelSize,
                    _stride = stride,
                    _padding = padding,
                    _dilation = dilation,
                    _paddingModes = padding_mode,
                    _groups = groups,
                    _bias = bias
                }.MoveModule<Conv1d>(device, dtype);
            }

            /// <summary>
            /// Applies a 1D convolution over an input signal composed of several input planes.
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
            /// <returns>Tensor of shape (N,C_out,L_out)</returns>
            public static Conv1d Conv1d(long in_channels, long out_channels, long kernel_size, Padding padding, long stride = 1, long dilation = 1, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Conv1d_ctor(in_channels, out_channels, kernelSize, stride, padding == Padding.Valid ? 0 : -1, dilation, (long)padding_mode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv1d(res, boxedHandle, in_channels) {
                    _in_channel = in_channels,
                    _out_channel = out_channels,
                    _kernel = kernelSize,
                    _stride = stride,
                    _padding = (long)padding,
                    _dilation = dilation,
                    _paddingModes = padding_mode,
                    _groups = groups,
                    _bias = bias
                }.MoveModule<Conv1d>(device, dtype);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 1D convolution over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="weight">weight matrix of the convolution</param>
                /// <param name="bias">Optional; bias vector of the convolution</param>
                /// <param name="stride">Stride of the convolution. Default: (1,)</param>
                /// <param name="padding">Zero-padding added to both sides of the input. Default: (0,)</param>
                /// <param name="dilation">Spacing between kernel elements. Default: (1,)</param>
                /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
                /// <returns></returns>
                public static Tensor conv1d(Tensor input, Tensor weight, Tensor? bias = null,
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
                                THSTensor_conv1d(input.Handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, paddingArray.Length,
                                    (IntPtr)pdilation, dilationArray.Length,
                                    groups);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            res = AutocastMode.AutoCast(res);
                            return new Tensor(res);
                        }
                    }
                }

                /// <summary>
                /// Applies a 1D convolution over an input signal composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="weight">weight matrix of the convolution</param>
                /// <param name="bias">Optional; bias vector of the convolution</param>
                /// <param name="stride">Stride of the convolution. Default: (1,)</param>
                /// <param name="padding">Zero-padding added to both sides of the input. padding=Valid is the same as no padding. padding=Same pads the input so the output has the shape as the input. </param>
                /// <param name="dilation">Spacing between kernel elements. Default: (1,)</param>
                /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
                /// <returns></returns>
                public static Tensor conv1d_padding(Tensor input, Tensor weight, Tensor? bias = null,
                    long? stride = null,
                    Padding padding = Padding.Valid,
                    long? dilation = null,
                    long groups = 1)
                {
                    var strides = new long[] { stride ?? 1 };
                    var dilationArray = new long[] { dilation ?? 1 };
                    var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
                    unsafe {
                        fixed (long* pstrides = strides, pdilation = dilationArray) {
                            var res =
                                THSTensor_conv1d_padding(input.Handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (int)padding,
                                    (IntPtr)pdilation, dilationArray.Length,
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
