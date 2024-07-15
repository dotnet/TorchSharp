// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class Conv3d : Convolution
        {
            internal Conv3d(long in_channels, long out_channels, (long, long, long) kernel_size, (long, long, long) stride, (long, long, long)? padding, Padding? padding_type, (long, long, long) dilation, long groups = 1, bool bias = true, PaddingModes padding_mode = PaddingModes.Zeros, torch.Device? device = null, ScalarType? dtype = null)
                        : base(nameof(Conv3d), in_channels, out_channels, new[] { kernel_size.Item1, kernel_size.Item2, kernel_size.Item3 }, new[] { stride.Item1, stride.Item2, stride.Item3 }, padding.HasValue ? new[] { padding.Value.Item1, padding.Value.Item2, padding.Value.Item3 } : null, padding_type, new[] { dilation.Item1, dilation.Item2, dilation.Item3 }, false, new[] { 0, 0, 0L }, groups, bias, padding_mode, device, dtype) { }

            public override Tensor forward(Tensor input)
            {
                if (!ValidateShape(input, 3))
                    throw new ArgumentException($"Expected 4D (unbatched) or 5D (batched) input with {in_channels} channels to Conv3d.");

                if (padding_mode != PaddingModes.Zeros) {
                    using var paddedInput = torch.nn.functional.pad(input, _reversed_padding_repeated_twice, padding_mode);
                    return torch.nn.functional.conv3d(paddedInput, weight, bias, stride, new[] { 0L, 0L, 0L }, dilation, groups);
                }

                if (padding_type.HasValue)
                    return torch.nn.functional.conv3d_padding(input, weight, bias, stride, padding_type.Value, dilation, groups);

                return torch.nn.functional.conv3d(input, weight, bias, stride, padding, dilation, groups);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 3D convolution over an input signal composed of several input planes
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
            public static Conv3d Conv3d(long in_channels, long out_channels, long kernel_size, long stride = 1, long padding = 0, long dilation = 1, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                return new Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), (stride, stride, stride), (padding, padding, padding), null, (dilation, dilation, dilation), groups, bias, padding_mode, device, dtype);
            }

            /// <summary>
            /// Applies a 3D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernel_size">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: (1,1,1)</param>
            /// <param name="padding">Zero-padding added to both sides of the input. Default: (0,0,0)</param>
            /// <param name="dilation">Spacing between kernel elements. Default: (1,1,1)</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            public static Conv3d Conv3d(long in_channels, long out_channels, (long, long, long) kernel_size, (long, long, long)? stride = null, (long, long, long)? padding = null, (long, long, long)? dilation = null, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                stride ??= (1, 1, 1);
                padding ??= (0, 0, 0);
                dilation ??= (1, 1, 1);

                return new Conv3d(in_channels, out_channels, kernel_size, stride.Value, padding, null, dilation.Value, groups, bias, padding_mode, device, dtype);
            }

            /// <summary>
            /// Applies a 3D convolution over an input signal composed of several input planes
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
            public static Conv3d Conv3d(long in_channels, long out_channels, long kernel_size, Padding padding, long stride = 1, long dilation = 1, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                return new Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), (stride, stride, stride), null, padding, (dilation, dilation, dilation), groups, bias, padding_mode, device, dtype);
            }

            /// <summary>
            /// Applies a 3D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernel_size">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: (1,1,1)</param>
            /// <param name="padding">Zero-padding added to both sides of the input. padding=Valid is the same as no padding. padding=Same pads the input so the output has the shape as the input. </param>
            /// <param name="dilation">Spacing between kernel elements. Default: (1,1,1)</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            public static Conv3d Conv3d(long in_channels, long out_channels, (long, long, long) kernel_size, Padding padding, (long, long, long)? stride = null, (long, long, long)? dilation = null, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                stride ??= (1, 1, 1);
                dilation ??= (1, 1, 1);
                return new Conv3d(in_channels, out_channels, kernel_size, stride.Value, null, padding, dilation.Value, groups, bias, padding_mode, device, dtype);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 3D convolution over an input image composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="weight">weight matrix of the convolution</param>
                /// <param name="bias">Optional; bias vector of the convolution</param>
                /// <param name="strides">Stride of the convolution. Default: (1,1,1)</param>
                /// <param name="padding">Zero-padding added to both sides of the input. Default: (0,0,0)</param>
                /// <param name="dilation">Spacing between kernel elements. Default: (1,1,1)</param>
                /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
                /// <returns></returns>
                public static Tensor conv3d(Tensor input, Tensor weight, Tensor? bias = null,
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
                                THSTensor_conv3d(input.Handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    (IntPtr)pdilation, dilation.Length,
                                    groups);
                            if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                            return new Tensor(res);
                        }
                    }
                }

                /// <summary>
                /// Applies a 3D convolution over an input image composed of several input planes.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="weight">weight matrix of the convolution</param>
                /// <param name="bias">Optional; bias vector of the convolution</param>
                /// <param name="strides">Stride of the convolution. Default: (1,1,1)</param>
                /// <param name="padding">Zero-padding added to both sides of the input. padding=Valid is the same as no padding. padding=Same pads the input so the output has the shape as the input. </param>
                /// <param name="dilation">Spacing between kernel elements. Default: (1,1,1)</param>
                /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
                /// <returns></returns>
                public static Tensor conv3d_padding(Tensor input, Tensor weight, Tensor? bias = null,
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
                                THSTensor_conv3d_padding(input.Handle, weight.Handle, biasHandle,
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
