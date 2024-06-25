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
        public sealed class ConvTranspose2d : ConvolutionTranspose
        {
            internal ConvTranspose2d(long in_channels, long out_channels, (long, long) kernel_size, (long, long) stride, (long, long) padding, (long, long) dilation, (long, long) output_padding, long groups = 1, bool bias = true, PaddingModes padding_mode = PaddingModes.Zeros, torch.Device? device = null, ScalarType? dtype = null)
                        : base(nameof(ConvTranspose2d), in_channels, out_channels, new[] { kernel_size.Item1, kernel_size.Item2 }, new[] { stride.Item1, stride.Item2 }, new[] { padding.Item1, padding.Item2 }, null, new[] { dilation.Item1, dilation.Item2 }, true, new[] { output_padding.Item1, output_padding.Item2 }, groups, bias, padding_mode, device, dtype) { }

            public override Tensor forward(Tensor input, long[]? output_size)
            {
                if (!ValidateShape(input, 2))
                    throw new ArgumentException($"Expected 3D (unbatched) or 4D (batched) input with {in_channels} channels to ConvTranspose2d.");

                var output_padding = this._output_padding(input, output_size, kernel_size, stride, padding!, dilation, 2);
                return torch.nn.functional.conv_transpose2d(input, weight, bias, stride, padding!, output_padding, dilation, groups);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies a 2D transposed convolution over an input signal composed of several input planes.
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernel_size">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: 1</param>
            /// <param name="padding">Zero-padding added to both sides of the input. Default: 0</param>
            /// <param name="output_padding">Additional size added to one side of the output shape. Default: 0</param>
            /// <param name="dilation">Spacing between kernel elements. Default: 1</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns>Tensor of shape (N,C_out,L_out)</returns>
            public static ConvTranspose2d ConvTranspose2d(long in_channels, long out_channels, long kernel_size, long stride = 1, long padding = 0, long output_padding = 0, long dilation = 1, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                return new ConvTranspose2d(in_channels, out_channels, (kernel_size, kernel_size), (stride, stride), (padding, padding), (dilation, dilation), (output_padding, output_padding), groups, bias, padding_mode, device, dtype);
            }



            /// <summary>
            /// Applies a 2D transposed convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernel_size">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: (1,1)</param>
            /// <param name="padding">Zero-padding added to both sides of the input. Default: (0,0)</param>
            /// <param name="output_padding">Additional size added to one side of the output shape. Default: 0</param>
            /// <param name="dilation">Spacing between kernel elements. Default: (1,1)</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static ConvTranspose2d ConvTranspose2d(long in_channels, long out_channels, (long, long) kernel_size, (long, long)? stride = null, (long, long)? padding = null, (long, long)? output_padding = null, (long, long)? dilation = null, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                stride ??= (1, 1);
                padding ??= (0, 0);
                output_padding ??= (0, 0);
                dilation ??= (1, 1);
                return new ConvTranspose2d(in_channels, out_channels, kernel_size, stride.Value, padding.Value, dilation.Value, output_padding.Value, groups, bias, padding_mode, device, dtype);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 2D transposed convolution operator over an input image composed of several input planes, sometimes also called “deconvolution”.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="weight"></param>
                /// <param name="bias"></param>
                /// <param name="strides"></param>
                /// <param name="padding"></param>
                /// <param name="output_padding"></param>
                /// <param name="dilation"></param>
                /// <param name="groups"></param>
                /// <returns></returns>
                public static Tensor conv_transpose2d(Tensor input, Tensor weight, Tensor? bias = null,
                    long[]? strides = null,
                    long[]? padding = null,
                    long[]? output_padding = null,
                    long[]? dilation = null,
                    long groups = 1)
                {
                    strides = (strides == null) ? new long[] { 1, 1 } : strides;
                    padding = (padding == null) ? new long[] { 0, 0 } : padding;
                    output_padding = (output_padding == null) ? new long[] { 0, 0 } : output_padding;
                    dilation = (dilation == null) ? new long[] { 1, 1 } : dilation;
                    var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
                    unsafe {
                        fixed (long* pstrides = strides, ppadding = padding, poutputPadding = output_padding, pdilation = dilation) {
                            var res =
                                THSTensor_conv_transpose2d(input.Handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
                                    (IntPtr)poutputPadding, output_padding.Length,
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
