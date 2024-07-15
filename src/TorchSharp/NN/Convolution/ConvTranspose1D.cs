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
        public sealed class ConvTranspose1d : ConvolutionTranspose
        {
            internal ConvTranspose1d(long in_channels, long out_channels, long kernel_size, long stride, long padding, long dilation, long output_padding, long groups = 1, bool bias = true, PaddingModes padding_mode = PaddingModes.Zeros, torch.Device? device = null, ScalarType? dtype = null)
                        : base(nameof(ConvTranspose1d), in_channels, out_channels, new[] { kernel_size }, new[] { stride }, new[] { padding }, null, new[] { dilation }, true, new[] { output_padding }, groups, bias, padding_mode, device, dtype) { }

            public override Tensor forward(Tensor input, long[]? output_size)
            {
                if (!ValidateShape(input, 1))
                    throw new ArgumentException($"Expected 2D (unbatched) or 3D (batched) input with {in_channels} channels to ConvTranspose1d.");

                var output_padding = this._output_padding(input, output_size, kernel_size, stride, padding!, dilation, 1);
                return torch.nn.functional.conv_transpose1d(input, weight, bias, stride[0], padding![0], output_padding[0], dilation[0], groups);
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
            /// <param name="output_padding">Additional size added to one side of the output shape. Default: 0</param>
            /// <param name="dilation">Spacing between kernel elements. Default: 1</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns>Tensor of shape (N,C_out,L_out)</returns>
            public static ConvTranspose1d ConvTranspose1d(long in_channels, long out_channels, long kernel_size, long stride = 1, long padding = 0, long output_padding = 0, long dilation = 1, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                return new ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding, groups, bias, padding_mode, device, dtype);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a 1D transposed convolution operator over an input signal composed of several input planes, sometimes also called “deconvolution”.
                /// </summary>
                /// <param name="input">The input tensor.</param>
                /// <param name="weight"></param>
                /// <param name="bias"></param>
                /// <param name="stride"></param>
                /// <param name="padding"></param>
                /// <param name="output_padding"></param>
                /// <param name="dilation"></param>
                /// <param name="groups"></param>
                /// <returns></returns>
                public static Tensor conv_transpose1d(Tensor input, Tensor weight, Tensor? bias = null,
                    long? stride = null,
                    long? padding = null,
                    long? output_padding = null,
                    long? dilation = null,
                    long groups = 1)
                {
                    var strides = new long[] { stride ?? 1 };
                    var paddings = new long[] { padding ?? 0 };
                    var outputPaddings = new long[] { output_padding ?? 0 };
                    var dilations = new long[] { dilation ?? 1 };
                    var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
                    unsafe {
                        fixed (long* pstrides = strides, ppadding = paddings, poutputPadding = outputPaddings, pdilation = dilations) {
                            var res =
                                THSTensor_conv_transpose1d(input.Handle, weight.Handle, biasHandle,
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

            }
        }
    }
}
