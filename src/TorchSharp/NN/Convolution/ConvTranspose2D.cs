// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class ConvTranspose2d : torch.nn.Module<Tensor, Tensor>
        {
            internal ConvTranspose2d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_ConvTranspose2d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Parameter? bias {
                get {
                    var res = THSNN_ConvTranspose2d_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    THSNN_ConvTranspose2d_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }
            public Parameter? weight
                {
                get {
                    var res = THSNN_ConvTranspose2d_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_ConvTranspose2d_set_weight(handle, value is null ? IntPtr.Zero : value.Handle);
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("weight", value);
                }
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
            /// <param name="inputChannel">Number of channels in the input image</param>
            /// <param name="outputChannel">Number of channels produced by the convolution</param>
            /// <param name="kernelSize">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: 1</param>
            /// <param name="padding">Zero-padding added to both sides of the input. Default: 0</param>
            /// <param name="outputPadding">Additional size added to one side of the output shape. Default: 0</param>
            /// <param name="dilation">Spacing between kernel elements. Default: 1</param>
            /// <param name="paddingMode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns>Tensor of shape (N,C_out,L_out)</returns>
            public static ConvTranspose2d ConvTranspose2d(long inputChannel, long outputChannel, long kernelSize, long stride = 1, long padding = 0, long outputPadding = 0, long dilation = 1, PaddingModes paddingMode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_ConvTranspose2d_ctor(inputChannel, outputChannel, kernelSize, stride, padding, outputPadding, dilation, (long)paddingMode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new ConvTranspose2d(res, boxedHandle).MoveModule<ConvTranspose2d>(device, dtype);
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
                /// <param name="outputPadding"></param>
                /// <param name="dilation"></param>
                /// <param name="groups"></param>
                /// <returns></returns>
                public static Tensor conv_transpose2d(Tensor input, Tensor weight, Tensor? bias = null,
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
                                THSTensor_conv_transpose2d(input.Handle, weight.Handle, biasHandle,
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

            }
        }
    }
}
