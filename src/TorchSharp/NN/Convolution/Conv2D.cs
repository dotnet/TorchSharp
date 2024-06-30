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
        public sealed class Conv2d : Convolution
        {
            internal Conv2d(IntPtr handle, IntPtr boxedHandle, long input_channels) : base(handle, boxedHandle, input_channels) { }

            public override Tensor forward(Tensor input)
            {
                if (ValidateShape(input, 2)) {
                    var res = THSNN_Conv2d_forward(handle, input.Handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
                throw new ArgumentException($"Expected 3D (unbatched) or 4D (batched) input with {input_channels} channels to Conv2d.");
            }

            public Parameter? bias {
                get {
                    var res = THSNN_Conv2d_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    // Please ignore, for now, that the litorch call thinks you *can* set it to null.
                    if (value is null) throw new ArgumentNullException("bias cannot be set to 'null'");
                    THSNN_Conv2d_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }
            public Parameter? weight {
                get {
                    var res = THSNN_Conv2d_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    // Please ignore, for now, that the litorch call thinks you *can* set it to null.
                    if (value is null) throw new ArgumentNullException("weight cannot be set to 'null'");
                    THSNN_Conv2d_set_weight(handle, value is null ? IntPtr.Zero : value.Handle);
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
            /// Applies a 2D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernelSize">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: 1</param>
            /// <param name="padding">Zero-padding added to both sides of the input. Default: 0</param>
            /// <param name="dilation">Spacing between kernel elements. Default: 1</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static Conv2d Conv2d(long in_channels, long out_channels, long kernelSize, long stride = 1, long padding = 0, long dilation = 1, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Conv2d_ctor(in_channels, out_channels, kernelSize, stride, padding, dilation, (long)padding_mode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv2d(res, boxedHandle, in_channels).MoveModule<Conv2d>(device, dtype);
            }

            /// <summary>
            /// Applies a 2D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernelSize">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: (1,1)</param>
            /// <param name="padding">Zero-padding added to both sides of the input. Default: (0,0)</param>
            /// <param name="dilation">Spacing between kernel elements. Default: (1,1)</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static Conv2d Conv2d(long in_channels, long out_channels, (long, long) kernelSize, (long, long)? stride = null, (long, long)? padding = null, (long, long)? dilation = null, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                if (stride == null) stride = (1, 1);
                if (padding == null) padding = (0, 0);
                if (dilation == null) dilation = (1, 1);

                var res = THSNN_Conv2d_ctor_1(in_channels, out_channels, kernelSize.Item1, kernelSize.Item2, stride.Value.Item1, stride.Value.Item2, padding.Value.Item1, padding.Value.Item2, dilation.Value.Item1, dilation.Value.Item2, (long)padding_mode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv2d(res, boxedHandle, in_channels).MoveModule<Conv2d>(device, dtype);
            }

            /// <summary>
            /// Applies a 2D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernelSize">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: 1</param>
            /// <param name="padding">Zero-padding added to both sides of the input. padding=Valid is the same as no padding. padding=Same pads the input so the output has the shape as the input. </param>
            /// <param name="dilation">Spacing between kernel elements. Default: 1</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static Conv2d Conv2d(long in_channels, long out_channels, long kernelSize, Padding padding, long stride = 1, long dilation = 1, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Conv2d_ctor(in_channels, out_channels, kernelSize, stride, padding == Padding.Valid ? 0 : -1, dilation, (long)padding_mode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv2d(res, boxedHandle, in_channels).MoveModule<Conv2d>(device, dtype);
            }

            /// <summary>
            /// Applies a 2D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="in_channels">Number of channels in the input image</param>
            /// <param name="out_channels">Number of channels produced by the convolution</param>
            /// <param name="kernelSize">Size of the convolving kernel</param>
            /// <param name="padding">Zero-padding added to both sides of the input. padding=Valid is the same as no padding. padding=Same pads the input so the output has the shape as the input. </param>
            /// <param name="stride">Stride of the convolution. Default: (1,1)</param>
            /// <param name="dilation">Spacing between kernel elements. Default: (1,1)</param>
            /// <param name="padding_mode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static Conv2d Conv2d(long in_channels, long out_channels, (long, long) kernelSize, Padding padding, (long, long)? stride = null, (long, long)? dilation = null, PaddingModes padding_mode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                if (stride == null) stride = (1, 1);
                if (dilation == null) dilation = (1, 1);

                var res = THSNN_Conv2d_ctor_1(in_channels, out_channels, kernelSize.Item1, kernelSize.Item2, stride.Value.Item1, stride.Value.Item2, padding == Padding.Valid ? 0 : -1, 0, dilation.Value.Item1, dilation.Value.Item2, (long)padding_mode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv2d(res, boxedHandle, in_channels).MoveModule<Conv2d>(device, dtype);
            }

            public static partial class functional
            {
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
                public static Tensor conv2d(Tensor input, Tensor weight, Tensor? bias = null,
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
                                THSTensor_conv2d(input.Handle, weight.Handle, biasHandle,
                                    (IntPtr)pstrides, strides.Length,
                                    (IntPtr)ppadding, padding.Length,
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
