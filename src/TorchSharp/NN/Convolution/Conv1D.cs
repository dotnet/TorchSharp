// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

#nullable enable
namespace TorchSharp
{
    using Modules;

    public enum PaddingModes
    {
        Zeros = 0,
        Reflect = 1,
        Replicate = 2,
        Circular = 3,
        Constant = 4,
    }

    public enum Padding
    {
        Valid = 0,
        Same = 1
    }

    namespace Modules
    {
        public sealed class Conv1d : torch.nn.Module<Tensor, Tensor>
        {
            internal Conv1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Conv1d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Parameter? bias {
                get {
                    var res = THSNN_Conv1d_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    THSNN_Conv1d_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }
            public Parameter? weight {
                get {
                    var res = THSNN_Conv1d_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_Conv1d_set_weight(handle, value is null ? IntPtr.Zero : value.Handle);
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
            /// <param name="dilation">Spacing between kernel elements. Default: 1</param>
            /// <param name="paddingMode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns>Tensor of shape (N,C_out,L_out)</returns>
            public static Conv1d Conv1d(long inputChannel, long outputChannel, long kernelSize, long stride = 1, long padding = 0, long dilation = 1, PaddingModes paddingMode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Conv1d_ctor(inputChannel, outputChannel, kernelSize, stride, padding, dilation, (long)paddingMode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv1d(res, boxedHandle).MoveModule<Conv1d>(device, dtype);
            }

            /// <summary>
            /// Applies a 1D convolution over an input signal composed of several input planes.
            /// </summary>
            /// <param name="inputChannel">Number of channels in the input image</param>
            /// <param name="outputChannel">Number of channels produced by the convolution</param>
            /// <param name="kernelSize">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: 1</param>
            /// <param name="padding">Zero-padding added to both sides of the input. padding=Valid is the same as no padding. padding=Same pads the input so the output has the shape as the input. </param>
            /// <param name="dilation">Spacing between kernel elements. Default: 1</param>
            /// <param name="paddingMode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns>Tensor of shape (N,C_out,L_out)</returns>
            public static Conv1d Conv1d(long inputChannel, long outputChannel, long kernelSize, Padding padding, long stride = 1, long dilation = 1, PaddingModes paddingMode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Conv1d_ctor(inputChannel, outputChannel, kernelSize, stride, padding == Padding.Valid ? 0 : -1, dilation, (long)paddingMode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv1d(res, boxedHandle).MoveModule<Conv1d>(device, dtype);
            }

            public static partial class functional
            {
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
                            return new Tensor(res);
                        }
                    }
                }

            }
        }
    }
}
