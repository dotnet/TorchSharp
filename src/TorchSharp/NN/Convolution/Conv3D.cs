// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class Conv3d : torch.nn.Module<Tensor, Tensor>
        {
            internal Conv3d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Conv3d_forward(torch.nn.Module<Tensor, Tensor>.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Conv3d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Conv3d_bias(torch.nn.Module<Tensor, Tensor>.HType module);
            [DllImport("LibTorchSharp")]
            extern static void THSNN_Conv3d_set_bias(torch.nn.Module<Tensor, Tensor>.HType module, IntPtr tensor);

            public Parameter? bias {
                get {
                    var res = THSNN_Conv3d_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    THSNN_Conv3d_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Conv3d_weight(torch.nn.Module<Tensor, Tensor>.HType module);
            [DllImport("LibTorchSharp")]
            extern static void THSNN_Conv3d_set_weight(torch.nn.Module<Tensor, Tensor>.HType module, IntPtr tensor);

            public Parameter? weight {
                get {
                    var res = THSNN_Conv3d_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_Conv3d_set_weight(handle, (value is null ? IntPtr.Zero : value.Handle));
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
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Conv3d_ctor(long inputChannel, long outputChannel, long kernelSize, long stride, long padding, long dilation, long paddingMode, long groups, bool bias, out IntPtr pBoxedModule);

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_Conv3d_ctor_1(long inputChannel, long outputChannel, long kernelSizeX, long kernelSizeY, long kernelSizeZ, long strideX, long strideY, long strideZ, long paddingX, long paddingY, long paddingZ, long dilationX, long dilationY, long dilationZ, long paddingMode, long groups, bool bias, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies a 3D convolution over an input signal composed of several input planes
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
            static public Conv3d Conv3d(long inputChannel, long outputChannel, long kernelSize, long stride = 1, long padding = 0, long dilation = 1, PaddingModes paddingMode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Conv3d_ctor(inputChannel, outputChannel, kernelSize, stride, padding, dilation, (long)paddingMode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv3d(res, boxedHandle).MoveModule<Conv3d>(device, dtype);
            }

            /// <summary>
            /// Applies a 3D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="inputChannel">Number of channels in the input image</param>
            /// <param name="outputChannel">Number of channels produced by the convolution</param>
            /// <param name="kernelSize">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: (1,1,1)</param>
            /// <param name="padding">Zero-padding added to both sides of the input. Default: (0,0,0)</param>
            /// <param name="dilation">Spacing between kernel elements. Default: (1,1,1)</param>
            /// <param name="paddingMode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            static public Conv3d Conv3d(long inputChannel, long outputChannel, (long, long, long) kernelSize, (long, long, long)? stride = null, (long, long, long)? padding = null, (long, long, long)? dilation = null, PaddingModes paddingMode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                if (stride == null) stride = (1, 1, 1);
                if (padding == null) padding = (0, 0, 0);
                if (dilation == null) dilation = (1, 1, 1);

                var res = THSNN_Conv3d_ctor_1(inputChannel, outputChannel, kernelSize.Item1, kernelSize.Item2, kernelSize.Item3, stride.Value.Item1, stride.Value.Item2, stride.Value.Item3, padding.Value.Item1, padding.Value.Item2, padding.Value.Item3, dilation.Value.Item1, dilation.Value.Item2, dilation.Value.Item3, (long)paddingMode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv3d(res, boxedHandle).MoveModule<Conv3d>(device, dtype);
            }

            /// <summary>
            /// Applies a 3D convolution over an input signal composed of several input planes
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
            static public Conv3d Conv3d(long inputChannel, long outputChannel, long kernelSize, Padding padding, long stride = 1, long dilation = 1, PaddingModes paddingMode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Conv3d_ctor(inputChannel, outputChannel, kernelSize, stride, padding == Padding.Valid ? 0 : -1, dilation, (long)paddingMode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv3d(res, boxedHandle).MoveModule<Conv3d>(device, dtype);
            }

            /// <summary>
            /// Applies a 3D convolution over an input signal composed of several input planes
            /// </summary>
            /// <param name="inputChannel">Number of channels in the input image</param>
            /// <param name="outputChannel">Number of channels produced by the convolution</param>
            /// <param name="kernelSize">Size of the convolving kernel</param>
            /// <param name="stride">Stride of the convolution. Default: (1,1,1)</param>
            /// <param name="padding">Zero-padding added to both sides of the input. padding=Valid is the same as no padding. padding=Same pads the input so the output has the shape as the input. </param>
            /// <param name="dilation">Spacing between kernel elements. Default: (1,1,1)</param>
            /// <param name="paddingMode">'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'</param>
            /// <param name="groups">Number of blocked connections from input channels to output channels. Default: 1</param>
            /// <param name="bias">If true, adds a learnable bias to the output. Default: true</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            static public Conv3d Conv3d(long inputChannel, long outputChannel, (long, long, long) kernelSize, Padding padding, (long, long, long)? stride = null, (long, long, long)? dilation = null, PaddingModes paddingMode = PaddingModes.Zeros, long groups = 1, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                if (stride == null) stride = (1, 1, 1);
                if (dilation == null) dilation = (1, 1, 1);

                var res = THSNN_Conv3d_ctor_1(inputChannel, outputChannel, kernelSize.Item1, kernelSize.Item2, kernelSize.Item3, stride.Value.Item1, stride.Value.Item2, stride.Value.Item3, padding == Padding.Valid ? 0 : -1, 0, 0, dilation.Value.Item1, dilation.Value.Item2, dilation.Value.Item3, (long)paddingMode, groups, bias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Conv3d(res, boxedHandle).MoveModule<Conv3d>(device, dtype);
            }
        }
    }
}
