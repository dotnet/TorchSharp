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

        /// <summary>
        /// This class is used to represent a LayerNorm module.
        /// </summary>
        public class LayerNorm : torch.nn.Module<Tensor, Tensor>
        {
            internal LayerNorm(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LayerNorm_forward(IntPtr module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_LayerNorm_forward(handle.DangerousGetHandle(), tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LayerNorm_bias(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            private static extern void THSNN_LayerNorm_set_bias(torch.nn.Module.HType module, IntPtr bias);
            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_LayerNorm_weight(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            private static extern void THSNN_LayerNorm_set_weight(torch.nn.Module.HType module, IntPtr weight);

            public Parameter? bias {
                get {
                    var res = THSNN_LayerNorm_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_LayerNorm_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }

            public Parameter? weight {
                get {
                    var res = THSNN_LayerNorm_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_LayerNorm_set_weight(handle, value is null ? IntPtr.Zero : value.Handle);
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
            extern static IntPtr THSNN_LayerNorm_ctor(IntPtr norm_shape, long norm_shape_len, double eps, [MarshalAs(UnmanagedType.U1)] bool elementwise_affine, out IntPtr pBoxedModule);

            /// <summary>
            /// Applies Layer Normalization over a mini-batch of inputs as described in the paper Layer Normalization
            /// </summary>
            /// <param name="normalized_shape">Input shape from an expected input.</param>
            /// <param name="eps">A value added to the denominator for numerical stability. Default: 1e-5</param>
            /// <param name="elementwise_affine">a boolean value that when set to true, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases).</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            static public LayerNorm LayerNorm(long[] normalized_shape, double eps = 1e-05, bool elementwise_affine = true, Device? device = null, ScalarType? dtype = null)
            {
                unsafe {
                    fixed (long* pNormShape = normalized_shape) {
                        var handle = THSNN_LayerNorm_ctor((IntPtr)pNormShape, normalized_shape.Length, eps, elementwise_affine, out var boxedHandle);
                        if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new LayerNorm(handle, boxedHandle).MoveModule<LayerNorm>(device, dtype);
                    }
                }
            }

            /// <summary>
            /// Applies Layer Normalization over a mini-batch of inputs as described in the paper Layer Normalization
            /// </summary>
            /// <param name="normalized_shape">Input shape from an expected input.</param>
            /// <param name="eps">A value added to the denominator for numerical stability. Default: 1e-5</param>
            /// <param name="elementwise_affine">a boolean value that when set to true, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases).</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            static public LayerNorm LayerNorm(long normalized_shape, double eps = 1e-05, bool elementwise_affine = true, Device? device = null, ScalarType? dtype = null)
            {
                return LayerNorm(new[] { normalized_shape }, eps, elementwise_affine, device, dtype);
            }
        }
    }
}
