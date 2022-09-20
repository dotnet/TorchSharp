// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public class Linear : torch.nn.Module
        {
            internal Linear(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public new static Linear Load(String modelPath)
            {
                var res = Module.Load(modelPath);
                return new Linear(res.handle.DangerousGetHandle(), IntPtr.Zero);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Linear_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Linear_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Linear_bias(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            extern static void THSNN_Linear_set_bias(torch.nn.Module.HType module, IntPtr tensor);
            public Parameter? bias {
                get {
                    var res = THSNN_Linear_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    THSNN_Linear_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Linear_weight(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            extern static void THSNN_Linear_set_weight(torch.nn.Module.HType module, IntPtr tensor);

            public Parameter weight {
                get {
                    var res = THSNN_Linear_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Parameter(res);
                }
                set {
                    THSNN_Linear_set_weight(handle, value.Handle);
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
            private static extern IntPtr THSNN_Linear_ctor(long input_size, long output_size, bool bias, out IntPtr pBoxedModule);

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_functional_linear(IntPtr input, IntPtr weights, IntPtr bias);

            /// <summary>
            /// Applies a linear transformation to the incoming data.
            /// </summary>
            /// <param name="inputSize">Size of each input sample</param>
            /// <param name="outputSize">Size of each output sample</param>
            /// <param name="hasBias">If set to false, the layer will not learn an additive bias.</param>
            /// <returns></returns>
            static public Linear Linear(long inputSize, long outputSize, bool hasBias = true)
            {
                var res = THSNN_Linear_ctor(inputSize, outputSize, hasBias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Linear(res, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a linear transformation to the incoming data.
                /// </summary>
                /// <param name="input">Input tensor of shape (*,Hin)</param>
                /// <param name="weights">Weights of shape (Hout,Hin) or (Hin)</param>
                /// <param name="bias">Bias of shape (Hout) or ()</param>
                /// <returns>A tensor of shape (*,Hout) where '*' is the same as the subshape of the input.</returns>
                static public Tensor linear(Tensor input, Tensor weights, Tensor? bias = null)
                {
                    IntPtr bPtr = bias is null ? IntPtr.Zero : bias.Handle;
                    var res = THSNN_functional_linear(input.Handle, weights.Handle, bPtr);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
