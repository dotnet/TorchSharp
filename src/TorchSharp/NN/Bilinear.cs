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
        public class Bilinear : torch.nn.Module
        {
            internal Bilinear(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public new static Bilinear Load(String modelPath)
            {
                var res = Module.Load(modelPath);
                return new Bilinear(res.handle.DangerousGetHandle(), IntPtr.Zero);
            }

            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Bilinear_forward(torch.nn.Module.HType module, IntPtr input1, IntPtr input2);

            public override Tensor forward(Tensor input1, Tensor input2)
            {
                var res = THSNN_Bilinear_forward(handle, input1.Handle, input2.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Bilinear_bias(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            extern static void THSNN_Bilinear_set_bias(torch.nn.Module.HType module, IntPtr tensor);

            public Parameter? bias {
                get {
                    var res = THSNN_Bilinear_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    THSNN_Bilinear_set_bias(handle, (value is null ? IntPtr.Zero : value.Handle));
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_Bilinear_weight(torch.nn.Module.HType module);
            [DllImport("LibTorchSharp")]
            extern static void THSNN_Bilinear_set_weight(torch.nn.Module.HType module, IntPtr tensor);

            public Parameter weight {
                get {
                    var res = THSNN_Bilinear_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Parameter(res);
                }
                set {
                    THSNN_Bilinear_set_weight(handle, value.Handle);
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
            private static extern IntPtr THSNN_Bilinear_ctor(long in1_features, long in2_features, long output_size, bool bias, out IntPtr pBoxedModule);

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_functional_bilinear(IntPtr input1, IntPtr input2, IntPtr weights, IntPtr bias);

            /// <summary>
            /// Applies a bilinear transformation to the incoming data
            /// </summary>
            /// <param name="in1Features">size of each first input sample</param>
            /// <param name="in2Features">size of each second input sample</param>
            /// <param name="outputSize">size of each output sample</param>
            /// <param name="hasBias">If set to false, the layer will not learn an additive bias</param>
            /// <returns></returns>
            static public Bilinear Bilinear(long in1Features, long in2Features, long outputSize, bool hasBias = true)
            {
                var res = THSNN_Bilinear_ctor(in1Features, in2Features, outputSize, hasBias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Bilinear(res, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Applies a bilinear transformation to the incoming data
                /// </summary>
                /// <param name="input1">Input tensor of shape (N,*,H1)</param>
                /// <param name="input2">Input tensor of shape (N,*,H2)</param>
                /// <param name="weight">Weights of shape (Hout,H1, H2)</param>
                /// <param name="bias">Optional bias of shape (Hout)</param>
                /// <returns>Tensor of shape (N,*,Hout)</returns>
                /// <remarks>The '*' sub-shape must be the same among the two inputs.</remarks>
                static public Tensor bilinear(Tensor input1, Tensor input2, Modules.Parameter weight, Modules.Parameter? bias = null)
                {
                    IntPtr bPtr = bias is null ? IntPtr.Zero : bias.Handle;
                    var res = THSNN_functional_bilinear(input1.Handle, input2.Handle, weight.Handle, bPtr);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}