// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.PInvoke.LibTorchSharp;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class Bilinear : Module<Tensor, Tensor, Tensor>
        {
            internal Bilinear(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public new static Bilinear Load(string modelPath)
            {
                var res = Module<Tensor, Tensor>.Load(modelPath);
                return new Bilinear(res.handle.DangerousGetHandle(), IntPtr.Zero);
            }

            protected override Tensor forward(Tensor input1, Tensor input2)
            {
                var res = THSNN_Bilinear_forward(handle, input1.Handle, input2.Handle);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res);
            }

            public Parameter? bias {
                get {
                    var res = THSNN_Bilinear_bias(handle);
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    THSNN_Bilinear_set_bias(handle, value?.Handle ?? IntPtr.Zero);
                    CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }

            public Parameter? weight {
                get {
                    var res = THSNN_Bilinear_weight(handle);
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    THSNN_Bilinear_set_weight(handle, value?.Handle ?? IntPtr.Zero);
                    CheckForErrors();
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
            /// Applies a bilinear transformation to the incoming data
            /// </summary>
            /// <param name="in1Features">size of each first input sample</param>
            /// <param name="in2Features">size of each second input sample</param>
            /// <param name="outputSize">size of each output sample</param>
            /// <param name="hasBias">If set to false, the layer will not learn an additive bias</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static Bilinear Bilinear(long in1Features, long in2Features, long outputSize, bool hasBias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Bilinear_ctor(in1Features, in2Features, outputSize, hasBias, out var boxedHandle);
                if (res == IntPtr.Zero) { CheckForErrors(); }

                return new Bilinear(res, boxedHandle).MoveModule<Bilinear>(device, dtype);
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
                public static Tensor bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias = null)
                {
                    IntPtr bPtr = bias?.Handle ?? IntPtr.Zero;
                    var res = THSNN_functional_bilinear(input1.Handle, input2.Handle, weight.Handle, bPtr);
                    if (res == IntPtr.Zero) { CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}