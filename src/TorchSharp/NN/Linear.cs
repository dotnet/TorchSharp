// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class Linear : torch.nn.Module<Tensor, Tensor>
        {
            internal Linear(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public long in_features {
                get {
                    var weight = this.weight;
                    Debug.Assert(weight is not null);
                    return weight.size(1);
                }
            }

            public long out_features {
                get {
                    var weight = this.weight;
                    Debug.Assert(weight is not null);
                    return weight.size(0);
                }
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Linear_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public Parameter? bias {
                get {
                    var res = THSNN_Linear_bias(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return ((res == IntPtr.Zero) ? null : new Parameter(res));
                }
                set {
                    if (value is null) throw new ArgumentNullException("bias cannot be set to 'null'");
                    THSNN_Linear_set_bias(handle, value?.Handle ?? IntPtr.Zero);
                    torch.CheckForErrors();
                    ConditionallyRegisterParameter("bias", value);
                }
            }

            public Parameter? weight {
                get {
                    var res = THSNN_Linear_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    if (value is null) throw new ArgumentNullException("weight cannot be set to 'null'");
                    THSNN_Linear_set_weight(handle, value!.Handle);
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
            /// Applies a linear transformation to the incoming data.
            /// </summary>
            /// <param name="inputSize">Size of each input sample</param>
            /// <param name="outputSize">Size of each output sample</param>
            /// <param name="hasBias">If set to false, the layer will not learn an additive bias.</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            public static Linear Linear(long inputSize, long outputSize, bool hasBias = true, Device? device = null, ScalarType? dtype = null)
            {
                var res = THSNN_Linear_ctor(inputSize, outputSize, hasBias, out var boxedHandle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }

                return new Linear(res, boxedHandle).MoveModule<Linear>(device, dtype);
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
                public static Tensor linear(Tensor input, Tensor weights, Tensor? bias = null)
                {
                    IntPtr bPtr = bias?.Handle ?? IntPtr.Zero;
                    var res = THSNN_functional_linear(input.Handle, weights.Handle, bPtr);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
