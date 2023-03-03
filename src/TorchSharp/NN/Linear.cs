// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        public sealed class Linear : torch.nn.Module<Tensor, Tensor>
        {
            internal Linear(long inputSize, long outputSize, bool hasBias = true, Device device = null, ScalarType? dtype = null) : base(nameof(Linear))
            {
                weight = torch.empty(outputSize, inputSize, device: device, dtype: dtype).AsParameter();
                init.kaiming_uniform_(weight, a: Math.Sqrt(5));
                
                if (hasBias) {
                    bias = torch.empty(outputSize, device: device, dtype: dtype).AsParameter();
                    var (fanIn, _) = init.CalculateFanInAndFanOut(weight);
                    var bound = fanIn > 0 ? 1 / Math.Sqrt(fanIn) : 0;
                    init.uniform_(bias, -bound, bound);
                }
                //NOTE: it's important not to call 'RegisterComponents' here.
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_functional_linear(tensor.Handle, weight!.Handle, bias is not null ? bias.Handle : IntPtr.Zero);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    weight.Dispose();
                    if (bias is not null) bias.Dispose();
                }
            }

            public Parameter bias {
                set {
                    _bias = value;
                    ConditionallyRegisterParameter("bias", bias);
                }
                get => _bias;
            }
            private Parameter _bias;

            public Parameter weight {
                get => _weight;
                set {
                    if (value is null) throw new ArgumentNullException("weight");
                    _weight = value;
                    ConditionallyRegisterParameter("weight", weight);
                }
            }

            private Parameter _weight;

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
            public static Linear Linear(long inputSize, long outputSize, bool hasBias = true, Device device = null, ScalarType? dtype = null)
            {
                return new Linear(inputSize, outputSize, hasBias, device, dtype);
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
                public static Tensor linear(Tensor input, Tensor weights, Tensor bias = null)
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
