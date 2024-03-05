// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;
    using TorchSharp.Utils;

    namespace Modules
    {
        public sealed class Bilinear : Module<Tensor, Tensor, Tensor>
        {
            internal Bilinear(long in1_features, long in2_features, long out_features, bool hasBias = true, Device? device = null, ScalarType? dtype = null) : base(nameof(Bilinear))
            {
                weight = torch.empty(out_features, in1_features, in2_features, device: device, dtype: dtype).AsParameter();
                var bound = 1 / Math.Sqrt(weight!.shape[1]);

                init.uniform_(_weight, -bound, bound);

                if (hasBias) {
                    bias = torch.empty(out_features, device: device, dtype: dtype).AsParameter();
                    var (fanIn, _) = init.CalculateFanInAndFanOut(weight);
                    init.uniform_(_bias, -bound, bound);
                }
                //NOTE: it's important not to call 'RegisterComponents' here.
            }

            public override Tensor forward(Tensor input1, Tensor input2)
            {
                return torch.nn.functional.bilinear(input1, input2, _weight!, _bias);
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    _weight?.Dispose();
                    _bias?.Dispose();
                }
            }

            public Parameter? bias {
                get => _bias;
                set {
                    _bias?.Dispose();
                    _bias = value?.DetachFromDisposeScope() as Parameter;
                    ConditionallyRegisterParameter(nameof(bias), _bias);
                }
            }

            public Parameter weight {
                get => _weight!;
                set {
                    if (value is null) throw new ArgumentNullException(nameof(weight));
                    if (value.Handle != _weight?.Handle) {
                        _weight?.Dispose();
                        _weight = (value.DetachFromDisposeScope() as Parameter)!;
                        ConditionallyRegisterParameter(nameof(weight), _weight);
                    }
                }
            }

            [ComponentName(Name = "bias")]
            private Parameter? _bias;
            [ComponentName(Name = "weight")]
            private Parameter? _weight;
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {

            /// <summary>
            /// Applies a bilinear transformation to the incoming data
            /// </summary>
            /// <param name="in1_features">size of each first input sample</param>
            /// <param name="in2_features">size of each second input sample</param>
            /// <param name="out_features">size of each output sample</param>
            /// <param name="hasBias">If set to false, the layer will not learn an additive bias</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static Bilinear Bilinear(long in1_features, long in2_features, long out_features, bool hasBias = true, Device? device = null, ScalarType? dtype = null)
            {
                return new Bilinear(in1_features, in2_features, out_features, hasBias, device, dtype);
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
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}