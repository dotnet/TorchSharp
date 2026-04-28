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
            const string WeightComponentName = nameof(weight);
            const string BiasComponentName = nameof(bias);

            internal Bilinear(long in1_features, long in2_features, long out_features, bool hasBias = true, Device? device = null, ScalarType? dtype = null) : base(nameof(Bilinear))
            {
                this.in1_features = in1_features;
                this.in2_features = in2_features;
                this.out_features = out_features;

                weight = torch.empty(out_features, in1_features, in2_features, device: device, dtype: dtype).AsParameter();
                var bound = 1 / Math.Sqrt(weight!.shape[1]);

                init.uniform_(_weight, -bound, bound);

                if (hasBias) {
                    bias = torch.empty(out_features, device: device, dtype: dtype).AsParameter();
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
                    ConditionallyRegisterParameter(BiasComponentName, _bias);
                }
            }

            public Parameter weight {
                get => _weight!;
                set {
                    if (value.Handle != _weight?.Handle) {
                        _weight?.Dispose();
                        _weight = (value.DetachFromDisposeScope() as Parameter)!;
                        ConditionallyRegisterParameter(WeightComponentName, _weight);
                    }
                }
            }

            // Rather than spending cycles discovering what parameters exist, we can just hardcode it.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) {
                if (_weight is not null && ReplaceParameter(dtype, device, _weight, out Parameter? w)) {
                    weight = w!;
                }
                if (_bias is not null && ReplaceParameter(dtype, device, _bias, out Parameter? b)) {
                    bias = b!;
                }
                return this;
            }

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking)
            {
                var device = new Device(deviceType, deviceIndex);
                if (_weight is not null && ReplaceParameter(_weight.dtype, device, _weight, out Parameter? w)) {
                    weight = w!;
                }
                if (_bias is not null && ReplaceParameter(_bias.dtype, device, _bias, out Parameter? b)) {
                    bias = b!;
                }
                return this;
            }

            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) {
                if (_weight is not null && ReplaceParameter(dtype, _weight.device, _weight, out Parameter? w)) {
                    weight = w!;
                }
                if (_bias is not null && ReplaceParameter(dtype, _bias.device, _bias, out Parameter? b)) {
                    bias = b!;
                }
                return this;
            }

            [ComponentName(Name = BiasComponentName)]
            private Parameter? _bias;
            [ComponentName(Name = WeightComponentName)]
            private Parameter? _weight;

            public long in1_features { get; set; }
            public long in2_features { get; set; }
            public long out_features { get; set; }
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