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
    using F = TorchSharp.torch.nn.functional;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a LayerNorm module.
        /// </summary>
        public sealed class LayerNorm : torch.nn.Module<Tensor, Tensor>
        {
            const string WeightComponentName = nameof(weight);
            const string BiasComponentName = nameof(bias);

            internal LayerNorm(long[] normalized_shape, double eps, bool elementwise_affine, bool bias, Device? device, ScalarType? dtype) : base(nameof(LayerNorm))
            {
                this.normalized_shape = normalized_shape;
                this.eps = eps;
                this.elementwise_affine = elementwise_affine;

                if (elementwise_affine)
                {
                    weight = Parameter(torch.empty(normalized_shape, dtype, device));
                    if (bias)
                    {
                        this.bias = Parameter(torch.empty(normalized_shape, dtype, device));
                    }
                }

                reset_parameters();
            }

            public void reset_parameters()
            {
                if (elementwise_affine)
                {
                    init.ones_(weight);
                }
                if (bias is not null)
                {
                    init.zeros_(bias);
                }
            }

            public override Tensor forward(Tensor tensor)
            {
                return F.layer_norm(tensor, normalized_shape, weight, bias, eps);
            }

            protected override void Dispose(bool disposing)
            {
                _weight?.Dispose();
                _bias?.Dispose();
                base.Dispose(disposing);
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


            public long[] normalized_shape { get; set; }
            public double eps { get; set; }
            public bool elementwise_affine { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies Layer Normalization over a mini-batch of inputs as described in the paper Layer Normalization
            /// </summary>
            /// <param name="normalized_shape">Input shape from an expected input.</param>
            /// <param name="eps">A value added to the denominator for numerical stability. Default: 1e-5</param>
            /// <param name="elementwise_affine">a boolean value that when set to true, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases).</param>
            /// <param name="bias">A boolean value that when set to true, this module has learnable per-element bias parameters initialized to zeros</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static LayerNorm LayerNorm(long[] normalized_shape, double eps = 1e-05, bool elementwise_affine = true, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                return new LayerNorm(normalized_shape, eps, elementwise_affine, bias, device, dtype);
            }

            /// <summary>
            /// Applies Layer Normalization over a mini-batch of inputs as described in the paper Layer Normalization
            /// </summary>
            /// <param name="normalized_shape">Input shape from an expected input.</param>
            /// <param name="eps">A value added to the denominator for numerical stability. Default: 1e-5</param>
            /// <param name="elementwise_affine">a boolean value that when set to true, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases).</param>
            /// <param name="bias">A boolean value that when set to true, this module has learnable per-element bias parameters initialized to zeros</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static LayerNorm LayerNorm(long normalized_shape, double eps = 1e-05, bool elementwise_affine = true, bool bias = true, Device? device = null, ScalarType? dtype = null)
            {
                return LayerNorm(new[] { normalized_shape }, eps, elementwise_affine, bias, device, dtype);
            }
        }
    }
}
