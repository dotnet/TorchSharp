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
            public long[] normalized_shape { get; set; }
            public double eps { get; set; }

            public bool elementwise_affine { get; set; }

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


                reset_parameters(elementwise_affine);
            }

            private void reset_parameters(bool elementwise_affine)
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

            protected override void Dispose(bool disposing)
            {
                _weight?.Dispose();
                _bias?.Dispose();
                base.Dispose(disposing);
            }
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
