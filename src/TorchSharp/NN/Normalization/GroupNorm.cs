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
        /// This class is used to represent a GroupNorm module.
        /// </summary>
        public sealed class GroupNorm : torch.nn.Module<Tensor, Tensor>
        {
            internal GroupNorm(long num_groups, long num_channels, double eps, bool affine, Device? device, ScalarType? dtype) : base(nameof(GroupNorm))
            {
                this.eps = eps;
                this.affine = affine;
                this.num_groups = num_groups;

                if (affine) {
                    weight = Parameter(torch.empty(num_channels, dtype, device));
                    this.bias = Parameter(torch.empty(num_channels, dtype, device));
                }
            }

            public override Tensor forward(Tensor tensor)
            {
                if (tensor.Dimensions < 3)
                    throw new ArgumentException($"Invalid number of dimensions for GroupNorm argument: {tensor.Dimensions}");
                return F.group_norm(tensor, num_groups, weight, bias, eps);
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
                    ConditionallyRegisterParameter(nameof(bias), _bias);
                }
            }

            public Parameter weight {
                get => _weight!;
                set {
                    if (value.Handle != _weight?.Handle) {
                        _weight?.Dispose();
                        _weight = (value.DetachFromDisposeScope() as Parameter)!;
                        ConditionallyRegisterParameter(nameof(weight), _weight);
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

            [ComponentName(Name = nameof(bias))]
            private Parameter? _bias;
            [ComponentName(Name = nameof(weight))]
            private Parameter? _weight;
            public long num_groups { get; set; }
            public double eps { get; set; }
            public bool affine { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Applies Group Normalization over a mini-batch of inputs as described in the paper Group Normalization
            /// </summary>
            /// <param name="num_groups">Number of groups to separate the channels into</param>
            /// <param name="num_channels">Number of channels expected in input</param>
            /// <param name="eps">A value added to the denominator for numerical stability.</param>
            /// <param name="affine">A boolean value that when set to true, this module has learnable per-channel affine parameters initialized to ones (for weights) and zeros (for biases).</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            /// <returns></returns>
            public static GroupNorm GroupNorm(long num_groups, long num_channels, double eps = 1e-05, bool affine = true, Device? device = null, ScalarType? dtype = null)
            {
                return new GroupNorm(num_groups, num_channels, eps, affine, device, dtype);
            }
        }
    }
}
