// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using System.Collections.Generic;
    using System.Linq;
    using Modules;
    using TorchSharp.Utils;

    public enum PaddingModes
    {
        Zeros = 0,
        Reflect = 1,
        Replicate = 2,
        Circular = 3,
        Constant = 4,
    }

    public enum Padding
    {
        Valid = 0,
        Same = 1
    }

    namespace Modules
    {
        public abstract class Convolution : torch.nn.Module<Tensor, Tensor>
        {
            const string WeightComponentName = nameof(weight);
            const string BiasComponentName = nameof(bias);

            protected Convolution(string name, long in_channels, long out_channels, long[] kernel_size, long[] stride, long[]? padding, Padding? padding_type, long[] dilation, bool transposed, long[] output_padding, long groups, bool bias, PaddingModes padding_mode, torch.Device? device = null, ScalarType? dtype = null) : base(name)
            {
                if (transposed && padding_mode != PaddingModes.Zeros)
                    throw new ArgumentException("Only PaddingModes.Zeros is supported for ConvTransposeND", nameof(padding_mode));

                this.in_channels = in_channels;
                this.out_channels = out_channels;
                this.kernel_size = kernel_size;
                this.stride = stride;
                this.dilation = dilation;
                this.transposed = transposed;
                this.output_padding = output_padding;
                this.groups = groups;
                this.padding_mode = padding_mode;

                // Set this so the constructor doesn't give a non-null error, and the actual value is set in the
                // SetPadding function called right after.
                this._reversed_padding_repeated_twice = Array.Empty<long>();
                if (padding_type.HasValue)
                    SetPadding(padding_type.Value);
                else
                    SetPadding(padding ?? Enumerable.Repeat(0L, kernel_size.Length).ToArray());

                long[] weightDims = transposed ? new[] { in_channels, out_channels / groups } : new[] { out_channels, in_channels / groups };
                weightDims = weightDims.Concat(kernel_size).ToArray();

                weight = torch.nn.Parameter(torch.empty(weightDims, device: device, dtype: dtype));
                if (bias)
                    this.bias = torch.nn.Parameter(torch.empty(out_channels, device: device, dtype: dtype));

                this.reset_parameters();
            }

            public void reset_parameters()
            {
                torch.nn.init.kaiming_uniform_(this.weight, Math.Sqrt(5));
                if (this.bias is not null) {
                    var (fanIn, _) = torch.nn.init.CalculateFanInAndFanOut(this.weight);
                    if (fanIn != 0) {
                        var bound = 1 / Math.Sqrt(fanIn);
                        torch.nn.init.uniform_(this.bias, -bound, bound);
                    }
                }
            }

            public void SetPadding(long[] padding)
            {
                this.padding_type = null;
                this.padding = padding;
                UpdateReversedPaddingArray();
            }

            public void SetPadding(Padding padding)
            {
                if (this.transposed)
                    throw new ArgumentException("Cannot set padding type on ConvTransposeND");
                this.padding = null;
                this.padding_type = padding;
                UpdateReversedPaddingArray();
            }

            protected bool ValidateShape(Tensor input, long dimensions)
            {
                var shape = input.shape;
                var ndim = shape.LongLength;

                return (ndim == dimensions + 2) && (input.shape[1] == in_channels) ||  // Batched: N + C + dims
                       (ndim == dimensions + 1 && input.shape[0] == in_channels);      // Unbathced: C + dims

            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    _weight?.Dispose();
                    _bias?.Dispose();
                }
            }

            private void UpdateReversedPaddingArray()
            {
                if (this.padding != null) {
                    // Reverse the order of `padding` and repeat each element for `2` times.
                    _reversed_padding_repeated_twice = padding.Reverse().SelectMany(value => Enumerable.Repeat(value, 2)).ToArray();
                } else {
                    _reversed_padding_repeated_twice = Enumerable.Repeat(0L, kernel_size.Length * 2).ToArray();
                    if (padding_type == Padding.Same) {
                        if (stride.Any(i => i != 1))
                            throw new ArgumentException("padding='same' is not supported for strided convolutions");

                        for (int i = kernel_size.Length - 1; i >= 0; i--) {
                            long totalPadding = dilation[i] * (kernel_size[i] - 1);
                            long leftPad = totalPadding / 2;
                            _reversed_padding_repeated_twice[2 * i] = leftPad;
                            _reversed_padding_repeated_twice[2 * i + 1] = totalPadding - leftPad;
                        }
                    }
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

            // Included to avoid API compat issues.
            [Obsolete("Deprecated API", true)]
            protected Convolution(IntPtr handle, IntPtr boxedHandle, long input_channels) : base(handle, boxedHandle) {
                throw new NotImplementedException("Deprecated API.");
            }

            [ComponentName(Name = BiasComponentName)]
            protected Parameter? _bias;
            [ComponentName(Name = WeightComponentName)]
            protected Parameter? _weight;

            // `_reversed_padding_repeated_twice` is the padding to be passed to
            // `F.pad` if needed (e.g., for non-zero padding types that are
            // implemented as two ops: padding + conv). `F.pad` accepts paddings in
            // reverse order than the dimension.
            protected long[] _reversed_padding_repeated_twice;

            [Obsolete("Deprecated.", true)]
            protected long input_channels;

            public long in_channels { get; }
            public long out_channels { get; }
            public bool transposed { get; }

            public Padding? padding_type { get; private set; }
            public long[]? padding { get; private set; }
            public PaddingModes padding_mode { get; set; }
            public long groups { get; set; }
            public long[] kernel_size { get; set; }
            public long[] stride { get; set; }
            public long[] dilation { get; set; }
            public long[] output_padding { get; set; }
        }
    }
}