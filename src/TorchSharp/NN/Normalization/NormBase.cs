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
        public abstract class NormBase : torch.nn.Module<Tensor, Tensor>
        {
            public NormBase(long num_features,
                            double eps,
                            double? momentum,
                            bool affine,
                            bool track_running_stats,
                            Device? device,
                            ScalarType? dtype,
                            string name) : base(name)
            {
                this.num_features = num_features;
                this.eps = eps;
                this.momentum = momentum;
                this.affine = affine;
                this.track_running_stats = track_running_stats;

                if (affine) {
                    this.weight = Parameter(torch.empty(num_features, dtype, device));
                    this.bias = Parameter(torch.empty(num_features, dtype, device));
                }

                if (track_running_stats) {
                    this.running_mean = torch.zeros(num_features, dtype, device);
                    this.running_var = torch.ones(num_features, dtype, device);
                    this.num_batches_tracked = torch.tensor(0L, dtype, device);
                }
                reset_parameters();
            }

            private void ResetRunningStats()
            {
                if (track_running_stats){
                    init.zeros_(this._running_mean);
                    init.ones_(this._running_var);
                    init.zeros_(this._num_batches_tracked);
                }
            }

            // For backward compat.
            public void reset_running_stats() => ResetRunningStats();

            public void reset_parameters() {
                ResetRunningStats();
                if (affine) {
                    init.ones_(this._weight);
                    init.zeros_(this._bias);
                }
            }

            protected abstract void ValidateInputDimensions(Tensor input);

            protected override void Dispose(bool disposing)
            {
                _weight?.Dispose();
                _bias?.Dispose();
                _running_mean?.Dispose();
                _running_var?.Dispose();
                _num_batches_tracked?.Dispose();
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

            public Tensor? running_mean {
                get => _running_mean;
                set {
                    _running_mean?.Dispose();
                    _running_mean = value?.DetachFromDisposeScope();
                    ConditionallyRegisterBuffer(nameof(running_mean), _running_mean);
                }
            }

            public Tensor? running_var {
                get => _running_var;
                set {
                    _running_var?.Dispose();
                    _running_var = value?.DetachFromDisposeScope();
                    ConditionallyRegisterBuffer(nameof(running_var), _running_var);
                }
            }

            public Tensor? num_batches_tracked {
                get => _num_batches_tracked;
                set {
                    _num_batches_tracked?.Dispose();
                    _num_batches_tracked = value?.DetachFromDisposeScope();
                    ConditionallyRegisterBuffer(nameof(num_batches_tracked), _num_batches_tracked);
                }
            }

            // Rather than spending cycles discovering what parameters exist, we can just hardcode it.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) {
                if (_weight is not null && ReplaceParameter(dtype, device, _weight, out var w)) {
                    weight = w!;
                }
                if (_bias is not null && ReplaceParameter(dtype, device, _bias, out var b)) {
                    bias = b!;
                }
                if (_running_mean is not null && ReplaceBuffer(dtype, device, _running_mean, out Tensor? rm)) {
                    running_mean = rm!;
;               }
                if (_running_var is not null && ReplaceBuffer(dtype, device, _running_var, out Tensor? rv)) {
                    running_var = rv!;
;               }
                if (_num_batches_tracked is not null && ReplaceBuffer(dtype, device, _num_batches_tracked, out Tensor? nbt)) {
                    num_batches_tracked = nbt!;
;               }
                return this;
            }

            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking)
            {
                var device = new Device(deviceType, deviceIndex);
                if (_weight is not null && ReplaceParameter(_weight.dtype, device, _weight, out var w)) {
                    weight = w!;
                }
                if (_bias is not null && ReplaceParameter(_bias.dtype, device, _bias, out var b)) {
                    bias = b!;
                }
                if (_running_mean is not null && ReplaceBuffer(_running_mean.dtype, device, _running_mean, out Tensor? rm)) {
                    running_mean = rm!;
;               }
                if (_running_var is not null && ReplaceBuffer(_running_var.dtype, device, _running_var, out Tensor? rv)) {
                    running_var = rv!;
;               }
                if (_num_batches_tracked is not null && ReplaceBuffer(_num_batches_tracked.dtype, device, _num_batches_tracked, out Tensor? nbt)) {
                    num_batches_tracked = nbt!;
;               }
                return this;
            }

            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) {
                if (_weight is not null && ReplaceParameter(dtype, _weight.device, _weight, out var w)) {
                    weight = w!;
                }
                if (_bias is not null && ReplaceParameter(dtype, _bias.device, _bias, out var b)) {
                    bias = b!;
                }
                if (_running_mean is not null && ReplaceBuffer(dtype, _running_mean.device, _running_mean, out Tensor? rm)) {
                    running_mean = rm!;
;               }
                if (_running_var is not null && ReplaceBuffer(dtype, _running_var.device, _running_var, out Tensor? rv)) {
                    running_var = rv!;
;               }
                if (_num_batches_tracked is not null && ReplaceBuffer(dtype, _num_batches_tracked.device, _num_batches_tracked, out Tensor? nbt)) {
                    num_batches_tracked = nbt!;
;               }
                return this;
            }

            public long num_features { get; private set; }

            public double eps { get; set; }

            public double? momentum { get; set; }

            public bool affine { get; private set; }

            public bool track_running_stats { get; private set; }

            [ComponentName(Name = nameof(bias))]
            private Parameter? _bias;

            [ComponentName(Name = nameof(weight))]
            private Parameter? _weight;

            [ComponentName(Name = nameof(running_mean))]
            private Tensor? _running_mean;

            [ComponentName(Name = nameof(running_var))]
            private Tensor? _running_var;

            [ComponentName(Name = nameof(num_batches_tracked))]
            private Tensor? _num_batches_tracked;
        }
    }
}