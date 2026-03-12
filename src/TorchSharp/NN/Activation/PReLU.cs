// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;
    using TorchSharp.Utils;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a PReLU module.
        /// </summary>
        public sealed class PReLU : torch.nn.Module<Tensor, Tensor>
        {
            internal PReLU(long num_parameters, double init, Device? device = null, ScalarType? dtype = null) : base(nameof(PReLU)) 
            { 
                this.init = init;
                this.num_parameters = num_parameters;
 
                var w = torch.empty(num_parameters, device:device, dtype:dtype);
                w.fill_(init);

                this.weight = new Parameter(w);
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.prelu(tensor, weight);
            }

            public override string GetName()
            {
                return typeof(PReLU).Name;
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

            public long num_parameters {
                get; private set;
            }

            public double init {
                get; private set;
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    _weight?.Dispose();
                }
            }

            [ComponentName(Name = nameof(weight))]
            private Parameter? _weight;
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Parameterized Rectified Linear Unit
            /// </summary>
            /// <param name="num_parameters">
            /// Number of 'a' to learn.
            /// Although it takes an int as input, there is only two values are legitimate: 1, or the number of channels at input.
            /// </param>
            /// <param name="init">The initial value of 'a'.</param>
            /// <param name="device">The desired device of the parameters and buffers in this module</param>
            /// <param name="dtype">The desired floating point or complex dtype of the parameters and buffers in this module</param>
            public static PReLU PReLU(long num_parameters, double init = 0.25, Device? device = null, ScalarType? dtype = null)
            {
                return new PReLU(num_parameters, init).MoveModule<PReLU>(device, dtype);
            }

            public static partial class functional
            {
                /// <summary>
                /// Parameterized Rectified Linear Unit
                /// </summary>
                /// <param name="input">The input tensor</param>
                /// <param name="weight">Weight is expected to be a scalar or 1-D tensor.</param>
                public static Tensor prelu(Tensor input, Tensor weight)
                {
                    return input.prelu(weight);
                }
            }
        }
    }
}
