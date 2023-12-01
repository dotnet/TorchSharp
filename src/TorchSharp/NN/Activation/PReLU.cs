// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

#nullable enable
namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a PReLU module.
        /// </summary>
        public sealed class PReLU : torch.nn.Module<Tensor, Tensor>
        {
            internal PReLU(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_PReLU_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            public override string GetName()
            {
                return typeof(PReLU).Name;
            }

            public Parameter? weight {
                get {
                    var res = THSNN_PReLU_weight(handle);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return (res == IntPtr.Zero) ? null : new Parameter(res);
                }
                set {
                    if (value is null) throw new ArgumentNullException("weight cannot be set to 'null'");
                    THSNN_PReLU_set_weight(handle, value!.Handle);
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
                var handle = THSNN_PReLU_ctor(num_parameters, init, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new PReLU(handle, boxedHandle).MoveModule<PReLU>(device, dtype);
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
