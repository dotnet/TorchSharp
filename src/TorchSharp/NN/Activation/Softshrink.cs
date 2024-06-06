// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Softshrink module.
        /// </summary>
        public sealed class Softshrink : ParamLessModule<Tensor, Tensor>
        {
            internal Softshrink(double lambda = 0.5) : base(nameof(Softshrink)) 
            { 
                this.lambda = lambda;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.softshrink(tensor, lambda);
            }

            // Rather than spending cycles only to discover that this module has neither
            // parameters nor buffers, just shortcut the move completely.
            protected internal override nn.Module _to(Device device, ScalarType dtype, bool non_blocking) => this;
            protected internal override nn.Module _to(DeviceType deviceType, int deviceIndex, bool non_blocking) => this;
            protected internal override nn.Module _to(ScalarType dtype, bool non_blocking) => this;

            public double lambda {get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Softshrink
            /// </summary>
            /// <param name="lambda"> the λ value for the Softshrink formulation. Default: 0.5</param>
            /// <returns></returns>
            public static Softshrink Softshrink(double lambda = 0.5)
            {
                return new Softshrink(lambda);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softshrink
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="lambda">The λ value for the Softshrink formulation. Default: 0.5</param>
                /// <returns></returns>
                public static Tensor softshrink(Tensor x, double lambda = 0.5)
                {
                    using var sc = (Scalar)lambda;
                    var result = THSTensor_softshrink(x.Handle, sc.Handle);
                    if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(result);
                }
            }
        }
    }
}
