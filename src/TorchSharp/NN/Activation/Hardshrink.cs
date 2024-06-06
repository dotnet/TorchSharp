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
        /// This class is used to represent a Hardshrink module.
        /// </summary>
        public sealed class Hardshrink : ParamLessModule<Tensor, Tensor>
        {
            internal Hardshrink(double lambda = 0.5) : base(nameof(Hardshrink)) 
            { 
                this.lambda = lambda;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.hardshrink(tensor, lambda);
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
            /// Hardshrink
            /// </summary>
            /// <param name="lambda"> the λ value for the Hardshrink formulation. Default: 0.5</param>
            /// <returns></returns>
            public static Hardshrink Hardshrink(double lambda = 0.5)
            {
                return new Hardshrink(lambda);
            }

            public static partial class functional
            {
                /// <summary>
                /// Hardshrink
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="lambda">The λ value for the Hardshrink formulation. Default: 0.5</param>
                /// <returns></returns>
                public static Tensor hardshrink(Tensor x, double lambda = 0.5)
                {
                    using var sc = (Scalar)lambda;
                    var result = THSTensor_hardshrink(x.Handle, sc.Handle);
                    if (result == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(result);
                }
            }
        }
    }
}
