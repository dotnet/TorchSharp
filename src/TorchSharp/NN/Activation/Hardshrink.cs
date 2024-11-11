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
        public sealed class Hardshrink : ParameterLessModule<Tensor, Tensor>
        {
            internal Hardshrink(double lambda = 0.5) : base(nameof(Hardshrink))
            {
                this.lambda = lambda;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.hardshrink(tensor, lambda);
            }

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

                /// <summary>
                /// Hardshrink
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="lambda">The λ value for the Hardshrink formulation. Default: 0.5</param>
                /// <remarks>Only here for backward comaptibility.</remarks>
                [Obsolete("Not using the PyTorch naming convention.",false)]
                public static Tensor Hardshrink(Tensor x, double lambda = 0.5) => hardshrink(x, lambda);
            }
        }
    }
}
