// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using TorchSharp.Amp;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.NativeMethods;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Softplus module.
        /// </summary>
        public sealed class Softplus : ParameterLessModule<Tensor, Tensor>
        {
            internal Softplus(double beta = 1, double threshold = 20) : base(nameof(Softplus))
            {
                this.beta = beta;
                this.threshold = threshold;
            }

            public override Tensor forward(Tensor tensor)
            {
                return torch.nn.functional.softplus(tensor, beta, threshold);
            }

            public double beta {get; set;}
            public double threshold {get; set;}
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Softplus
            /// </summary>
            /// <param name="beta">The β value for the Softplus formulation.</param>
            /// <param name="threshold">Values above this revert to a linear function</param>
            /// <returns></returns>
            public static Softplus Softplus(double beta = 1, double threshold = 20)
            {
                var handle = THSNN_Softplus_ctor(beta, threshold, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                handle = AutocastMode.AutoCast(handle, ScalarType.Float32); //Should put this here
                return new Softplus(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Softplus
                /// </summary>
                /// <param name="x">The input tensor</param>
                /// <param name="beta">The β value for the Softplus formulation.</param>
                /// <param name="threshold">Values above this revert to a linear function</param>
                /// <returns></returns>
                public static Tensor softplus(Tensor x, double beta = 1, double threshold = 20)
                {
                    return x.softplus(beta, threshold);
                }
            }
        }
    }
}
