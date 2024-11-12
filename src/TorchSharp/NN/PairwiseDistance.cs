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
        /// Computes the pairwise distance between vectors using the p-norm.
        /// </summary>
        public sealed class PairwiseDistance : ParameterLessModule<Tensor, Tensor, Tensor>
        {
            public double norm { get; set; }
            public double eps { get; set; }
            public bool keepdim { get; set; }

            internal PairwiseDistance(
                double p = 2.0, double eps = 1e-6, bool keepdim = false)
                : base(nameof(PairwiseDistance))
            {
                this.norm = p;
                this.eps = eps;
                this.keepdim = keepdim;
            }

            public override Tensor forward(Tensor input1, Tensor input2)
            {
                return nn.functional.pairwise_distance(input1, input2, norm, eps, keepdim);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            public static PairwiseDistance PairwiseDistance(double p = 2.0, double eps = 1e-6, bool keepdim = false)
            {
                return new PairwiseDistance(p, eps, keepdim);
            }

            public static partial class functional
            {
                /// <summary>
                /// Computes the pairwise distance between vectors using the p-norm.
                /// </summary>
                /// <param name="input1">(N,D) or (D) where N = batch dimension and D = vector dimension</param>
                /// <param name="input2">(N, D) or (D), same shape as the Input1</param>
                /// <param name="p">The norm degree. Default: 2</param>
                /// <param name="eps">Small value to avoid division by zero.</param>
                /// <param name="keepdim">Determines whether or not to keep the vector dimension.</param>
                /// <returns></returns>
                public static Tensor pairwise_distance(Tensor input1, Tensor input2, double p = 2.0, double eps = 1e-6, bool keepdim = false)
                {
                    var res = THSNN_pairwise_distance(input1.Handle, input2.Handle, p, eps, keepdim);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
