// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// Computes the pairwise distance between vectors using the p-norm.
        /// </summary>
        public sealed class PairwiseDistance : torch.nn.Module<Tensor, Tensor, Tensor>
        {
            internal PairwiseDistance(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor input1, Tensor input2)
            {
                var res = THSNN_PairwiseDistance_forward(handle, input1.Handle, input2.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            public static PairwiseDistance PairwiseDistance(double p = 2.0, double eps = 1e-6, bool keep_dim = false)
            {
                var handle = THSNN_PairwiseDistance_ctor(p, eps, keep_dim, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new PairwiseDistance(handle, boxedHandle);
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
                /// <param name="keep_dim">Determines whether or not to keep the vector dimension.</param>
                /// <returns></returns>
                public static Tensor pairwise_distance(Tensor input1, Tensor input2, double p = 2.0, double eps = 1e-6, bool keep_dim = false)
                {
                    using (var f = nn.PairwiseDistance(p, eps, keep_dim)) {
                        return f.call(input1, input2);
                    }
                }
            }
        }
    }
}
