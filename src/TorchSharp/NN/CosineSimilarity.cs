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
        /// A cosine similarity module.
        /// </summary>
        public sealed class CosineSimilarity : ParameterLessModule<Tensor, Tensor, Tensor>
        {
            internal CosineSimilarity(long dim = 1, double eps = 1e-8) : base(nameof(CosineSimilarity))
            {
                this.dim = dim;
                this.eps = eps;
            }

            public override Tensor forward(Tensor input1, Tensor input2)
            {
                var res = THSNN_CosineSimilarity_forward(handle, input1.Handle, input2.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                res= AutocastMode.AutoCast(res, ScalarType.Float32);
                return new Tensor(res);
            }

            public long dim { get; set; }
            public double eps { get; set; }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Returns cosine similarity between x1 and x2, computed along dim. Inputs must have same shape.
            /// </summary>
            /// <param name="dim">Dimension where cosine similarity is computed. Default: 1</param>
            /// <param name="eps">Small value to avoid division by zero. Default: 1e-8</param>
            /// <returns></returns>
            public static CosineSimilarity CosineSimilarity(long dim = 1, double eps = 1e-8)
            {
                var handle = THSNN_CosineSimilarity_ctor(dim, eps, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                handle = AutocastMode.AutoCast(handle, ScalarType.Float32);
                return new CosineSimilarity(handle, boxedHandle);
            }

            public static partial class functional
            {
                /// <summary>
                /// Returns cosine similarity between x1 and x2, computed along dim. Inputs must have same shape.
                /// </summary>
                /// <param name="x1">First input.</param>
                /// <param name="x2">Second input (of size matching x1).</param>
                /// <param name="dim">Dimension where cosine similarity is computed. Default: 1</param>
                /// <param name="eps">Small value to avoid division by zero. Default: 1e-8</param>
                /// <returns></returns>
                public static Tensor cosine_similarity(Tensor x1, Tensor x2, long dim = 1, double eps = 1e-8)
                {
                    var res = THSNN_cosine_similarity(x1.Handle, x2.Handle, dim, eps);
                    if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                    return new Tensor(res);
                }
            }
        }
    }
}
