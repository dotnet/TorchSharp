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
                return torch.nn.functional.cosine_similarity(input1, input2, this.dim, this.eps);
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
                return new CosineSimilarity(dim, eps);
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
