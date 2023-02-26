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
        /// This class is used to represent a dropout module for 2d/3d convolutational layers.
        /// </summary>
        public sealed class CosineSimilarity : torch.nn.Module<Tensor, Tensor, Tensor>
        {
            internal CosineSimilarity(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor input1, Tensor input2)
            {
                var res = THSNN_CosineSimilarity_forward(handle, input1.Handle, input2.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
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
                    using (var f = nn.CosineSimilarity(dim, eps)) {
                        return f.call(x1, x2);
                    }
                }
            }
        }
    }
}
