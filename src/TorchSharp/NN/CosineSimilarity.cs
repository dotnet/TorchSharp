// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a dropout module for 2d/3d convolutational layers.
        /// </summary>
        public class CosineSimilarity : torch.nn.Module
        {
            internal CosineSimilarity(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_CosineSimilarity_forward(torch.nn.Module.HType module, IntPtr input1, IntPtr input2);

            public Tensor forward(Tensor input1, Tensor input2)
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
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_CosineSimilarity_ctor(long dim, double eps, out IntPtr pBoxedModule);

            static public CosineSimilarity CosineSimilarity(long dim = 1, double eps = 1e-8)
            {
                var handle = THSNN_CosineSimilarity_ctor(dim, eps, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new CosineSimilarity(handle, boxedHandle);
            }

            public static partial class functional
            {
                static public Tensor cosine_similarity(Tensor input1, Tensor input2, long dim = 1, double eps = 1e-8)
                {
                    using (var f = nn.CosineSimilarity(dim, eps)) {
                        return f.forward(input1, input2);
                    }
                }
            }
        }
    }
}
