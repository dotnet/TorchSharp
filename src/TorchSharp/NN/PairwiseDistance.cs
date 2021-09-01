// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
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
        public class PairwiseDistance : torch.nn.Module
        {
            internal PairwiseDistance(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_PairwiseDistance_forward(torch.nn.Module.HType module, IntPtr input1, IntPtr input2);

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
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSNN_PairwiseDistance_ctor(double p, double eps, bool keep_dim, out IntPtr pBoxedModule);

            static public PairwiseDistance PairwiseDistance(double p = 2.0, double eps = 1e-6, bool keep_dim = false)
            {
                var handle = THSNN_PairwiseDistance_ctor(p, eps, keep_dim, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new PairwiseDistance(handle, boxedHandle);
            }
            public static partial class functional
            {
                static public Tensor pairwise_distance(Tensor input1, Tensor input2, double p = 2.0, double eps = 1e-6, bool keep_dim = false)
                {
                    using (var f = nn.PairwiseDistance(p, eps, keep_dim)) {
                        return f.forward(input1, input2);
                    }
                }
            }
        }
    }
}
