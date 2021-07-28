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
        public class FeatureAlphaDropout : torch.nn.Module
        {
            internal FeatureAlphaDropout(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            [DllImport("LibTorchSharp")]
            private static extern IntPtr THSNN_FeatureAlphaDropout_forward(torch.nn.Module.HType module, IntPtr tensor);

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_FeatureAlphaDropout_forward(handle, tensor.Handle);
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
            extern static IntPtr THSNN_FeatureAlphaDropout_ctor(double probability, out IntPtr pBoxedModule);

            static public FeatureAlphaDropout FeatureAlphaDropout(double probability = 0.5)
            {
                var handle = THSNN_FeatureAlphaDropout_ctor(probability, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new FeatureAlphaDropout(handle, boxedHandle);
            }

            public static partial class functional
            {
                static public Tensor feature_alpha_dropout(Tensor x, double probability = 0.5)
                {
                    using (var f = nn.FeatureAlphaDropout(probability)) {
                        return f.forward(x);
                    }
                }
            }
        }
    }
}
