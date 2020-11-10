// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a dropout module for 2d/3d convolutational layers.
    /// </summary>
    public class FeatureAlphaDropout : Module
    {
        internal FeatureAlphaDropout (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_FeatureAlphaDropout_forward (Module.HType module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_FeatureAlphaDropout_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_FeatureAlphaDropout_ctor (double probability, out IntPtr pBoxedModule);

        static public FeatureAlphaDropout FeatureAlphaDropout (double probability = 0.5)
        {
            var handle = THSNN_FeatureAlphaDropout_ctor (probability, out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new FeatureAlphaDropout (handle, boxedHandle);
        }
    }

    public static partial class Functions
    {
        static public TorchTensor FeatureAlphaDropout (TorchTensor x, double probability = 0.5)
        {
            using (var f = Modules.FeatureAlphaDropout (probability)) {
                return f.Forward (x);
            }
        }
    }
}
