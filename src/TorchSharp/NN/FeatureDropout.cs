// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a dropout module for 2d/3d convolutational layers.
    /// </summary>
    public class FeatureDropout : Module
    {
        internal FeatureDropout (IntPtr handle) : base (handle)
        {
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_featureDropoutApply (IntPtr tensor);

        public override TorchTensor Forward (TorchTensor tensor)
        {
            return new TorchTensor (THSNN_featureDropoutApply (tensor.Handle));
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_featureDropoutModule ();

        static public FeatureDropout FeatureDropout ()
        {
            var handle = THSNN_featureDropoutModule ();
            Torch.CheckForErrors ();
            return new FeatureDropout (handle);
        }
    }

    public static partial class Functions
    {
        static public TorchTensor FeatureDropout (TorchTensor x)
        {
            using (var f = Modules.FeatureDropout ()) {
                return f.Forward (x);
            }
        }
    }
}
