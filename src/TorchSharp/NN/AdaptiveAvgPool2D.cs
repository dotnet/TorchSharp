// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a AdaptiveAvgPool2D module.
    /// </summary>
    public class AdaptiveAvgPool2D : Module
    {
        internal AdaptiveAvgPool2D (IntPtr handle, params long[] outputSize) : base (handle)
        {
            _outputSize = outputSize;
        }

        private long[] _outputSize;

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_adaptiveAvgPool2DApply (IntPtr tensor, int length, long[] outputSize);

        public override TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_adaptiveAvgPool2DApply (tensor.Handle, _outputSize.Length, _outputSize);
            Torch.CheckForErrors ();
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_adaptiveAvgPool2dModule ();

        static public AdaptiveAvgPool2D AdaptiveAvgPool2D (params long[] outputSize)
        {
            var handle = THSNN_adaptiveAvgPool2dModule ();
            Torch.CheckForErrors ();
            return new AdaptiveAvgPool2D (handle, outputSize);
        }
    }

    public static partial class Functions
    {
        static public TorchTensor AdaptiveAvgPool2D (TorchTensor x, params long[] outputSize)
        {
            using (var d = Modules.AdaptiveAvgPool2D (outputSize)) {
                return d.Forward (x);
            }
        }
    }
}
