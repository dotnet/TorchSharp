// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a ReLu module.
    /// </summary>
    public class AdaptiveAvgPool2D : FunctionalModule<AdaptiveAvgPool2D>
    {
        private long[] _outputSize;

        internal AdaptiveAvgPool2D(params long []outputSize) : base()
        {
            _outputSize = outputSize;
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_adaptiveAvgPool2DApply(IntPtr tensor, int length, long[] outputSize);

        public override TorchTensor Forward(TorchTensor tensor)
        {
            return new TorchTensor(THSNN_adaptiveAvgPool2DApply(tensor.Handle, _outputSize.Length, _outputSize));
        }
    }
}
