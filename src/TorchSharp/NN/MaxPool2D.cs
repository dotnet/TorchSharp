// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a MaxPool2D module.
    /// </summary>
    public class MaxPool2D : Module
    {
        internal MaxPool2D (IntPtr handle, long[] kernelSize, long[] stride) : base (handle)
        {
            _kernelSize = kernelSize;
            _stride = stride ?? new long[0];
        }

        private readonly long[] _kernelSize;
        private readonly long[] _stride;

        public override TorchTensor Forward (TorchTensor tensor)
        {
            return tensor.MaxPool2D (_kernelSize, _stride);
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_maxPool2dModule ();
        static public MaxPool2D MaxPool2D (long[] kernelSize, long[] stride = null)
        {
            var handle = THSNN_maxPool2dModule ();
            Torch.CheckForErrors ();
            return new MaxPool2D (handle, kernelSize, stride);
        }
    }
    public static partial class Functions
    {

        static public TorchTensor MaxPool2D (TorchTensor x, long[] kernelSize, long[] stride = null)
        {
            using (var m = Modules.MaxPool2D (kernelSize, stride)) {
                return m.Forward (x);
            }
        }

    }
}
