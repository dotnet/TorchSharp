// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class AvgPool2D : Module
    {
        private readonly long[] _kernelSize;
        private readonly long[] _stride;

        internal AvgPool2D (IntPtr handle, long[] kernelSize, long[] stride) : base (handle)
        {
            _kernelSize = kernelSize;
            _stride = stride ?? new long[0];
        }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_avgPool2DApply (IntPtr tensor, int kernelSizeLength, long[] kernelSize, int strideLength, long[] stride);

        public override TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_avgPool2DApply (tensor.Handle, _kernelSize.Length, _kernelSize, _stride.Length, _stride);
            Torch.CheckForErrors ();
            return new TorchTensor (res);
        }
    }
    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_avgPool2dModule ();

        static public AvgPool2D AvgPool2D (long[] kernelSize, long[] stride = null)
        {
            var handle = THSNN_avgPool2dModule ();
            Torch.CheckForErrors ();
            return new AvgPool2D (handle, kernelSize, stride);
        }
    }
    public static partial class Functions
    {
        static public TorchTensor AvgPool2D (TorchTensor x, long[] kernelSize, long[] stride = null)
        {
            using (var d = Modules.AvgPool2D (kernelSize, stride)) {
                return d.Forward (x);
            }
        }
    }

}
