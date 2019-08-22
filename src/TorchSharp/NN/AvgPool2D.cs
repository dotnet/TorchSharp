using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class AvgPool2D : FunctionalModule<AdaptiveAvgPool2D>
    {
        private readonly long[] _kernelSize;
        private readonly long[] _stride;

        internal AvgPool2D(long[] kernelSize, long[] stride) : base()
        {
            _kernelSize = kernelSize;
            _stride = stride ?? new long[0];
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_avgPool2DApply(IntPtr tensor, int kernelSizeLength, long[] kernelSize, int strideLength, long[] stride);

        public override TorchTensor Forward(TorchTensor tensor)
        {
            return new TorchTensor(THSNN_avgPool2DApply(tensor.Handle, _kernelSize.Length, _kernelSize, _stride.Length, _stride));
        }
    }
}
