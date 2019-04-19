using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a ReLu module.
    /// </summary>
    public class MaxPool2D : FunctionalModule<MaxPool2D>
    {
        private long _kernelSize;

        internal MaxPool2D(long kernelSize) : base()
        {
            _kernelSize = kernelSize;
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_maxPool2DApply(IntPtr tensor, long kernelSize);

        public override ITorchTensor Forward(ITorchTensor tensor)
        {
            return new TorchTensor(THSNN_maxPool2DApply(tensor.Handle, _kernelSize));
        }
    }
}
