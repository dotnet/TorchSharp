using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a ReLu module.
    /// </summary>
    public class MaxPool2D : FunctionalModule
    {
        private long _kernelSize;

        internal MaxPool2D(long kernelSize) : base()
        {
            _kernelSize = kernelSize;
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType NN_MaxPool2DModule_Forward(IntPtr tensor, long kernelSize);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(NN_MaxPool2DModule_Forward(tensor.Handle, _kernelSize));
        }
    }
}
