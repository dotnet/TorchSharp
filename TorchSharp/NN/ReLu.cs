using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a ReLu module.
    /// </summary>
    public class ReLu : FunctionalModule
    {
        internal ReLu() : base()
        {
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType NN_ReluModule_Forward(IntPtr tensor);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(NN_ReluModule_Forward(tensor.Handle));
        }
    }
}
