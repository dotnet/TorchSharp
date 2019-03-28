using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a ReLU module.
    /// </summary>
    public class ReLU : FunctionalModule<ReLU>
    {
        internal ReLU() : base()
        {
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr NN_ReluModule_Forward(IntPtr tensor);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(NN_ReluModule_Forward(tensor.Handle));
        }

        public override string GetName()
        {
            return typeof(ReLU).Name;
        }
    }
}
