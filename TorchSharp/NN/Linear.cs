using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Linear : Module
    {
        internal Linear(IntPtr handle) : base(handle)
        {
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType NN_linearModule_Forward(Module.HType module, IntPtr tensor);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(NN_linearModule_Forward(handle, tensor.Handle));
        }
    }
}
