using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Linear : ProvidedModule
    {
        internal Linear(IntPtr handle) : base(handle)
        {
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_linearModuleApply(Module.HType module, IntPtr tensor);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(THSNN_linearModuleApply(handle, tensor.Handle));
        }
    }
}
