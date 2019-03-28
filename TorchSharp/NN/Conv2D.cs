using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class Conv2D : ProvidedModule
    {
        internal Conv2D(IntPtr handle) : base(handle)
        {
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr NN_conv2DModule_Forward(Module.HType module, IntPtr tensor);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(NN_conv2DModule_Forward(handle, tensor.Handle));
        }
    }
}
