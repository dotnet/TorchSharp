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
        extern static IntPtr THSNN_conv2DModuleApply(Module.HType module, IntPtr tensor);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(THSNN_conv2DModuleApply(handle, tensor.Handle));
        }
    }
}
