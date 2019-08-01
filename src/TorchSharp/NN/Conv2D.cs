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

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSNN_conv2d_load_module(string location);

        public new static Conv2D Load(String location)
        {
            return new Conv2D(THSNN_conv2d_load_module(location));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_conv2DModuleApply(Module.HType module, IntPtr tensor);

        public override TorchTensor Forward(TorchTensor tensor)
        {
            return new TorchTensor(THSNN_conv2DModuleApply(handle, tensor.Handle));
        }
    }
}
