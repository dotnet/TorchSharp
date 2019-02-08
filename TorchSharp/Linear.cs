using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp.NN
{
    public class Linear : Module
    {
        internal Linear(IntPtr handle) : base(handle)
        {
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType Forward_linear(Module.HType module, FloatTensor.HType tensor);

        public override FloatTensor Forward(FloatTensor tensor)
        {
            return new FloatTensor(Forward_linear(handle, tensor.handle));
        }
    }
}
