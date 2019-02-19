using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a functional modules (i.e., ReLu)
    /// </summary>
    public class Functional : Module
    {
        internal Functional(IntPtr handle) : base(handle)
        {
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType NN_functionalModule_Forward(HType module, IntPtr tensor);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(NN_functionalModule_Forward(handle, tensor.Handle));
        }
    }
}
