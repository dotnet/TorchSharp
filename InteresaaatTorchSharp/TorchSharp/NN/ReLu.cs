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
        extern static IntPtr THSNN_reluApply(IntPtr tensor);

        public override TorchTensor Forward(TorchTensor tensor)
        {
            return new TorchTensor(THSNN_reluApply(tensor.Handle));
        }

        public override string GetName()
        {
            return typeof(ReLU).Name;
        }
    }
}
