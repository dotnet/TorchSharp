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
        private readonly bool _inPlace;

        internal ReLU(bool inPlace = false) : base()
        {
            _inPlace = inPlace;
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_reluApply(IntPtr tensor, bool inPlace);

        public override TorchTensor Forward(TorchTensor tensor)
        {
            return new TorchTensor(THSNN_reluApply(tensor.Handle, _inPlace));
        }

        public override string GetName()
        {
            return typeof(ReLU).Name;
        }
    }
}
