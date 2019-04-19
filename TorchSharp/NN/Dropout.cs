using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a dropout module.
    /// </summary>
    public class Dropout : FunctionalModule<Dropout>
    {
        private double _probability;
        private bool _isTraining;

        internal Dropout(double probability, bool isTraining) : base()
        {
            _probability = probability;
            _isTraining = isTraining;
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_dropoutModuleApply(IntPtr tensor, double probability, bool isTraining);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(THSNN_dropoutModuleApply(tensor.Handle, _probability, _isTraining));
        }
    }
}
