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

        internal Dropout(double probability, bool isTraining) : base()
        {
            _probability = probability;
            _isTraining = isTraining;
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr NN_DropoutModule_Forward(IntPtr tensor, double probability, bool isTraining);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(NN_DropoutModule_Forward(tensor.Handle, _probability, _isTraining));
        }
    }
}
