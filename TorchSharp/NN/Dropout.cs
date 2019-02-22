using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a dropout module.
    /// </summary>
    public class Dropout : FunctionalModule
    {
        private double _probability;
        private Func<bool> _isTraining;

        internal Dropout(double probability, Func<bool> isTraining) : base()
        {
            _probability = probability;
            _isTraining = isTraining;
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType NN_LogSoftMaxModule_Forward(IntPtr tensor, double probability, bool isTraining);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(NN_LogSoftMaxModule_Forward(tensor.Handle, _probability, _isTraining.Invoke()));
        }
    }
}
