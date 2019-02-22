using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a dropout module for 2d/3d convolutational layers.
    /// </summary>
    public class FeatureDropout : FunctionalModule
    {
        internal FeatureDropout() : base()
        {
        }

        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType NN_FeatureDropout_Forward(IntPtr tensor);

        public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
        {
            return new FloatTensor(NN_FeatureDropout_Forward(tensor.Handle));
        }
    }
}
