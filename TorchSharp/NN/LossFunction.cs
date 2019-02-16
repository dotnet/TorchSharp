using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// Class maintaing the supported loss functions.
    /// </summary>
    public class LossFunction
    {
        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType NN_LossMSE(IntPtr srct, IntPtr trgt, long reduction);

        public static ITorchTensor<float> MSE<T>(ITorchTensor<T> src, ITorchTensor<T> target, Reduction reduction = Reduction.None)
        {
            return new FloatTensor(NN_LossMSE(src.Handle, target.Handle, (long)reduction));
        }
    }

    public enum Reduction : long
    {
        None = 1,
        Mean = 2,
        Sum = 3
    }
}
