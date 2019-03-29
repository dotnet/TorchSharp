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
        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_lossMSE(IntPtr srct, IntPtr trgt, long reduction);

        public static ITorchTensor<float> MSE<T>(ITorchTensor<T> src, ITorchTensor<T> target, Reduction reduction = Reduction.Mean)
        {
            return new FloatTensor(THSNN_lossMSE(src.Handle, target.Handle, (long)reduction));
        }

        [DllImport("libTorchSharp")]
        extern static IntPtr THSNN_lossNLL(IntPtr srct, IntPtr trgt, long reduction);

        public static ITorchTensor<float> NLL<T, U>(ITorchTensor<T> src, ITorchTensor<U> target, Reduction reduction = Reduction.Mean)
        {
            return new FloatTensor(THSNN_lossNLL(src.Handle, target.Handle, (long)reduction));
        }
    }

    public enum Reduction : long
    {
        None = 0,
        Mean = 1,
        Sum = 2
    }
}
