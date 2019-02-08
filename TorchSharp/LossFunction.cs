using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp
{
    public class LossFunction
    {
        [DllImport("LibTorchSharp")]
        extern static FloatTensor.HType Loss_mse(FloatTensor.HType srct, FloatTensor.HType trgt, long reduction);

        public static FloatTensor MSELoss(FloatTensor src, FloatTensor target, Reduction reduction = Reduction.None)
        {
            return new FloatTensor(Loss_mse(src.handle, target.handle, (long)reduction));
        }
    }

    public enum Reduction : long
    {
        None = 0,
        Mean = 1,
        Sum = 2
    }
}
