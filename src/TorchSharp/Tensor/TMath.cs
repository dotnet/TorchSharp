using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp.Tensor
{
    public static class TMath
    {
        public static TorchTensor Exp(TorchTensor value)
        {
            return value.Exp();
        }

        public static TorchTensor Erf(TorchTensor value)
        {
            return value.Erf();
        }

        public static TorchTensor Log(TorchTensor value)
        {
            return value.Log();
        }
    }
}
