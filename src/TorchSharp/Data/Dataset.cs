using System;
using System.Collections.Generic;

using static TorchSharp.torch;

namespace TorchSharp.Data
{
    public interface Dataset: IDisposable
    {
        public long Count();
        public (Tensor, Tensor) GetTensor(int index);
    }
}