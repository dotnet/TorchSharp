using System;
using System.Collections.Generic;

using static TorchSharp.torch;

namespace TorchSharp.Data
{
    public interface Dataset: IDisposable
    {
        public long Count { get; }
        public (Tensor, Tensor) GetTensor(int index);
    }
}