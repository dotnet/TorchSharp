using System;
using System.Collections.Generic;
using System.Text;

using TorchSharp.Tensor;

namespace TorchSharp.TorchVision
{
    public interface ITransform
    {
        TorchTensor forward(TorchTensor input);
    }
}
