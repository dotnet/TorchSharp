using System;
using System.Collections.Generic;
using System.Text;

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    public interface ITransform
    {
        Tensor forward(Tensor input);
    }
}
