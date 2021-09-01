// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

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
