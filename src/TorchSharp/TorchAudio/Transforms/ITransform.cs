// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp
{
    public partial class torchaudio
    {
        public interface ITransform : IModule<Tensor, Tensor>
        {
        }
    }
}