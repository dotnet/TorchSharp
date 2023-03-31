// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        /// <summary>
        /// This is essentially an alias. We're keeping it because it was
        /// introduced and thus leaving it in will avoid a breaking change.
        /// </summary>
        public interface ITransform : nn.IModule<Tensor, Tensor>
        {
        }
    }
}
