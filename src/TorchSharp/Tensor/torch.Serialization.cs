// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
#nullable enable
namespace TorchSharp
{
    // https://pytorch.org/docs/stable/torch#serialization
    public static partial class torch
    {
        // https://pytorch.org/docs/stable/generated/torch.save
        public static void save(Tensor t, string location) => t.save(location);

        // https://pytorch.org/docs/stable/generated/torch.load
        public static Tensor load(string location) => Tensor.load(location);
    }
}