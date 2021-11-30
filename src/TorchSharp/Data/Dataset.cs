// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using static TorchSharp.torch;

namespace TorchSharp.Data
{
    public interface Dataset: IDisposable
    {
        public int Count { get; }
        public Dictionary<string, Tensor> GetTensor(int index);
    }
}