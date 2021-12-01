// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class utils
        {
            public static partial class data
            {
                public abstract class Dataset : IDisposable
                {
                    public virtual void Dispose()
                    {
                    }

                    public abstract int Count { get; }

                    public abstract Dictionary<string, Tensor> GetTensor(int index);
                }
            }
        }
    }
}