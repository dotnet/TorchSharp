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
                /// <summary>
                /// Interface for Dataloader
                /// </summary>
                public abstract class Dataset : IDisposable
                {
                    public virtual void Dispose()
                    {
                    }

                    /// <summary>
                    /// Size of dataset
                    /// </summary>
                    public abstract long Count { get; }

                    /// <summary>
                    /// Get tensor according to index
                    /// </summary>
                    /// <param name="index">Index for tensor</param>
                    /// <returns>Tensors of index. DataLoader will catenate these tensors.</returns>
                    public abstract Dictionary<string, Tensor> GetTensor(long index);
                }
            }
        }
    }
}