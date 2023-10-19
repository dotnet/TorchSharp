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
                public abstract class Dataset : Dataset<Dictionary<string, torch.Tensor>>
                {
                }

                public abstract class IterableDataset : Dataset<IList<Tensor>>
                {
                }

                /// <summary>
                /// Interface for Dataloader
                /// </summary>
                public abstract class Dataset<T> : IDisposable
                {
                    public void Dispose()
                    {
                        Dispose(true);
                        GC.SuppressFinalize(this);
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
                    public abstract T GetTensor(long index);

                    protected virtual void Dispose(bool disposing)
                    {
                    }
                }
            }
        }
    }
}