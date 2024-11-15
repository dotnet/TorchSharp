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
                /// Map-style data set
                /// </summary>
                public abstract class Dataset : Dataset<Dictionary<string, torch.Tensor>>
                {
                }

                /// <summary>
                /// Iterable-style data sets
                /// </summary>
                public abstract class IterableDataset : Dataset<IList<Tensor>>
                {
                }

                /// <summary>
                /// The base interface for all Datasets.
                /// </summary>
                public interface IDataset<out T> : IDisposable
                {
                    /// <summary>
                    /// Size of dataset
                    /// </summary>
                    long Count { get; }

                    /// <summary>
                    /// Get tensor according to index
                    /// </summary>
                    /// <param name="index">Index for tensor</param>
                    /// <returns>Tensors of index. DataLoader will catenate these tensors into batches.</returns>
                    T this[long index] { get; }
                }

                /// <summary>
                /// The base nterface for all Datasets.
                /// </summary>
                public abstract class Dataset<T> : IDataset<T>, IDisposable
                {
                    public void Dispose()
                    {
                        Dispose(true);
                        GC.SuppressFinalize(this);
                    }

                    /// <inheritdoc />
                    public abstract long Count { get; }

                    /// <summary>
                    /// Get tensor according to index
                    /// </summary>
                    /// <param name="index">Index for tensor</param>
                    /// <returns>Tensors of index. DataLoader will catenate these tensors into batches.</returns>
                    public abstract T GetTensor(long index);

                    /// <inheritdoc />
                    public T this[long index] => GetTensor(index);

                    protected virtual void Dispose(bool disposing)
                    {
                    }
                }
            }
        }
    }
}