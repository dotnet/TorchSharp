// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

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
                public abstract class Dataset : Dataset<Dictionary<string, Tensor>>,
                    IDataset<IReadOnlyDictionary<string, Tensor>>
                {
                    // Due to covariation, it should naturally be IDataset<IReadOnlyDictionary<string, Tensor>>.
                    // However FSharp.Examples will break down without this.
                    IReadOnlyDictionary<string, Tensor> IDataset<IReadOnlyDictionary<string, Tensor>>.this[long index] => this[index];
                }

                /// <summary>
                /// Iterable-style data sets
                /// </summary>
                public abstract class IterableDataset : Dataset<IList<Tensor>>,
                    IDataset<IEnumerable<Tensor>>
                {
                    // Due to covariation, it should naturally be IDataset<IEnumerable<Tensor>>.
                    // However FSharp.Examples will break down without this.
                    IEnumerable<Tensor> IDataset<IEnumerable<Tensor>>.this[long index] => this[index];
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

                    /// <summary>
                    /// Size of dataset
                    /// </summary>
                    public abstract long Count { get; }

                    [IndexerName("DatasetItems")]
                    public T this[long index] => this.GetTensor(index);

                    // GetTensor is kept for compatibility.
                    // Perhaps we should remove that and make the indexer abstract later.

                    /// <summary>
                    /// Get tensor according to index
                    /// </summary>
                    /// <param name="index">Index for tensor</param>
                    /// <returns>Tensors of index. DataLoader will catenate these tensors into batches.</returns>
                    public abstract T GetTensor(long index);

                    protected virtual void Dispose(bool disposing)
                    {
                        IDataset<Dictionary<string, string>> a = null;
                        IDataset<IReadOnlyDictionary<string, string>> b = a;
                    }
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
                    [IndexerName("DatasetItems")]
                    T this[long index] { get; }

                    // TODO: support System.Index
                }
            }
        }
    }
}