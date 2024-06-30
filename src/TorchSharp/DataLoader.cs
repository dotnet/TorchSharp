// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.SymbolStore;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp.Utils;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class utils
        {
            public static partial class data
            {

                public static Modules.DataLoader DataLoader(
                    Dataset dataset,
                    int batchSize, IEnumerable<long> shuffler,
                    Device device = null,
                    int num_worker = 1, bool drop_last = false,
                    bool disposeBatch = true, bool disposeDataset = true)
                {
                    return new Modules.DataLoader(
                        dataset,
                        batchSize, shuffler,
                        device,
                        num_worker, drop_last,
                        disposeBatch, disposeDataset);
                }

                public static Modules.DataLoader DataLoader(
                    Dataset dataset,
                    int batchSize, bool shuffle = false,
                    Device device = null, int? seed = null,
                    int num_worker = 1, bool drop_last = false,
                    bool disposeBatch = true, bool disposeDataset = true)
                {
                    return new Modules.DataLoader(
                        dataset,
                        batchSize, shuffle,
                        device, seed,
                        num_worker, drop_last,
                        disposeBatch, disposeDataset);
                }

                public static Modules.IterableDataLoader DataLoader(
                    IterableDataset dataset,
                    int batchSize, IEnumerable<long> shuffler,
                    Device device = null,
                    int num_worker = 1, bool drop_last = false,
                    bool disposeBatch = true, bool disposeDataset = true)
                {
                    return new Modules.IterableDataLoader(
                        dataset,
                        batchSize, shuffler,
                        device,
                        num_worker, drop_last,
                        disposeBatch, disposeDataset);
                }

                public static Modules.IterableDataLoader DataLoader(
                    IterableDataset dataset,
                    int batchSize, bool shuffle = false,
                    Device device = null, int? seed = null,
                    int num_worker = 1, bool drop_last = false,
                    bool disposeBatch = true, bool disposeDataset = true)
                {
                    return new Modules.IterableDataLoader(
                        dataset,
                        batchSize, shuffle,
                        device, seed,
                        num_worker, drop_last,
                        disposeBatch, disposeDataset);
                }
            }
        }
    }

    namespace Modules
    {
        using static torch;
        using static torch.utils.data;

        /// <summary>
        /// Data loader. Combines a dataset and a sampler, and provides an enumerator over the given dataset.
        /// </summary>
        /// <remarks>This class is used for map-style data sets</remarks>
        public class DataLoader : DataLoader<Dictionary<string, torch.Tensor>, Dictionary<string, torch.Tensor>>
        {
            /// <summary>
            /// Pytorch style dataloader
            /// </summary>
            /// <param name="dataset">Dataset for create batch</param>
            /// <param name="batchSize">Size of batch</param>
            /// <param name="device">device for output tensor</param>
            /// <param name="shuffler">Shuffler for dataloader</param>
            /// <param name="num_worker">Count of worker</param>
            /// <param name="drop_last">
            /// Set to true to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
            /// If alse and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
            /// </param>
            /// <param name="disposeBatch">
            /// Indicates whether to automatically dispose the collated tensors after an iteration.
            /// </param>
            /// <param name="disposeDataset">
            /// Indicates whether to dispose the dataset when being disposed.
            /// </param>
            public DataLoader(
                Dataset dataset,
                int batchSize, IEnumerable<long> shuffler,
                Device device = null,
                int num_worker = 1, bool drop_last = false,
                bool disposeBatch = true, bool disposeDataset = true)
                : base(dataset,
                      batchSize, Collate, shuffler,
                      device,
                      num_worker, drop_last,
                      disposeBatch, disposeDataset)
            {
            }

            /// <summary>
            /// Pytorch style dataloader
            /// </summary>
            /// <param name="dataset">Dataset for create batch</param>
            /// <param name="batchSize">Size of batch</param>
            /// <param name="shuffle">true if shuffle dataset, false for not</param>
            /// <param name="device">device for output tensor</param>
            /// <param name="seed">Seed for generating shuffle</param>
            /// <param name="num_worker">Count of worker</param>
            /// <param name="drop_last">
            /// Set to true to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
            /// If alse and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
            /// </param>
            /// <param name="disposeBatch">
            /// Indicates whether to automatically dispose the collated tensors after an iteration.
            /// </param>
            /// <param name="disposeDataset">
            /// Indicates whether to dispose the dataset when being disposed.
            /// </param>
            public DataLoader(
                Dataset dataset,
                int batchSize, bool shuffle = false,
                Device device = null, int? seed = null,
                int num_worker = 1, bool drop_last = false,
                bool disposeBatch = true, bool disposeDataset = true)
                : base(dataset,
                      batchSize, Collate, shuffle,
                      device, seed,
                      num_worker, drop_last,
                      disposeBatch, disposeDataset)
            {
            }

            private static Dictionary<string, torch.Tensor> Collate(IEnumerable<Dictionary<string, torch.Tensor>> dic, torch.Device device)
            {
                using (torch.NewDisposeScope()) {
                    Dictionary<string, torch.Tensor> batch = new();
                    foreach (var x in dic.First().Keys) {
                        var t = cat(dic.Select(k => k[x].unsqueeze(0)).ToArray(), 0);
                        if (t.device_type != device.type || t.device_index != device.index)
                            t = t.to(device);
                        batch[x] = t.MoveToOuterDisposeScope();
                    }
                    return batch;
                }
            }
        }

        /// <summary>
        /// Data loader. Combines a dataset and a sampler, and provides an enumerator over the given dataset.
        /// </summary>
        /// <remarks>This class is used for list-style data sets</remarks>
        public class IterableDataLoader : DataLoader<IList<torch.Tensor>, IList<torch.Tensor>>
        {
            /// <summary>
            /// Pytorch style dataloader
            /// </summary>
            /// <param name="dataset">Dataset for create batch</param>
            /// <param name="batchSize">Size of batch</param>
            /// <param name="device">device for output tensor</param>
            /// <param name="shuffler">Shuffler for dataloader</param>
            /// <param name="num_worker">Count of worker</param>
            /// <param name="drop_last">
            /// Set to true to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
            /// If alse and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
            /// </param>
            /// <param name="disposeBatch">
            /// Indicates whether to automatically dispose the collated tensors after an iteration.
            /// </param>
            /// <param name="disposeDataset">
            /// Indicates whether to dispose the dataset when being disposed.
            /// </param>
            public IterableDataLoader(
                IterableDataset dataset,
                int batchSize, IEnumerable<long> shuffler,
                Device device = null,
                int num_worker = 1, bool drop_last = false,
                bool disposeBatch = true, bool disposeDataset = true)
                : base(dataset,
                      batchSize, Collate, shuffler,
                      device,
                      num_worker, drop_last,
                      disposeBatch, disposeDataset)
            {
            }

            /// <summary>
            /// Pytorch style dataloader
            /// </summary>
            /// <param name="dataset">Dataset for create batch</param>
            /// <param name="batchSize">Size of batch</param>
            /// <param name="shuffle">true if shuffle dataset, false for not</param>
            /// <param name="device">device for output tensor</param>
            /// <param name="seed">Seed for generating shuffle</param>
            /// <param name="num_worker">Count of worker</param>
            /// <param name="drop_last">
            /// Set to true to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
            /// If alse and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
            /// </param>
            /// <param name="disposeBatch">
            /// Indicates whether to automatically dispose the collated tensors after an iteration.
            /// </param>
            /// <param name="disposeDataset">
            /// Indicates whether to dispose the dataset when being disposed.
            /// </param>
            public IterableDataLoader(
                IterableDataset dataset,
                int batchSize, bool shuffle = false,
                Device device = null, int? seed = null,
                int num_worker = 1, bool drop_last = false,
                bool disposeBatch = true, bool disposeDataset = true)
                : base(dataset,
                      batchSize, Collate, shuffle,
                      device, seed,
                      num_worker, drop_last,
                      disposeBatch, disposeDataset)
            {
            }

            private static IList<torch.Tensor> Collate(IEnumerable<IList<torch.Tensor>> dic, torch.Device device)
            {
                using (torch.NewDisposeScope()) {
                    List<torch.Tensor> batch = new();
                    for (var x = 0; x < dic.First().Count; x++) {
                        var t = cat(dic.Select(k => k[x].unsqueeze(0)).ToArray(), 0);
                        if (t.device_type != device.type || t.device_index != device.index)
                            t = t.to(device);
                        batch.Add(t.MoveToOuterDisposeScope());
                    }
                    return batch;
                }
            }
        }

#nullable enable
        /// <summary>
        /// This class supports creating batches from data sets.
        /// </summary>
        public class DataLoader<T, S> : IEnumerable<S>, IDisposable
        {
            public Dataset<T> dataset { get; }
            public int batch_size { get; }
            public bool drop_last { get; }
            public IEnumerable<long> sampler { get; }
            public int num_workers { get; }
            public Func<IEnumerable<T>, Device, S> collate_fn { get; }

            public Device Device { get; }
            public bool DisposeBatch { get; }
            public bool DisposeDataset { get; }

            /// <summary>
            /// Pytorch style dataloader
            /// </summary>
            /// <param name="dataset">Dataset for create batch</param>
            /// <param name="batchSize">Size of batch</param>
            /// <param name="collate_fn">Callback to merge items make to a batch</param>
            /// <param name="device">device for output tensor</param>
            /// <param name="shuffler">Shuffler for dataloader</param>
            /// <param name="num_worker">Count of worker</param>
            /// <param name="drop_last">
            /// Set to true to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
            /// If alse and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
            /// </param>
            /// <param name="disposeBatch">
            /// Indicates whether to automatically dispose the collated tensors after an iteration.
            /// </param>
            /// <param name="disposeDataset">
            /// Indicates whether to dispose the dataset when being disposed.
            /// </param>
            public DataLoader(
                Dataset<T> dataset,
                int batchSize,
                Func<IEnumerable<T>, torch.Device, S> collate_fn,
                IEnumerable<long> shuffler,
                Device? device = null,
                int num_worker = 1,
                bool drop_last = false,
                bool disposeBatch = true,
                bool disposeDataset = true)
            {
                this.dataset = dataset;
                this.batch_size = batchSize;
                this.drop_last = drop_last;
                this.Device = device ?? CPU;
                this.sampler = shuffler;
                this.num_workers = Math.Max(num_worker, 1);
                this.collate_fn = collate_fn;
                this.DisposeBatch = disposeBatch;
                this.DisposeDataset = disposeDataset;
            }

            /// <summary>
            /// Pytorch style dataloader
            /// </summary>
            /// <param name="dataset">Dataset for create batch</param>
            /// <param name="batchSize">Size of batch</param>
            /// <param name="collate_fn">Callback to merge items to make a batch</param>
            /// <param name="shuffle">true if shuffle dataset, false for not</param>
            /// <param name="device">device for output tensor</param>
            /// <param name="seed">Seed for generating shuffle</param>
            /// <param name="num_worker">Count of worker</param>
            /// <param name="drop_last">
            /// Set to true to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
            /// If alse and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
            /// </param>
            /// <param name="disposeBatch">
            /// Indicates whether to automatically dispose the collated tensors (a batch) after an iteration.
            /// </param>
            /// <param name="disposeDataset">
            /// Indicates whether to dispose the dataset when being disposed.
            /// </param>
            public DataLoader(
                Dataset<T> dataset,
                int batchSize,
                Func<IEnumerable<T>, torch.Device, S> collate_fn,
                bool shuffle = false,
                Device? device = null,
                int? seed = null,
                int num_worker = 1,
                bool drop_last = false,
                bool disposeBatch = true,
                bool disposeDataset = true) :
                this(dataset, batchSize, collate_fn,
                    shuffle ? new FisherYatesShuffler(dataset.Count, seed) : LongRange(dataset.Count),
                    device, num_worker, drop_last, disposeBatch, disposeDataset)
            { }

            static IEnumerable<long> LongRange(long count)
            {
                for (long i = 0; i < count; i++)
                    yield return i;
            }

            /// <summary>
            /// Generate enumerator
            /// </summary>
            /// <returns>Enumerator for batch</returns>
            public IEnumerator<S> GetEnumerator() => new DataLoaderEnumerator(this);

            IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

            /// <summary>
            /// Size of batch
            /// </summary>
            public long Count => drop_last ? (dataset.Count / batch_size) : ((dataset.Count - 1) / batch_size + 1);

            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            protected virtual void Dispose(bool disposing)
            {
                if (disposing && DisposeDataset) {
                    dataset.Dispose();
                }
            }

            sealed class DataLoaderEnumerator : IEnumerator<S>
            {
                private readonly DataLoader<T, S> loader;
                private IEnumerator<long> shuffler;
                private HashSet<IDisposable>? currentDisposables;
                public DataLoaderEnumerator(DataLoader<T, S> loader)
                {
                    this.loader = loader;
                    // TODO: Use MemberNotNull instead.
                    shuffler = null!;
                    Reset();
                }

                private static void DisposeAll(HashSet<IDisposable> disposables)
                {
                    foreach (var disposable in disposables) {
                        disposable.Dispose();
                    }
                }

                /// <summary>
                /// Get next batch
                /// </summary>
                /// <returns>true if batch created, false if batch has finished</returns>
                public bool MoveNext()
                {
                    DisposeCurrent();

                    var indices = Enumerable.Range(0, loader.batch_size)
                        .Select(_ => shuffler.MoveNext() ? shuffler.Current : (long?)null)
                        .Where(x => x.HasValue)
                        .Cast<long>()
                        .ToArray();

                    if (indices.Length is 0)
                        return false;

                    if (loader.drop_last && indices.Length < loader.batch_size) {
                        return false;
                    }

                    var tensors = new T[indices.Length];
                    var getTensorDisposables = new HashSet<IDisposable>[indices.Length];
                    Enumerable.Range(0, indices.Length)
                        .AsParallel()
                        .WithDegreeOfParallelism(loader.num_workers)
                        .ForAll((i) => {
                            using var getTensorScope = torch.NewDisposeScope();
                            tensors[i] = loader.dataset.GetTensor(indices[i]);
                            getTensorDisposables[i] = getTensorScope.DetachAllAndDispose();
                        });

                    using var collateScope = torch.NewDisposeScope();
                    this.current = loader.collate_fn(tensors, loader.Device);
                    var collateDisposables = collateScope.DetachAllAndDispose();
                    if (loader.DisposeBatch) {
                        this.currentDisposables = collateDisposables;
                    }

                    foreach (var set in getTensorDisposables) {
                        DisposeAll(set);
                    }

                    return true;
                }

                /// <summary>
                /// Reset enumerator
                /// </summary>
                public void Reset()
                {
                    DisposeCurrent();
                    shuffler?.Dispose();
                    shuffler = loader.sampler.GetEnumerator();
                }

                S? current;
                /// <summary>
                /// Current tensor
                /// </summary>
                public S Current => current!;

                object IEnumerator.Current => current!;

                public void Dispose()
                {
                    shuffler.Dispose();
                    DisposeCurrent();
                }

                private void DisposeCurrent()
                {
                    if (this.currentDisposables is not null) {
                        DisposeAll(this.currentDisposables);
                        this.currentDisposables = null;
                    }
                }
            }
        }
    }
}
