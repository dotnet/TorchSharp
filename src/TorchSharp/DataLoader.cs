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
                /// <summary>
                /// This class makes easier to create batch. Data set must implement Dataset interface
                /// </summary>
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
                    public DataLoader(Dataset dataset, int batchSize, IEnumerable<long> shuffler, Device device = null, int num_worker = 1, bool drop_last = false)
                        : base(dataset, batchSize, Collate, shuffler, device, num_worker, drop_last)
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
                    public DataLoader(Dataset dataset, int batchSize, bool shuffle = false, Device device = null, int? seed = null, int num_worker = 1, bool drop_last = false)
                        : base(dataset, batchSize, Collate, shuffle, device, seed, num_worker, drop_last)
                    {
                    }

                    private static Dictionary<string, torch.Tensor> Collate(IEnumerable<Dictionary<string, torch.Tensor>> dic, torch.Device device)
                    {
                        Dictionary<string, torch.Tensor> batch = new();
                        foreach (var x in dic.First().Keys) {
                            var t = cat(dic.Select(k => k[x].unsqueeze(0)).ToArray(), 0);
                            if (t.device_type != device.type || t.device_index != device.index)
                                t = t.to(device);
                            batch[x] = t;
                        }
                        return batch;
                    }
                }

                /// <summary>
                /// This class makes easier to create batch. Data set must implement Dataset interface
                /// </summary>
                public class DataLoader<T, S> : IEnumerable<S>, IDisposable
                {
                    private Dataset<T> dataset;
                    private int batchSize;
                    private bool shuffle;
                    private bool drop_last;
                    private Device device;
                    private IEnumerable<long> shuffler;
                    private int num_worker;
                    private Func<IEnumerable<T>, torch.Device, S> collate_fn;

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
                    public DataLoader(Dataset<T> dataset, int batchSize, Func<IEnumerable<T>, torch.Device, S> collate_fn, IEnumerable<long> shuffler, Device device = null, int num_worker = 1, bool drop_last = false)
                    {
                        this.dataset = dataset;
                        this.batchSize = batchSize;
                        this.shuffle = true;
                        this.drop_last = drop_last;
                        this.device = device ?? CPU;
                        this.shuffler = shuffler;
                        this.num_worker = num_worker;
                        this.collate_fn = collate_fn;
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
                    public DataLoader(Dataset<T> dataset, int batchSize, Func<IEnumerable<T>, torch.Device, S> collate_fn, bool shuffle = false, Device device = null, int? seed = null, int num_worker = 1, bool drop_last = false)
                    {
                        this.dataset = dataset;
                        this.batchSize = batchSize;
                        this.shuffle = shuffle;
                        this.drop_last = drop_last;
                        this.device = device ?? CPU;
                        this.shuffler = seed is null ? new FisherYatesShuffler(dataset.Count) : new FisherYatesShuffler(dataset.Count, seed);
                        this.num_worker = num_worker;
                        this.collate_fn = collate_fn;
                    }

                    /// <summary>
                    /// Generate enumerator
                    /// </summary>
                    /// <returns>Enumerator for batch</returns>
                    public IEnumerator<S> GetEnumerator() =>
                        new DataLoaderEnumerator(dataset, batchSize, shuffle, device, shuffler, num_worker, collate_fn);

                    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

                    /// <summary>
                    /// Size of batch
                    /// </summary>
                    public long Count => drop_last ? (dataset.Count / batchSize) : ((dataset.Count - 1) / batchSize + 1);

                    private class DataLoaderEnumerator : IEnumerator<S>
                    {
                        private Dataset<T> dataset;
                        private int batchSize;
                        private Device device;
                        private bool shuffle;
                        private IEnumerable<long> shuffleEnumerable;
                        private IEnumerator<long> shuffler;
                        private long currentVal = 0;
                        private int num_worker = 0;
                        private IList<IDisposable> currentDisposables;
                        private Func<IEnumerable<T>, torch.Device, S> collate_fn;
                        public DataLoaderEnumerator(Dataset<T> dataset, int batchSize, bool shuffle, Device device, IEnumerable<long> shuffleEnumerable, int num_worker, Func<IEnumerable<T>, torch.Device, S> collate_fn)
                        {
                            this.dataset = dataset;
                            this.batchSize = batchSize;
                            this.device = device;
                            this.shuffle = shuffle;
                            this.shuffleEnumerable = shuffleEnumerable;
                            if (num_worker < 1) num_worker = 1;
                            this.num_worker = num_worker;
                            this.collate_fn = collate_fn;
                            Reset();
                        }

                        private bool MoveNextValue()
                        {
                            if (shuffle) {
                                if (!shuffler.MoveNext()) return false;
                                currentVal = shuffler.Current;
                                return true;
                            } else {
                                currentVal++;
                                return currentVal < dataset.Count;
                            }
                        }

                        /// <summary>
                        /// Get next batch
                        /// </summary>
                        /// <returns>true if batch created, false if batch has finished</returns>
                        public bool MoveNext()
                        {
                            DisposeCurrent();
                            using (var scope = DisposeScopeManager.NewDisposeScope()) {
                                if (!MoveNextValue()) return false;

                                var tensorIndexList = new List<long> {currentVal};
                                for (int i = 1; i < batchSize; i++) {
                                    if (!MoveNextValue()) break;
                                    tensorIndexList.Add(currentVal);
                                }

                                var items = new List<T>(new T[tensorIndexList.Count]);
                                var taskedBatchCount = 0;

                                //Run Async
                                var tasks = new List<Task>();
                                foreach (var _ in Enumerable.Range(1, num_worker - 1))
                                    tasks.Add(new(ProcessPendingBatches));
                                tasks.ForEach(x => x.Start());

                                ProcessPendingBatches();

                                foreach (var task in tasks)
                                    task.Wait();

                                using (var collate_scope = DisposeScopeManager.NewDisposeScope()) {
                                    Current = collate_fn(items, device);
                                    currentDisposables = collate_scope.DisposablesView.ToList();
                                    collate_scope.Detach(currentDisposables);
                                }

                                return true;

                                void ProcessPendingBatches()
                                {
                                    while (true) {
                                        var idx = ScheduleBatch();
                                        if (idx is null) break;
                                        items[idx.Value.Item1] = dataset.GetTensor(idx.Value.Item2);
                                    }
                                }

                                (int, long)? ScheduleBatch()
                                {
                                    var t = Interlocked.Increment(ref taskedBatchCount) - 1;
                                    if (t < tensorIndexList.Count)
                                        return (t, tensorIndexList[t]);
                                    return null;
                                }
                            }
                        }

                        /// <summary>
                        /// Reset enumerator
                        /// </summary>
                        public void Reset()
                        {
                            DisposeCurrent();
                            if(shuffle) shuffler = shuffleEnumerable.GetEnumerator();
                            currentVal = -1;
                        }

                        /// <summary>
                        /// Current tensor
                        /// </summary>
                        public S Current { get; private set; }

                        object IEnumerator.Current => Current;

                        public void Dispose()
                        {
                            DisposeCurrent();
                        }

                        private void DisposeCurrent()
                        {
                            if (currentDisposables is null) return;
                            foreach(var x in currentDisposables)
                                x.Dispose();
                            currentDisposables = null;
                        }
                    }

                    public void Dispose()
                    {
                        dataset.Dispose();
                    }
                }
            }
        }
    }
}
