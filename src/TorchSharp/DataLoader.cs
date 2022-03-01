// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
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
                public class DataLoader : IEnumerable<Dictionary<string, Tensor>>, IDisposable
                {
                    private Dataset dataset;
                    private int batchSize;
                    private bool shuffle;
                    private Device device;
                    private IEnumerable<long> shuffler;
                    private int num_worker;

                    /// <summary>
                    /// Pytorch style dataloader
                    /// </summary>
                    /// <param name="dataset">Dataset for create batch</param>
                    /// <param name="batchSize">Size of batch</param>
                    /// <param name="device">device for output tensor</param>
                    /// <param name="shuffler">Shuffler for dataloader</param>
                    /// <param name="num_worker">Count of worker</param>
                    public DataLoader(Dataset dataset, int batchSize, IEnumerable<long> shuffler, Device device = null, int num_worker = 1)
                    {
                        this.dataset = dataset;
                        this.batchSize = batchSize;
                        this.shuffle = true;
                        this.device = device ?? CPU;
                        this.shuffler = shuffler;
                        this.num_worker = num_worker;
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
                    public DataLoader(Dataset dataset, int batchSize, bool shuffle = false, Device device = null, int? seed = null, int num_worker = 1)
                    {
                        this.dataset = dataset;
                        this.batchSize = batchSize;
                        this.shuffle = shuffle;
                        this.device = device ?? CPU;
                        this.shuffler = seed is null ? new FisherYatesShuffler(dataset.Count) : new FisherYatesShuffler(dataset.Count, seed);
                        this.num_worker = num_worker;
                    }

                    /// <summary>
                    /// Generate enumerator
                    /// </summary>
                    /// <returns>Enumerator for batch</returns>
                    public IEnumerator<Dictionary<string, Tensor>> GetEnumerator() =>
                        new DataLoaderEnumerator(dataset, batchSize, shuffle, device, shuffler, num_worker);

                    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

                    /// <summary>
                    /// Size of batch
                    /// </summary>
                    public long Count => (dataset.Count - 1) / batchSize + 1;

                    private class DataLoaderEnumerator : IEnumerator<Dictionary<string, Tensor>>
                    {
                        private Dataset dataset;
                        private int batchSize;
                        private Device device;
                        private bool shuffle;
                        private IEnumerable<long> shuffleEnumerable;
                        private IEnumerator<long> shuffler;
                        private long currentVal = 0;
                        private int num_worker = 0;
                        public DataLoaderEnumerator(Dataset dataset, int batchSize, bool shuffle, Device device, IEnumerable<long> shuffleEnumerable, int num_worker)
                        {
                            this.dataset = dataset;
                            this.batchSize = batchSize;
                            this.device = device;
                            this.shuffle = shuffle;
                            this.shuffleEnumerable = shuffleEnumerable;
                            if (num_worker < 1) num_worker = 1;
                            this.num_worker = num_worker;
                            Reset();
                        }

                        private object moveNextLock = new();

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
                            if (!MoveNextValue()) return false;

                            var tensorIndexList = new List<long> {currentVal};
                            for (int i = 1; i < batchSize; i++) {
                                if (!MoveNextValue()) break;
                                tensorIndexList.Add(currentVal);
                            }

                            ConcurrentBag<Dictionary<string, Tensor>> dic = new();
                            var taskedBatchCount = 0;
                            var taskBatchLock = new object();

                            //Run Async
                            foreach(var _ in Enumerable.Range(1, num_worker - 1))
                                ThreadPool.QueueUserWorkItem(CreateBatch);

                            CreateBatch(null);

                            while (dic.Count < tensorIndexList.Count) { } //Wait for tensors ready

                            Current = new();
                            foreach (var x in dic.First().Keys)
                                Current[x] = cat(dic.Select(k => k[x].unsqueeze(0)).ToArray(), 0).to(device);
                            return true;

                            void CreateBatch(object _)
                            {
                                while (true) {
                                    var idx = TaskBatch();
                                    if (idx is null) break;
                                    dic.Add(dataset.GetTensor(idx.Value));
                                }
                            }

                            long? TaskBatch()
                            {
                                lock (taskBatchLock) {
                                    if (taskedBatchCount < tensorIndexList.Count)
                                        return tensorIndexList[taskedBatchCount++];
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
                        public Dictionary<string, Tensor> Current { get; private set; }

                        object IEnumerator.Current => Current;

                        public void Dispose()
                        {
                            DisposeCurrent();
                        }

                        private void DisposeCurrent()
                        {
                            if (Current is null) return;
                            foreach(var x in Current.Values)
                                x.Dispose();
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
