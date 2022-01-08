// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
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

                    /// <summary>
                    /// Pytorch style dataloader
                    /// </summary>
                    /// <param name="dataset">Dataset for create batch</param>
                    /// <param name="batchSize">Size of batch</param>
                    /// <param name="device">device for output tensor</param>
                    /// <param name="shuffler">Shuffler for dataloader</param>
                    public DataLoader(Dataset dataset, int batchSize, IEnumerable<long> shuffler, Device device = null)
                    {
                        this.dataset = dataset;
                        this.batchSize = batchSize;
                        this.shuffle = true;
                        this.device = device ?? CPU;
                        this.shuffler = shuffler;
                    }

                    /// <summary>
                    /// Pytorch style dataloader
                    /// </summary>
                    /// <param name="dataset">Dataset for create batch</param>
                    /// <param name="batchSize">Size of batch</param>
                    /// <param name="shuffle">true if shuffle dataset, false for not</param>
                    /// <param name="device">device for output tensor</param>
                    /// <param name="seed">Seed for generating shuffle</param>
                    public DataLoader(Dataset dataset, int batchSize, bool shuffle = false, Device device = null, int? seed = null)
                    {
                        this.dataset = dataset;
                        this.batchSize = batchSize;
                        this.shuffle = shuffle;
                        this.device = device ?? CPU;
                        this.shuffler = seed is null ? new FisherYatesShuffler(dataset.Count) : new FisherYatesShuffler(dataset.Count, seed);
                    }

                    /// <summary>
                    /// Generate enumerator
                    /// </summary>
                    /// <returns>Enumerator for batch</returns>
                    public IEnumerator<Dictionary<string, Tensor>> GetEnumerator() =>
                        new DataLoaderEnumerator(dataset, batchSize, shuffle, device, shuffler);

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
                        public DataLoaderEnumerator(Dataset dataset, int batchSize, bool shuffle, Device device, IEnumerable<long> shuffleEnumerable)
                        {
                            this.dataset = dataset;
                            this.batchSize = batchSize;
                            this.device = device;
                            this.shuffle = shuffle;
                            this.shuffleEnumerable = shuffleEnumerable;
                            Reset();
                        }
                        private bool MoveNextValue()
                        {
                            if (shuffle) {
                                if (!shuffler.MoveNext()) return false;
                                currentVal = shuffler.Current;
                                return true;
                            }
                            else {
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
                            List<Dictionary<string, Tensor>> dic = new();
                            dic.Add(dataset.GetTensor(currentVal));
                            for (var i = 1; i < batchSize; i++) {
                                if (!MoveNextValue()) break;
                                dic.Add(dataset.GetTensor(currentVal));
                            }

                            Current = new();
                            foreach (var x in dic[0].Keys)
                                Current[x] = cat(dic.Select(k => k[x].unsqueeze(0)).ToArray(), 0).to(device);
                            return true;
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
                            dataset.Dispose();
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