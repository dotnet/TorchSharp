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
                    private int? seed;

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
                        this.seed = seed;
                    }

                    /// <summary>
                    /// Generate enumerator
                    /// </summary>
                    /// <returns>Enumerator for batch</returns>
                    public IEnumerator<Dictionary<string, Tensor>> GetEnumerator() =>
                        new DataLoaderEnumerator(dataset, batchSize, shuffle, device, seed);

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
                        private ShuffleGenerator shuffleGenerator;
                        private int currentVal = 0;
                        private int? seed;
                        public DataLoaderEnumerator(Dataset dataset, int batchSize, bool shuffle, Device device, int? seed)
                        {
                            this.dataset = dataset;
                            this.batchSize = batchSize;
                            this.device = device;
                            this.shuffle = shuffle;
                            this.seed = seed;
                            Reset();
                        }

                        private bool IsFinished() =>
                            shuffle ? !shuffleGenerator.HasNext() : currentVal >= dataset.Count;

                        private long GetNextValue() => shuffle ? shuffleGenerator.Next() : currentVal++;

                        /// <summary>
                        /// Get next batch
                        /// </summary>
                        /// <returns>true if batch created, false if batch has finished</returns>
                        public bool MoveNext()
                        {
                            DisposeCurrent();
                            if (IsFinished()) return false;
                            List<Dictionary<string, Tensor>> dic = new();
                            for (var i = 0; i < batchSize; i++) {
                                if (IsFinished()) break;
                                dic.Add(dataset.GetTensor(GetNextValue()));
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
                            shuffleGenerator = seed is null ? new ShuffleGenerator(dataset.Count) : new ShuffleGenerator(dataset.Count, seed);
                            currentVal = 0;
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