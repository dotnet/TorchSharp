// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections;
using System.Collections.Generic;

using TorchSharp.Utils;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class utils
        {
            public static partial class data
            {
                public class DataLoader : IEnumerable<Dictionary<string, Tensor>>, IDisposable
                {
                    private Dataset dataset;
                    private int batchSize;
                    private bool shuffle;
                    private Device device;
                    private int? seed;

                    /// <summary>
                    /// Create pytorch style dataloader
                    /// </summary>
                    /// <param name="dataset"></param>
                    /// <param name="batchSize"></param>
                    /// <param name="shuffle"></param>
                    /// <param name="device"></param>
                    /// <param name="seed"></param>
                    public DataLoader(Dataset dataset, int batchSize, bool shuffle = false, Device device = null, int? seed = null)
                    {
                        this.dataset = dataset;
                        this.batchSize = batchSize;
                        this.shuffle = shuffle;
                        this.device = device ?? CPU;
                        this.seed = seed;
                    }

                    public IEnumerator<Dictionary<string, Tensor>> GetEnumerator() =>
                        new DataLoaderEnumerator(dataset, batchSize, shuffle, device, seed);

                    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

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

                        public bool MoveNext()
                        {
                            if (IsFinished()) return false;
                            Current = dataset.GetTensor(GetNextValue());
                            var currentKeys = Current.Keys;
                            foreach (var x in currentKeys)
                                Current[x].unsqueeze_(0);
                            Dictionary<string, Tensor> dic;
                            for (var i = 1; i < batchSize; i++) {
                                if (IsFinished())
                                    break;
                                dic = dataset.GetTensor(GetNextValue());
                                foreach (var x in currentKeys)
                                    Current[x] = cat(new List<Tensor>() {Current[x], dic[x].unsqueeze(0)}, 0);
                            }

                            foreach (var x in currentKeys)
                                Current[x].to(device);
                            return true;
                        }

                        public void Reset()
                        {
                            shuffleGenerator = seed is null ? new ShuffleGenerator(dataset.Count) : new ShuffleGenerator(dataset.Count, seed);
                            currentVal = 0;
                        }

                        public Dictionary<string, Tensor> Current { get; private set; }

                        object IEnumerator.Current => Current;

                        public void Dispose()
                        {
                            dataset.Dispose();
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