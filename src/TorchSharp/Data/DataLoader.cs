// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.Collections;
using System.Collections.Generic;

using static TorchSharp.torch;

namespace TorchSharp.Data
{
    public class DataLoader: IEnumerable<Dictionary<string, Tensor>>
    {
        private Dataset dataset;
        private int batchSize;
        private bool shuffle;
        private Device device;

        /// <summary>
        /// Create pytorch style dataloader
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="batchSize"></param>
        /// <param name="shuffle"></param>
        /// <param name="device"></param>
        public DataLoader(Dataset dataset, int batchSize, bool shuffle = false, Device device = null)
        {
            this.dataset = dataset;
            this.batchSize = batchSize;
            this.shuffle = shuffle;
            this.device = device ?? CPU;
        }

        public IEnumerator<Dictionary<string, Tensor>> GetEnumerator() => new DataLoaderEnumerator(dataset, batchSize, shuffle, device);

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

            public DataLoaderEnumerator(Dataset dataset, int batchSize, bool shuffle, Device device)
            {
                this.dataset = dataset;
                this.batchSize = batchSize;
                this.device = device;
                this.shuffle = shuffle;
                reset();
            }

            private bool isFinished() => shuffle ? !shuffleGenerator.hasNext() : currentVal >= dataset.Count;

            private int getNextValue() => shuffle ? shuffleGenerator.next() : currentVal++;

            private void reset()
            {
                shuffleGenerator = new ShuffleGenerator(dataset.Count);
                currentVal = 0;
            }

            public bool MoveNext()
            {
                if (isFinished()) return false;
                Current = dataset.GetTensor(getNextValue());
                var currentKeys = Current.Keys;
                foreach (var x in currentKeys)
                    Current[x].unsqueeze_(0);
                Dictionary<string, Tensor> dic;
                for (var i = 1; i < batchSize; i++)
                {
                    if (isFinished())
                        break;
                    dic = dataset.GetTensor(getNextValue());
                    foreach (var x in currentKeys)
                        Current[x] = cat(new List<Tensor>() {dic[x].unsqueeze(0), Current[x]}, 0);
                }
                return true;
            }

            public void Reset() => reset();

            public Dictionary<string, Tensor> Current { get; private set; }

            object IEnumerator.Current => Current;

            public void Dispose()
            {
                dataset.Dispose();
            }
        }
    }
}