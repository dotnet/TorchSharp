using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

using static TorchSharp.torch;

namespace TorchSharp.Data
{
    public class DataLoader: IEnumerable<(Tensor, Tensor)>
    {
        private Dataset dataset;
        private int batchSize;
        private bool shuffle;
        private Device device;

        public DataLoader(Dataset dataset, int batchSize, bool shuffle = false, Device device = null)
        {
            this.dataset = dataset;
            this.batchSize = batchSize;
            this.shuffle = shuffle;
            this.device = device ?? CPU;
        }
        public IEnumerator<(Tensor, Tensor)> GetEnumerator()
        {
            return new DataLoaderEnumerator(dataset, batchSize, shuffle, device);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public long Count => (dataset.Count - 1) / batchSize + 1;

        private class DataLoaderEnumerator : IEnumerator<(Tensor, Tensor)>
        {
            private Dataset dataset;
            private int batchSize;
            private Device device;
            private bool shuffle;
            public DataLoaderEnumerator(Dataset dataset, int batchSize, bool shuffle, Device device)
            {
                this.dataset = dataset;
                this.batchSize = batchSize;
                this.device = device;
                this.shuffle = shuffle;
                reset();
            }

            private Tensor dataTensor;
            private Tensor labelTensor;

            private bool isFinished()
            {
                throw new NotImplementedException();
            }
            private int getNextValue()
            {
                throw new NotImplementedException();
            }

            private void reset()
            {
                throw new NotImplementedException();
            }
            public bool MoveNext()
            {
                if (isFinished()) return false;
                (dataTensor, labelTensor) = dataset.GetTensor(getNextValue());
                dataTensor.unsqueeze_(0);
                for (var i = 1; i < batchSize; i++)
                {
                    if (isFinished())
                        break;
                    var (data, label) = dataset.GetTensor(getNextValue());
                    dataTensor = cat(new List<Tensor> {dataTensor, data.unsqueeze(0)}, 0);
                    labelTensor = cat(new List<Tensor> {labelTensor, label}, 0);
                }
                return true;
            }

            public void Reset() => reset();

            public (Tensor, Tensor) Current => (dataTensor.to(device), labelTensor.to(device));

            object IEnumerator.Current => Current;

            public void Dispose()
            {
                dataset.Dispose();
            }
        }
    }
}