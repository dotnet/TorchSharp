// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp.torchvision;
using static TorchSharp.torch;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Data reader utility for datasets that follow the MNIST data set's layout:
    ///
    /// A number of single-channel (grayscale) images are laid out in a flat file with four 32-bit integers at the head.
    /// The format is documented at the bottom of the page at: http://yann.lecun.com/exdb/mnist/
    /// </summary>
    public sealed class MNISTReader : IEnumerable<(Tensor, Tensor)>, IDisposable
    {
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="path">Path to the folder containing the image files.</param>
        /// <param name="prefix">The file name prefix, either 'train' or 't10k' (the latter being the test data set).</param>
        /// <param name="batch_size">The batch size</param>
        /// <param name="shuffle">Randomly shuffle the images.</param>
        /// <param name="device">The device, i.e. CPU or GPU to place the output tensors on.</param>
        /// <param name="transform"></param>
        public MNISTReader(string path, string prefix, int batch_size = 32, bool shuffle = false, torch.Device device = null, ITransform transform = null)
        {
            // The MNIST data set is small enough to fit in memory, so let's load it there.

            BatchSize = batch_size;

            var dataPath = Path.Combine(path, prefix + "-images-idx3-ubyte");
            var labelPath = Path.Combine(path, prefix + "-labels-idx1-ubyte");

            var count = -1;
            var height = 0;
            var width = 0;

            byte[] dataBytes = null;
            byte[] labelBytes = null;

            using (var file = File.Open(dataPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var rdr = new System.IO.BinaryReader(file)) {

                var reader = new Utils.BigEndianReader(rdr);
                var x = reader.ReadInt32(); // Magic number
                count = reader.ReadInt32();

                height = reader.ReadInt32();
                width = reader.ReadInt32();

                // Read all the data into memory.
                dataBytes = reader.ReadBytes(height * width * count);
            }

            using (var file = File.Open(labelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var rdr = new System.IO.BinaryReader(file)) {

                var reader = new Utils.BigEndianReader(rdr);
                var x = reader.ReadInt32(); // Magic number
                var lblcnt = reader.ReadInt32();

                if (lblcnt != count) throw new InvalidDataException("Image data and label counts are different.");

                // Read all the data into memory.
                labelBytes = reader.ReadBytes(lblcnt);
            }

            // Set up the indices array.
            Random rnd = new Random();
            var indices = !shuffle ?
                Enumerable.Range(0, count).ToArray() :
                Enumerable.Range(0, count).OrderBy(c => rnd.Next()).ToArray();

            var imgSize = height * width;

            // Go through the data and create tensors
            for (var i = 0; i < count;) {

                var take = Math.Min(batch_size, Math.Max(0, count - i));

                if (take < 1) break;

                var dataTensor = torch.zeros(new long[] { take, imgSize}, device: device);
                var lablTensor = torch.zeros(new long[] { take }, torch.int64, device: device);

                // Take
                for (var j = 0; j < take; j++) {
                    var idx = indices[i++];
                    var imgStart = idx * imgSize;

                    var floats = dataBytes[imgStart.. (imgStart+imgSize)].Select(b => b/256.0f).ToArray();
                    using (var inputTensor = torch.tensor(floats))
                        dataTensor.index_put_(inputTensor, TensorIndex.Single(j));
                    lablTensor[j] = torch.tensor(labelBytes[idx], torch.int64);
                }

                var batch = dataTensor.reshape(take, 1, height, width);

                if (transform != null) {
                    // Carefully dispose the original
                    using(var batch_copy = batch)
                    batch = transform.forward(batch);
                }

                data.Add(batch);
                dataTensor.Dispose();
                labels.Add(lablTensor);
            }

            Size = count;
        }

        public int Size { get; set; }

        public int BatchSize { get; private set; }

        private List<Tensor> data = new List<Tensor>();
        private List<Tensor> labels = new List<Tensor>();

        public IEnumerator<(Tensor, Tensor)> GetEnumerator()
        {
            return new MNISTEnumerator(data, labels);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void Dispose()
        {
            data.ForEach(d => d.Dispose());
            labels.ForEach(d => d.Dispose());
        }

        private class MNISTEnumerator : IEnumerator<(Tensor, Tensor)>
        {
            public MNISTEnumerator(List<Tensor> data, List<Tensor> labels)
            {
                this.data = data;
                this.labels = labels;
            }

            public (Tensor, Tensor) Current {
                get {
                    if (curIdx == -1) throw new InvalidOperationException("Calling 'Current' before 'MoveNext()'");
                    return (data[curIdx], labels[curIdx]);
                }
            }

            object IEnumerator.Current => Current;

            public void Dispose()
            {
            }

            public bool MoveNext()
            {
                curIdx += 1;
                return curIdx < data.Count;
            }

            public void Reset()
            {
                curIdx = -1;
            }

            private int curIdx = -1;
            private List<Tensor> data = null;
            private List<Tensor> labels = null;
        }
    }
}
