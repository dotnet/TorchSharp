// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp.torchvision;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;

namespace TorchSharp.torchvision.dsets
{
    /// <summary>
    /// Data reader utility for datasets that follow the MNIST data set's layout:
    ///
    /// A number of single-channel (grayscale) images are laid out in a flat file with four 32-bit integers at the head.
    /// The format is documented at the bottom of the page at: http://yann.lecun.com/exdb/mnist/
    /// </summary>
    public sealed class MNISTReader : Dataset
    {
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="path">Path to the folder containing the image files.</param>
        /// <param name="prefix">The file name prefix, either 'train' or 't10k' (the latter being the test data set).</param>
        /// <param name="device">The device, i.e. CPU or GPU to place the output tensors on.</param>
        /// <param name="transform"></param>
        public MNISTReader(string path, string prefix, Device device = null, ITransform transform = null)
        {
            // The MNIST data set is small enough to fit in memory, so let's load it there.

            this.transform = transform;

            var dataPath = Path.Combine(path, prefix + "-images-idx3-ubyte");
            var labelPath = Path.Combine(path, prefix + "-labels-idx1-ubyte");

            var count = -1;
            var height = 0;
            var width = 0;

            byte[] dataBytes = null;
            byte[] labelBytes = null;

            using (var file = File.Open(dataPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var rdr = new System.IO.BinaryReader(file)) {

                var reader = new BigEndianReader(rdr);
                var x = reader.ReadInt32(); // Magic number
                count = reader.ReadInt32();

                height = reader.ReadInt32();
                width = reader.ReadInt32();

                // Read all the data into memory.
                dataBytes = reader.ReadBytes(height * width * count);
            }

            using (var file = File.Open(labelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var rdr = new System.IO.BinaryReader(file)) {

                var reader = new BigEndianReader(rdr);
                var x = reader.ReadInt32(); // Magic number
                var lblcnt = reader.ReadInt32();

                if (lblcnt != count) throw new InvalidDataException("Image data and label counts are different.");

                // Read all the data into memory.
                labelBytes = reader.ReadBytes(lblcnt);
            }

            var imgSize = height * width;

            // Go through the data and create tensors
            for (var i = 0; i < count; i++) {
                var imgStart = i * imgSize;

                data.Add(tensor(dataBytes[imgStart..(imgStart+imgSize)].Select(b => b/256.0f).ToArray(), new long[] { width, height }));
                labels.Add(tensor(labelBytes[i], int64));
            }
        }

        private ITransform transform;

        public override long Count => data.Count;

        private List<Tensor> data = new();
        private List<Tensor> labels = new();

        public override void Dispose()
        {
            data.ForEach(d => d.Dispose());
            labels.ForEach(d => d.Dispose());
        }

        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            var rdic = new Dictionary<string, Tensor>();
            if (transform is not null)
                rdic.Add("data", transform.forward(data[(int)index].unsqueeze(0).unsqueeze(0)).squeeze(0));
            else rdic.Add("data", data[(int)index].unsqueeze(0));
            rdic.Add("label", labels[(int)index]);
            return rdic;
        }
    }
}
