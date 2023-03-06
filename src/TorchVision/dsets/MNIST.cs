// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System.Collections.Generic;
using System.IO;
using System.Net;
using TorchSharp.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class datasets
        {
            /// <summary>
            /// MNIST Dataset
            /// http://yann.lecun.com/exdb/mnist/
            /// </summary>
            /// <param name="root">Root directory of dataset where the MNIST .gz data files exist.</param>
            /// <param name="train">If true, creates dataset from the 'train' files, otherwise from the 't10k' files.</param>
            /// <param name="download">
            /// If true, downloads the dataset from the internet and puts it in root directory.
            /// If the dataset is already downloaded, it is not downloaded again.
            /// </param>
            /// <param name="target_transform">A function/transform that takes in the target and transforms it.</param>
            /// <returns>An iterable dataset.</returns>
            public static Dataset MNIST(string root, bool train, bool download = false, torchvision.ITransform target_transform = null)
            {
                return new Modules.MNIST(root, train, download, target_transform);
            }

            /// <summary>
            /// Fashion-MNIST Dataset
            /// https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/
            /// </summary>
            /// <param name="root">Root directory of dataset where the MNIST .gz data files exist.</param>
            /// <param name="train">If true, creates dataset from the 'train' files, otherwise from the 't10k' files.</param>
            /// <param name="download">
            /// If true, downloads the dataset from the internet and puts it in root directory.
            /// If the dataset is already downloaded, it is not downloaded again.
            /// </param>
            /// <param name="target_transform">A function/transform that takes in the target and transforms it.</param>
            /// <returns>An iterable dataset.</returns>
            public static Dataset FashionMNIST(string root, bool train, bool download = false, torchvision.ITransform target_transform = null)
            {
                return new Modules.FashionMNIST(root, train, download, target_transform);
            }

            /// <summary>
            /// Kuzushiji-MNIST Dataset (https://github.com/rois-codh/kmnist)
            /// </summary>
            /// <param name="root">Root directory of dataset where the KMNIST .gz data files exist.</param>
            /// <param name="train">If true, creates dataset from the 'train' files, otherwise from the 't10k' files.</param>
            /// <param name="download">
            /// If true, downloads the dataset from the internet and puts it in root directory.
            /// If the dataset is already downloaded, it is not downloaded again.
            /// </param>
            /// <param name="target_transform">A function/transform that takes in the target and transforms it.</param>
            /// <returns>An iterable dataset.</returns>
            public static Dataset KMNIST(string root, bool train, bool download = false, torchvision.ITransform target_transform = null)
            {
                return new Modules.KMNIST(root, train, download, target_transform);
            }
        }
    }

    namespace Modules
    {
        /// <summary>
        /// Data reader utility for datasets that follow the MNIST data set's layout:
        ///
        /// A number of single-channel (grayscale) images are laid out in a flat file with four 32-bit integers at the head.
        /// The format is documented at the bottom of the page at: http://yann.lecun.com/exdb/mnist/
        /// </summary>
        internal class MNIST : Dataset
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="root">Path to the folder containing the image files.</param>
            /// <param name="train">The file name prefix, either 'train' or 't10k' (the latter being the test data set).</param>
            /// <param name="download"></param>
            /// <param name="transform">Transform for input MNIST image</param>
            public MNIST(string root, bool train, bool download = false, torchvision.ITransform transform = null) :
                this(root, "mnist", train ? "train" : "t10k", "http://yann.lecun.com/exdb/mnist/", download, transform)
            {
            }

            protected MNIST(string root, string datasetName, string prefix, string baseUrl, bool download, torchvision.ITransform transform)
            {
                if (download) Download(root, baseUrl, datasetName);

                this.transform = transform;

#if NETSTANDARD2_0_OR_GREATER
                var datasetPath = NSPath.Join(root, datasetName, "test_data");
#else
                var datasetPath = Path.Join(root, datasetName, "test_data");
#endif // NETSTANDARD2_0_OR_GREATER

                var dataPath = Path.Combine(datasetPath, prefix + "-images-idx3-ubyte");
                var labelPath = Path.Combine(datasetPath, prefix + "-labels-idx1-ubyte");

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

                // A previous version of this relied on LINQ expressions, but it was about 20% slower.

                for (var i = 0; i < count; i++) {
                    var imgStart = i * imgSize;
                    var floats = new float[imgSize];
                    for (int j = 0; j < imgSize; j++)
                    {
                        floats[j] = dataBytes[j+imgStart] / 256.0f;
                    }
                    data.Add(tensor(floats, new long[] { width, height }));

                    labels.Add(tensor(labelBytes[i], int64));
                }
            }

            private void Download(string root, string baseUrl, string dataset)
            {
#if NETSTANDARD2_0_OR_GREATER
                var datasetPath = NSPath.Join(root, dataset);
#else
                var datasetPath = Path.Join(root, dataset);
#endif // NETSTANDARD2_0_OR_GREATER

                var sourceDir = datasetPath;
                var targetDir = Path.Combine(datasetPath, "test_data");

                if (!Directory.Exists(sourceDir)) {
                    Directory.CreateDirectory(sourceDir);
                }

                DownloadFile("train-images-idx3-ubyte.gz", sourceDir, baseUrl);
                DownloadFile("train-labels-idx1-ubyte.gz", sourceDir, baseUrl);
                DownloadFile("t10k-images-idx3-ubyte.gz", sourceDir, baseUrl);
                DownloadFile("t10k-labels-idx1-ubyte.gz", sourceDir, baseUrl);

                if (!Directory.Exists(targetDir)) {
                    Directory.CreateDirectory(targetDir);
                }

                DecompressFile("train-images-idx3-ubyte", sourceDir, targetDir);
                DecompressFile("train-labels-idx1-ubyte", sourceDir, targetDir);
                DecompressFile("t10k-images-idx3-ubyte", sourceDir, targetDir);
                DecompressFile("t10k-labels-idx1-ubyte", sourceDir, targetDir);
            }

            private static void DecompressFile(string file, string sourceDir, string targetDir)
            {
                var filePath = Path.Combine(targetDir, file);
                if (!File.Exists(filePath))
                    Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, file + ".gz"), targetDir);
            }

            private void DownloadFile(string file, string target, string baseUrl)
            {
#if NETSTANDARD2_0_OR_GREATER
                var filePath = NSPath.Join(target, file);
#else
                var filePath = Path.Join(target, file);
#endif // NETSTANDARD2_0_OR_GREATER

                var netPath = $"{baseUrl}{file}";

                if (!File.Exists(filePath)) {
                    WebClient webClient = new WebClient();
                    webClient.DownloadFile(netPath, filePath);
                }
            }

            private torchvision.ITransform transform;

            /// <summary>
            /// Size of dataset
            /// </summary>
            public override long Count => data.Count;

            private List<Tensor> data = new();
            private List<Tensor> labels = new();

            public override void Dispose()
            {
                data.ForEach(d => d.Dispose());
                labels.ForEach(d => d.Dispose());
            }

            /// <summary>
            /// Get tensor according to index
            /// </summary>
            /// <param name="index">Index for tensor</param>
            /// <returns>Tensors of index. DataLoader will catenate these tensors as batchSize * 784 for image, batchSize * 1 for label</returns>
            public override Dictionary<string, Tensor> GetTensor(long index)
            {
                var rdic = new Dictionary<string, Tensor>();
                if (transform is not null)
                    rdic.Add("data", transform.call(data[(int)index].unsqueeze(0).unsqueeze(0)).squeeze(0));
                else rdic.Add("data", data[(int)index].unsqueeze(0));
                rdic.Add("label", labels[(int)index]);
                return rdic;
            }
        }

        internal class FashionMNIST : MNIST
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="root">Path to the folder containing the image files.</param>
            /// <param name="train">The file name prefix, either 'train' or 't10k' (the latter being the test data set).</param>
            /// <param name="download"></param>
            /// <param name="transform">Transform for input MNIST image</param>
            public FashionMNIST(string root, bool train, bool download = false, torchvision.ITransform transform = null) :
                base(root, "fashion-mnist", train ? "train" : "t10k", "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/", download, transform)
            {
            }
        }

        internal class KMNIST : MNIST
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="root">Path to the folder containing the image files.</param>
            /// <param name="train">The file name prefix, either 'train' or 't10k' (the latter being the test data set).</param>
            /// <param name="download"></param>
            /// <param name="transform">Transform for input MNIST image</param>
            public KMNIST(string root, bool train, bool download = false, torchvision.ITransform transform = null) :
                base(root, "kmnist", train ? "train" : "t10k", "http://codh.rois.ac.jp/kmnist/dataset/kmnist/", download, transform)
            {
            }
        }
    }
}
