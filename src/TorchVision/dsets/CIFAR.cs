// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;

namespace TorchSharp 
{
    public static partial class torchvision
    {
        public static partial class datasets
        {
            /// <summary>
            /// CIFAR10 Dataset.
            /// </summary>
            /// <remarks>
            /// The dataset for this example can be found at: https://www.cs.toronto.edu/~kriz/cifar.html
            /// </remarks>
            /// <param name="root">Root directory of dataset where directory CIFAR{10,100} exists or will be saved to if download is set to true.</param>
            /// <param name="train">If true, creates dataset from training set, otherwise creates from test set</param>
            /// <param name="download">If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again</param>
            /// <param name="target_transform">A transform that takes in the target tensor and transforms it.</param>
            /// <returns>A CIFAR dataset.</returns>
            public static Dataset CIFAR10(string root, bool train, bool download = false, torchvision.ITransform target_transform = null)
            {
                return new Modules.CIFAR10(root, train, "https://www.cs.toronto.edu/~kriz/", download, target_transform);
            }

            /// <summary>
            /// CIFAR10 Dataset.
            /// </summary>
            /// <remarks>
            /// The dataset for this example can be found at: https://www.cs.toronto.edu/~kriz/cifar.html
            /// </remarks>
            /// <param name="root">Root directory of dataset where directory CIFAR{10,100} exists or will be saved to if download is set to true.</param>
            /// <param name="train">If true, creates dataset from training set, otherwise creates from test set</param>
            /// <param name="download">If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again</param>
            /// <param name="target_transform">A function that takes in the target tensor and transforms it.</param>
            /// <returns>A CIFAR dataset.</returns>
            public static Dataset CIFAR10(string root, bool train, bool download, Func<Tensor, Tensor> target_transform)
            {
                return new Modules.CIFAR10(root, train, "https://www.cs.toronto.edu/~kriz/", download, new torchvision.Lambda(target_transform));
            }

            /// <summary>
            /// CIFAR100 Dataset.
            /// </summary>
            /// <remarks>
            /// The dataset for this example can be found at: https://www.cs.toronto.edu/~kriz/cifar.html
            /// </remarks>
            /// <param name="root">Root directory of dataset where directory CIFAR{10,100} exists or will be saved to if download is set to true.</param>
            /// <param name="train">If true, creates dataset from training set, otherwise creates from test set</param>
            /// <param name="download">If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again</param>
            /// <param name="target_transform">A transform that takes in the target and transforms it.</param>
            /// <returns>A CIFAR dataset.</returns>
            public static Dataset CIFAR100(string root, bool train, bool download = false, torchvision.ITransform target_transform = null)
            {
                return new Modules.CIFAR100(root, train, "https://www.cs.toronto.edu/~kriz/", download, target_transform);
            }

            /// <summary>
            /// CIFAR100 Dataset.
            /// </summary>
            /// <remarks>
            /// The dataset for this example can be found at: https://www.cs.toronto.edu/~kriz/cifar.html
            /// </remarks>
            /// <param name="root">Root directory of dataset where directory CIFAR{10,100} exists or will be saved to if download is set to true.</param>
            /// <param name="train">If true, creates dataset from training set, otherwise creates from test set</param>
            /// <param name="download">If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again</param>
            /// <param name="target_transform">A function that takes in the target and transforms it.</param>
            /// <returns>A CIFAR dataset.</returns>
            public static Dataset CIFAR100(string root, bool train, bool download, Func<Tensor, Tensor> target_transform)
            {
                return new Modules.CIFAR100(root, train, "https://www.cs.toronto.edu/~kriz/", download, new torchvision.Lambda(target_transform));
            }
        }
    }

    namespace Modules
    {
        internal abstract class CIFAR : Dataset
        {
            protected void Download(string targetDir, string baseUrl, string dataset, string fileName)
            {

                if (!Directory.Exists(targetDir)) {
                    Directory.CreateDirectory(targetDir);
                }

                DownloadFile(fileName, targetDir, baseUrl);

                if (!File.Exists(Path.Combine(targetDir, $"{dataset}-binary.bin"))) {
                    DecompressFile(fileName, targetDir, targetDir);
                }
            }

            private static void DecompressFile(string file, string sourceDir, string targetDir)
            {
                Utils.Decompress.ExtractTGZ(Path.Combine(sourceDir, file), targetDir);
            }

            private void DownloadFile(string file, string target, string baseUrl)
            {
                var filePath = JoinPaths(target, file);

                var netPath = $"{baseUrl}{file}";

                if (!File.Exists(filePath)) {
                    WebClient webClient = new WebClient();
                    webClient.DownloadFile(netPath, filePath);
                }
            }

            protected static string JoinPaths(string directory, string file)
            {
#if NETSTANDARD2_0_OR_GREATER
                return NSPath.Join(directory, file);
#else
                return Path.Join(directory, file);
#endif // NETSTANDARD2_0_OR_GREATER
            }
        }

        internal class CIFAR10 : CIFAR, IDisposable
        {
            public CIFAR10(string root, bool train, string baseUrl, bool download, torchvision.ITransform transform)
            {
                var targetDir = JoinPaths(root, "CIFAR10");

                if (download) Download(targetDir, baseUrl, "cifar-10", "cifar-10-binary.tar.gz");

                this.transform = transform;

                var dataPath = Path.Combine(targetDir, "cifar-10-batches-bin");

                if (!train) {
                    _count = ReadSingleFile(Path.Combine(dataPath, "test_batch.bin"));
                } else {
                    _count += ReadSingleFile(Path.Combine(dataPath, "data_batch_1.bin"));
                    _count += ReadSingleFile(Path.Combine(dataPath, "data_batch_2.bin"));
                    _count += ReadSingleFile(Path.Combine(dataPath, "data_batch_3.bin"));
                    _count += ReadSingleFile(Path.Combine(dataPath, "data_batch_4.bin"));
                    _count += ReadSingleFile(Path.Combine(dataPath, "data_batch_5.bin"));
                }

            }

            private int ReadSingleFile(string path)
            {
                const int height = 32;
                const int width = 32;
                const int channels = 3;
                const int count = 10000;

                byte[] dataBytes = File.ReadAllBytes(path);

                var imgSize = channels * height * width + 1;

                if (dataBytes.Length != imgSize * count)
                    throw new InvalidDataException($"Not a proper CIFAR10 file: {path}");

                // Go through the data and create tensors

                // A previous version of this relied on LINQ expressions, but it was about 20% slower.

                for (var i = 0; i < count; i++) {

                    var imgStart = i * imgSize;

                    labels.Add(tensor(dataBytes[imgStart], int64));

                    var floats = new float[imgSize-1];
                    for (int j = 0; j < imgSize-1; j++) {
                        floats[j] = dataBytes[1 + j + imgStart] / 256.0f;
                    }

                    data.Add(tensor(floats, new long[] { channels, width, height }));
                }

                return count;
            }

            public override void Dispose()
            {
                data.ForEach(d => d.Dispose());
                labels.ForEach(d => d.Dispose());
            }

            /// <summary>
            /// Get tensor according to index
            /// </summary>
            /// <param name="index">Index for tensor</param>
            /// <returns>Tensors of index. DataLoader will catenate these tensors into batches.</returns>
            public override Dictionary<string, Tensor> GetTensor(long index)
            {
                var rdic = new Dictionary<string, Tensor>();
                if (transform is not null) {
                    rdic.Add("data", transform.call(data[(int)index]));
                }
                else {
                    rdic.Add("data", data[(int)index]);
                }
                rdic.Add("label", labels[(int)index]);
                return rdic;
            }

            private List<Tensor> data = new();
            private List<Tensor> labels = new();

            private torchvision.ITransform transform;
            public override long Count => _count;

            private int _count = 0;
        }

        internal class CIFAR100 : CIFAR, IDisposable
        {
            public CIFAR100(string root, bool train, string baseUrl, bool download, torchvision.ITransform transform)
            {
                var targetDir = JoinPaths(root, "CIFAR100");

                if (download) Download(targetDir, baseUrl, "cifar-100", "cifar-100-binary.tar.gz");

                this.transform = transform;

                var dataPath = Path.Combine(targetDir, "cifar-100-binary");

                if (train) {
                    _count = ReadSingleFile(Path.Combine(dataPath, "train.bin"));
                } else {
                    _count = ReadSingleFile(Path.Combine(dataPath, "test.bin"));
                }

            }

            private int ReadSingleFile(string path)
            {
                const int height = 32;
                const int width = 32;
                const int channels = 3;

                byte[] dataBytes = File.ReadAllBytes(path);

                var imgSize = channels * height * width;
                var recSize = imgSize + 2;

                if (dataBytes.Length % recSize  != 0)
                    throw new InvalidDataException($"Not a proper CIFAR100 file: {path}");

                int count = dataBytes.Length / recSize;

                // Go through the data and create tensors

                for (var i = 0; i < count; i++) {

                    var recStart = i * recSize;
                    var imgStart = recStart + 2;

                    // CIFAR100 has two labels -- one, which is 20 categories, one which is 100 entities.
                    coarse_labels.Add(tensor(dataBytes[recStart], int64));
                    fine_labels.Add(tensor(dataBytes[recStart+1], int64));

                    var floats = new float[imgSize];
                    for (int j = 0; j < imgSize; j++) {
                        floats[j] = dataBytes[j + imgStart] / 256.0f;
                    }

                    data.Add(tensor(floats, new long[] { channels, width, height }));
                }

                return count;
            }

            public override void Dispose()
            {
                data.ForEach(d => d.Dispose());
                fine_labels.ForEach(d => d.Dispose());
                coarse_labels.ForEach(d => d.Dispose());
            }

            /// <summary>
            /// Get tensor according to index
            /// </summary>
            /// <param name="index">Index for tensor</param>
            /// <returns>Tensors of index. DataLoader will catenate these tensors into batches.</returns>
            public override Dictionary<string, Tensor> GetTensor(long index)
            {
                var rdic = new Dictionary<string, Tensor>();
                if (transform is not null) {
                    rdic.Add("data", transform.call(data[(int)index]));
                } else {
                    rdic.Add("data", data[(int)index]);
                }
                rdic.Add("label", fine_labels[(int)index]);
                rdic.Add("categories", coarse_labels[(int)index]);
                return rdic;
            }

            private List<Tensor> data = new();
            private List<Tensor> coarse_labels = new();
            private List<Tensor> fine_labels = new();

            private torchvision.ITransform transform;
            public override long Count => _count;

            private int _count = 0;
        }
    }
}
