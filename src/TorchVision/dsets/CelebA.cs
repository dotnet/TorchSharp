// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision.datasets;
using static TorchSharp.torchvision.datasets.utils;

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/b78d98bb152ffb9c0c0f5365f59f475c70b1784e/torchvision/datasets/celeba.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

#nullable enable
namespace TorchSharp
{
    namespace Modules
    {
        internal sealed class CelebA : Dataset
        {
            private static readonly (string, string, string)[] file_list = new (string, string, string)[] {
                // File ID                                      MD5 Hash                            Filename
                ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
                // ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
                // ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
                ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
                ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
                ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
                ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
                // ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
                ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
            };

            private const string base_folder = "celeba";
            private readonly string root;
            private readonly CelebADatasetSplit split;
            private readonly string[] target_type;
            private readonly IModule<Tensor, Tensor>? transform;
            private readonly IModule<Tensor, Tensor>? target_transform;
            private string[]? filename;
            private Tensor? identity;
            private Tensor? bbox;
            private Tensor? landmarks_align;
            private Tensor? attr;
            private string[]? attr_names;

            internal CelebA(
                string root,
                CelebADatasetSplit split,
                string[] target_type,
                IModule<Tensor, Tensor>? transform,
                IModule<Tensor, Tensor>? target_transform)
            {
                this.root = root;
                this.split = split;
                this.target_type = target_type;
                this.transform = transform;
                this.target_transform = target_transform;
            }

            public override long Count => this.attr is null ? 0 : this.attr.shape[0];

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    identity?.Dispose();
                    bbox?.Dispose();
                    landmarks_align?.Dispose();
                    attr?.Dispose();
                }
            }

        internal void Load()
            {
                if (!this.CheckIntegrity()) {
                    throw new InvalidDataException("Dataset not found or corrupted. You can use download=True to download it");
                }

                var splits = this.LoadCsv("list_eval_partition.txt");
                var identity = this.LoadCsv("identity_CelebA.txt");
                var bbox = this.LoadCsv("list_bbox_celeba.txt", header: 1);
                var landmarks_align = this.LoadCsv("list_landmarks_align_celeba.txt", header: 1);
                var attr = this.LoadCsv("list_attr_celeba.txt", header: 1);

                if (this.split == CelebADatasetSplit.All) {
                    this.filename = splits.index;

                    this.identity = identity.data;
                    this.bbox = bbox.data;
                    this.landmarks_align = landmarks_align.data;
                    this.attr = attr.data;
                } else {
                    var mask = (splits.data == (long)this.split).squeeze();

                    var x = torch.squeeze(torch.nonzero(mask), dim: null);
                    var y = new List<string>();
                    for (int i = 0; i < x.shape[0]; i++) {
                        y.Add(splits.index[x[i].item<long>()]);
                    }
                    this.filename = y.ToArray();

                    this.identity = identity.data[mask];
                    this.bbox = bbox.data[mask];
                    this.landmarks_align = landmarks_align.data[mask];
                    this.attr = attr.data[mask];
                }
                // map from {-1, 1} to {0, 1}
                this.attr = torch.div(this.attr + 1, 2, rounding_mode: RoundingMode.floor).to(this.attr.dtype);
                this.attr_names = attr.header;
            }

            private CSV LoadCsv(string filename, int? header = null)
            {
                IList<string[]> data = (
                    File.ReadAllLines(Path.Combine(this.root, base_folder, filename))
                    .Select(line => line.TrimStart().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))).ToList();

                string[]? headers;
                if (header is not null) {
                    headers = data[(int)header];
                    data = data.Skip((int)header + 1).ToList();
                } else {
                    headers = null;
                }

                var indices = data.Select(row => row[0]).ToArray();

                data = data.Select(row => row.AsSpan(1).ToArray()).ToList();
                var data_long = new long[data.Count, data[0].Length];
                for (int i = 0; i < data_long.GetLength(0); i++) {
                    for (int j = 0; j < data_long.GetLength(1); j++) {
                        if (data[i].Length != data_long.GetLength(1)) {
                            throw new Exception();
                        }
                        long result = long.TryParse(data[i][j], out result) ? result : 0;
                        data_long[i, j] = result;
                    }
                }
                return new CSV(headers, indices, torch.tensor(data_long));
            }

            /// <summary>
            /// Get tensor according to index
            /// </summary>
            /// <param name="index">Index for tensor</param>
            /// <returns>Tensors of index.</returns>
            public override Dictionary<string, Tensor> GetTensor(long index)
            {
                if (this.filename is null) throw new InvalidOperationException();
                Tensor X = torchvision.io.read_image(Path.Combine(this.root, base_folder, "img_align_celeba", this.filename[index]));

                var result = new Dictionary<string, Tensor>();
                foreach (var t in this.target_type) {
                    if (t == "attr") {
                        if (this.attr is null) throw new InvalidDataException();
                        result[t] = this.attr[index, TensorIndex.Colon];
                    } else if (t == "identity") {
                        if (this.identity is null) throw new InvalidDataException();
                        result[t] = this.identity[index, 0];
                    } else if (t == "bbox") {
                        if (this.bbox is null) throw new InvalidDataException();
                        result[t] = this.bbox[index, TensorIndex.Colon];
                    } else if (t == "landmarks") {
                        if (this.landmarks_align is null) throw new InvalidDataException();
                        result[t] = this.landmarks_align[index, TensorIndex.Colon];
                    } else {
                        throw new InvalidDataException($"Target type \"{t}\" is not recognized.");
                    }
                }

                if (this.transform is not null) {
                    X = this.transform.call(X);
                }
                result["input"] = X;
                if (this.target_transform is not null) {
                    string t = this.target_type[0];
                    result[t] = this.target_transform.call(result[t]);
                }
                return result;
            }

            private bool CheckIntegrity()
            {
                foreach (var (_, md5, filename) in file_list) {
                    var fpath = Path.Combine(this.root, base_folder, filename);
                    var ext = Path.GetExtension(filename);
                    // Allow original archive to be deleted (zip and 7z)
                    // Only need the extracted images
                    if (ext != ".zip" && ext != ".7z" && !torchvision.io.GDriveDownload.CheckIntegrity(fpath, md5)) {
                        return false;
                    }
                }

                // Should check a hash of the images
                return Directory.Exists(Path.Combine(this.root, base_folder, "img_align_celeba"));
            }

            public void download()
            {
                if (this.CheckIntegrity()) {
                    Console.WriteLine("Files already downloaded and verified");
                    return;
                }

                foreach (var (file_id, md5, filename) in file_list) {
                    download_file_from_google_drive(file_id, Path.Combine(this.root, base_folder), filename, md5);
                }

                extract_archive(Path.Combine(this.root, base_folder, "img_align_celeba.zip"));
            }

            private struct CSV
            {
                public string[]? header;
                public string[] index;
                public torch.Tensor data;
                public CSV(string[]? header, string[] index, torch.Tensor data)
                {
                    this.header = header;
                    this.index = index;
                    this.data = data;
                }
            }
        }
    }

    public static partial class torchvision
    {
        public static partial class datasets
        {
            public enum CelebADatasetSplit
            {
                Train = 0,
                Valid = 1,
                Test = 2,
                All = -1,
            }

        /// <summary>
        /// `Large-scale CelebFaces Attributes (CelebA) Dataset http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html Dataset.
        /// </summary>
        /// <param name="root">Root directory where images are downloaded to.</param>
        /// <param name="split">One of Train, Valid, Test, All.
        /// Accordingly dataset is selected.</param>
        /// <param name="target_type">Type of target to use, ``attr``, ``identity``, ``bbox``,
        /// or ``landmarks``.</param>
        /// <param name="transform">A function/transform that  takes in an PIL image
        /// and returns a transformed version.</param>
        /// <param name="target_transform">A function/transform that takes in the
        /// target and transforms it.</param>
        /// <param name="download">If true, downloads the dataset from the internet and
        /// puts it in root directory. If dataset is already downloaded, it is not
        /// downloaded again.</param>
        public static Dataset CelebA(
                string root,
                CelebADatasetSplit split = CelebADatasetSplit.Train,
                string[]? target_type = null,
                IModule<Tensor, Tensor>? transform = null,
                IModule<Tensor, Tensor>? target_transform = null,
                bool download = false)
            {
                if (target_type == null) {
                    target_type = new string[] { "attr" };
                }
                var dataset = new Modules.CelebA(
                    root,
                    split,
                    target_type,
                    transform,
                    target_transform);
                if (download) {
                    dataset.download();
                }
                dataset.Load();
                return dataset;
            }
        }
    }
}
