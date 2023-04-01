// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision.datasets.utils;

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
            private readonly string split;
            private readonly string target_type;
            private readonly IModule<Tensor, Tensor>? transform;
            private readonly IModule<Tensor, Tensor>? target_transform;

            internal CelebA(
                string root,
                string split,
                string target_type,
                IModule<Tensor, Tensor>? transform,
                IModule<Tensor, Tensor>? target_transform)
            {
                this.root = root;
                this.split = split;
                this.target_type = target_type;
                this.transform = transform;
                this.target_transform = target_transform;
            }

            public override long Count => 100;

            /// <summary>
            /// Get tensor according to index
            /// </summary>
            /// <param name="index">Index for tensor</param>
            /// <returns>Tensors of index. DataLoader will catenate these tensors as batchSize * 784 for image, batchSize * 1 for label</returns>
            public override Dictionary<string, Tensor> GetTensor(long index)
            {
                Tensor input = torch.zeros(10, 10);
                Tensor target = torch.zeros(10, 10);
                if (this.transform is not null) {
                    input = this.transform.call(input);
                }
                if (this.target_transform is not null) {
                    target = this.target_transform.call(target);
                }
                return new Dictionary<string, Tensor> {
                    { "input", input },
                    { "target", target }
                };
            }

            public void Download()
            {
                if (this._check_integrity()) {
                    Console.WriteLine("Files already downloaded and verified");
                    return;
                }

                foreach (var (file_id, md5, filename) in file_list) {
                    download_file_from_google_drive(file_id, Path.Combine(this.root, base_folder), filename, md5);
                }

                extract_archive(Path.Combine(this.root, base_folder, "img_align_celeba.zip"));
            }
                
            private bool _check_integrity()
            {
                foreach (var (_, md5, filename) in file_list) {
                    var fpath = Path.Combine(this.root, base_folder, filename);
                    var ext = Path.GetExtension(filename);
                    // Allow original archive to be deleted (zip and 7z)
                    // Only need the extracted images
                    if (ext != "zip" && ext != ".7z" && !check_integrity(fpath, md5)) {
                        return false;
                    }
                }

                // Should check a hash of the images
                return Directory.Exists(Path.Combine(this.root, base_folder, "img_align_celeba"));
            }
        }
    }

    public static partial class torchvision
    {
        public static partial class datasets
        {
            public static Dataset CelebA(
                string root,
                IModule<Tensor, Tensor>? transform = null,
                IModule<Tensor, Tensor>? target_transform = null,
                string split = "train",
                string target_type = "attr",
                bool download = false)
            {
                var dataset = new Modules.CelebA(
                    root,
                    split,
                    target_type,
                    transform,
                    target_transform);
                if (download) {
                    dataset.Download();
                }
                return dataset;
            }
        }
    }
}
