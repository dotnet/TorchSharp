// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class datasets
        {
            private class YesnoDataset : torch.utils.data.Dataset<YesnoDatasetItem>
            {
                public const string ArchiveChecksum = "c3f49e0cca421f96b75b41640749167b52118f232498667ca7a5f9416aef8e73";

                private readonly string[] audioPathList;

                public YesnoDataset(string directoryPathInArchive)
                {
                    audioPathList = Directory.EnumerateFiles(directoryPathInArchive, "*.wav").ToArray();
                }

                public override long Count => audioPathList.LongLength;

                public override YesnoDatasetItem GetTensor(long index)
                {
                    var (waveform, sample_rate) = torchaudio.load(audioPathList[index]);
                    return new() {
                        waveform = waveform,
                        sample_rate = sample_rate,
                        labels = null
                    };
                }
            }

            /// <summary>
            /// Create a Yesno dataset
            /// </summary>
            /// <param name="root">The path to the dataset</param>
            /// <param name="url">The URL to download the dataset from</param>
            /// <param name="folder_in_archive">The top directory of the dataset</param>
            /// <param name="download">True to download the dataset</param>
            /// <returns>The dataset</returns>
            /// <exception cref="InvalidDataException"></exception>
            public static torch.utils.data.Dataset<YesnoDatasetItem> YESNO(
                string root,
                string url = "http://www.openslr.org/resources/1/waves_yesno.tar.gz",
                string folder_in_archive = "waves_yesno",
                bool download = false)
            {
                var directoryPathInArchive = Path.Combine(root, folder_in_archive);
                var archiveFileName = GetFileNameFromUrl(url);
                var archivePath = Path.Combine(root, archiveFileName);
                if (download) {
                    if (!Directory.Exists(directoryPathInArchive)) {
                        if (!File.Exists(archivePath)) {
                            torch.hub.download_url_to_file(url, archivePath, YesnoDataset.ArchiveChecksum);
                        }
                        utils.extract_archive(archivePath, root);
                    }
                }
                if (!Directory.Exists(directoryPathInArchive)) {
                    throw new InvalidDataException("Dataset not found. Please use `download=true` to download it.");
                }
                return new YesnoDataset(directoryPathInArchive);
            }

            private static string GetFileNameFromUrl(string url)
            {
                int index = url.LastIndexOf('/');
                if (index < 0) throw new ArgumentException();
                var fileName = url.Substring(index + 1);
                if (string.IsNullOrWhiteSpace(fileName)) throw new ArgumentException();
                return fileName;
            }
        }
    }
}