using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class datasets
        {
            private class YESNODataset : torch.utils.data.Dataset
            {
                public const string ArchiveChecksum = "c3f49e0cca421f96b75b41640749167b52118f232498667ca7a5f9416aef8e73";

                private readonly string[] audioPathList;

                public YESNODataset(string directoryPathInArchive)
                {
                    audioPathList = Directory.EnumerateFiles(directoryPathInArchive, "*.wav").ToArray();
                }

                public override long Count => audioPathList.LongLength;

                public override Dictionary<string, torch.Tensor> GetTensor(long index)
                {
                    var (waveform, sample_rate) = torchaudio.load(audioPathList[index]);
                    return new Dictionary<string, torch.Tensor>() {
                        ["waveform"] = waveform,
                        ["sample_rate"] = null,
                        ["labels"] = null
                    };
                }
            }

            public static torch.utils.data.Dataset YESNO(
                string root,
                string url = "http://www.openslr.org/resources/1/waves_yesno.tar.gz",
                string folder_in_archive = "waves_yesno",
                bool download = false)
            {
                var directoryPathInArchive = Path.Combine(root, folder_in_archive);
                var archiveFileName = GetFilenameFromUrl(url);
                var archivePath = Path.Combine(root, archiveFileName);
                if (download) {
                    if (!Directory.Exists(directoryPathInArchive)) {
                        if (!File.Exists(archivePath)) {
                            torch.hub.download_url_to_file(url, archivePath, YESNODataset.ArchiveChecksum);
                        }
                        ExtractTarGz(archivePath, root);
                    }
                }
                if (!Directory.Exists(directoryPathInArchive)) {
                    throw new InvalidDataException("Dataset not found. Please use `download=true` to download it.");
                }
                return new YESNODataset(directoryPathInArchive);
            }

            private static string GetFilenameFromUrl(string url)
            {
                int index = url.LastIndexOf('/');
                if (index < 0) throw new ArgumentException();
                var fileName = url.Substring(index + 1);
                if (string.IsNullOrWhiteSpace(fileName)) throw new ArgumentException();
                return fileName;
            }

            private static void ExtractTarGz(String archivePath, String destinationDirectory)
            {
                using (var fileStream = File.OpenRead(archivePath)) {
                    using (var inputStream = new GZipInputStream(fileStream)) {
                        using (TarArchive tarArchive = TarArchive.CreateInputTarArchive(inputStream, Encoding.UTF8)) {
                            tarArchive.ExtractContents(destinationDirectory);
                        }
                    }
                }
            }
        }
    }
}