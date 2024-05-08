// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class datasets
        {
            private class SpeechCommandsDataset : torch.utils.data.Dataset<SpeechCommandsDatasetItem>
            {
                internal const string URL = "speech_commands_v0.02";
                internal const string FOLDER_IN_ARCHIVE = "SpeechCommands";
                private const string HASH_DIVIDER = "_nohash_";
                private const string EXCEPT_FOLDER = "_background_noise_";
                internal static readonly IDictionary<string, string> _CHECKSUMS = new Dictionary<string, string> {
                    ["https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"] = "743935421bb51cccdb6bdd152e04c5c70274e935c82119ad7faeec31780d811d",
                    ["https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"] = "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
                };

                private string _path;
                private string[] _walker;

                internal SpeechCommandsDataset(string path, string subset)
                {
                    _path = path;
                    if (subset == "validation") {
                        _walker = LoadList(_path, "validation_list.txt");
                    } else if (subset == "testing") {
                        _walker = LoadList(_path, "testing_list.txt");
                    } else if (subset == "training") {
                        var excludes = new HashSet<string>(LoadList(_path, "validation_list.txt", "testing_list.txt"));
                        _walker = Directory.EnumerateFiles(_path, "*.wav", SearchOption.AllDirectories)
                            .Where(audioPath => audioPath.Contains(HASH_DIVIDER) && !audioPath.Contains(EXCEPT_FOLDER))
                            .Where(audioPath => !excludes.Contains(audioPath))
                            .OrderBy(audioPath => audioPath)
                            .ToArray();
                    } else {
                        _walker = Directory.EnumerateFiles(_path, "*.wav", SearchOption.AllDirectories)
                            .Where(audioPath => audioPath.Contains(HASH_DIVIDER) && !audioPath.Contains(EXCEPT_FOLDER))
                            .OrderBy(audioPath => audioPath)
                            .ToArray();
                    }
                }

                public override long Count => _walker.LongLength;

                public override SpeechCommandsDatasetItem GetTensor(long index)
                {
                    var audioPath = _walker[index];
                    return LoadSpeechCommandsItem(audioPath);
                }

                private string[] LoadList(string root, params string[] filenames)
                {
                    List<string> output = new();
                    foreach (var filename in filenames) {
                        var filepath = Path.Combine(root, filename);
                        var pathList = File.ReadAllLines(filepath)
                            .Select(line => Path.Combine(root, Path.Combine(line.Trim())));
                        output.AddRange(pathList);
                    }
                    return output.ToArray();
                }

                private SpeechCommandsDatasetItem LoadSpeechCommandsItem(string filepath)
                {
                    var filename = Path.GetFileName(filepath);
                    var label = Path.GetFileName(Path.GetDirectoryName(filepath));

                    // Some filenames have the form of "xxx.wav.wav"
                    var speaker = Path.GetFileNameWithoutExtension(filename);
                    speaker = Path.GetFileNameWithoutExtension(filename);

                    var parts = speaker.Split(new string[] { HASH_DIVIDER }, StringSplitOptions.None);
                    var speaker_id = parts[0];
                    int utterance_number = int.Parse(parts[1]);

                    // Load audio
                    var (waveform, sample_rate) = torchaudio.load(filepath);
                    return new SpeechCommandsDatasetItem {
                        waveform = waveform,
                        sample_rate = sample_rate,
                        label = label,
                        speaker_id = speaker_id,
                        utterance_number = utterance_number
                    };
                }
            }

            /// <summary>
            /// Create a Speech Commands dataset
            /// </summary>
            /// <param name="root">The path to the dataset</param>
            /// <param name="url">The URL to download the dataset from</param>
            /// <param name="folder_in_archive">The top directory of the dataset</param>
            /// <param name="download">True to download the dataset</param>
            /// <param name="subset">Select a subset of the dataset, null, "training", "validation" or "testing"</param>
            /// <returns>The dataset</returns>
            /// <exception cref="InvalidDataException"></exception>
            public static torch.utils.data.Dataset<SpeechCommandsDatasetItem> SPEECHCOMMANDS(
                string root,
                string url = SpeechCommandsDataset.URL,
                string folder_in_archive = SpeechCommandsDataset.FOLDER_IN_ARCHIVE,
                bool download = false,
                string subset = null)
            {
                if (url == "speech_commands_v0.01" || url == "speech_commands_v0.02") {
                    string base_url = "https://storage.googleapis.com/download.tensorflow.org/data/";
                    string ext_archive = ".tar.gz";
                    url = base_url + url + ext_archive;
                }

                string[] parts = url.Split('/');
                string basename = parts[parts.Length - 1];
                string archive = Path.Combine(root, basename);
                int index = basename.LastIndexOf('.');
                index = basename.LastIndexOf('.', index - 1);
                basename = basename.Substring(0, index);
                folder_in_archive = Path.Combine(folder_in_archive, basename);

                string path = Path.Combine(root, folder_in_archive);

                if (download) {
                    if (!Directory.Exists(path)) {
                        if (!File.Exists(archive)) {
                            string checksum = SpeechCommandsDataset._CHECKSUMS.TryGetValue(url, out checksum) ? checksum : null;
                            torch.hub.download_url_to_file(url, archive, hash_prefix: checksum);
                        }
                        utils.extract_archive(archive, path);
                    }
                } else {
                    if (!Directory.Exists(path)) {
                        throw new InvalidDataException("Dataset not found. Please use `download=true` to download it.");
                    }
                }

                return new SpeechCommandsDataset(path, subset);
            }
        }
    }
}