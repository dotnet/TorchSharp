// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Google.Protobuf;
using Tensorboard;

namespace TorchSharp
{
    public static partial class torch
    {
        public static partial class utils
        {
            public static partial class tensorboard
            {
                /// <summary>
                /// Writes entries directly to event files in the log_dir to be consumed by TensorBoard.
                /// 
                /// The SummaryWriter class provides a high-level API to create an event file in a given directory and add summaries and events to it.The class updates the file contents asynchronously.
                ///
                /// This allows a training program to call methods to add data to the file directly from the training loop, without slowing down training.
                /// </summary>
                /// <param name="log_dir">
                /// Save directory location. Default is runs/CURRENT_DATETIME_HOSTNAME, which changes after each run.
                /// Use hierarchical folder structure to compare between runs easily. e.g. pass in ‘runs/exp1’, ‘runs/exp2’, etc.
                /// for each new experiment to compare across them
                /// </param>
                /// <param name="filename_suffix">Suffix added to all event filenames in the log_dir directory.</param>
                /// <param name="createRunName">Create a time-based run name, even if log_dir is specified.</param>
                public static Modules.SummaryWriter SummaryWriter(string log_dir = null, string filename_suffix = null, bool createRunName = false)
                {
                    if (createRunName && !string.IsNullOrEmpty(log_dir)) {
                        log_dir = CreateRunName(log_dir);
                    }

                    return new Modules.SummaryWriter(log_dir, filename_suffix);
                }

                internal static string CreateRunName(string log_dir)
                {
                    var now = DateTime.Now;
                    var _months = System.Globalization.CultureInfo.InvariantCulture.DateTimeFormat.AbbreviatedMonthNames;
                    var name = $"{_months[now.Month - 1]}{now.Day}_{now.Hour}-{now.Minute}-{now.Second}_{System.Net.Dns.GetHostName()}";
                    log_dir = Path.Combine(log_dir, name);
                    return log_dir;
                }
            }
        }
    }

    namespace Modules
    {
        /// <summary>
        /// Writes entries directly to event files in the log_dir to be consumed by TensorBoard.
        /// 
        /// The SummaryWriter class provides a high-level API to create an event file in a given directory and add summaries and events to it.The class updates the file contents asynchronously.
        ///
        /// This allows a training program to call methods to add data to the file directly from the training loop, without slowing down training.
        /// </summary>
        public class SummaryWriter
        {
            internal SummaryWriter(string log_dir, string filename_suffix)
            {
                _suffix = filename_suffix;

                if (string.IsNullOrEmpty(log_dir)) {
                    _log_dir = torch.utils.tensorboard.CreateRunName("runs");
                } else {
                    _log_dir = log_dir;
                }

                if (!Directory.Exists(_log_dir)) {
                    Directory.CreateDirectory(_log_dir);
                }

                var fileName = Path.Combine(_log_dir, $"events.out.tfevents.{DateTime.Now.Ticks}.{System.Net.Dns.GetHostName()}.{Process.GetCurrentProcess().Id}.{Interlocked.Increment(ref _global_uid)}");
                if (!string.IsNullOrEmpty(_suffix)) {
                    fileName = fileName + "." + _suffix;
                }

                if (File.Exists(fileName)) {
                    File.Delete(fileName);
                }

                InitFile(fileName);

                _fileNames["__default__"] = fileName;
            }

            /// <summary>
            /// The directory/folder where logging is made.
            /// </summary>
            public string LogDir { get { return _log_dir; } }

            /// <summary>
            /// Add scalar data to summary.
            /// </summary>
            /// <param name="tag">Data identifier</param>
            /// <param name="scalar_value">Value to save</param>
            /// <param name="global_step">Global step value to record</param>
            /// <param name="walltime">Optional override default walltime (DateTimeOffset.Now.ToUnixTimeSeconds())</param>
            public void add_scalar(string tag, float scalar_value, int global_step, long? walltime = null)
            {
                var fileName = InitDefaultFile();
                SetWalltime(ref walltime);

                var summary = new Summary();
                summary.Value.Add(new Summary.Types.Value() { SimpleValue = scalar_value, Tag = tag });
                var evnt = new Event() { Step = global_step, WallTime = walltime.Value, Summary = summary };

                WriteEvent(fileName, evnt);
            }

            /// <summary>
            /// Adds many scalar data points to summary.
            /// </summary>
            /// <param name="main_tag">Data identifier</param>
            /// <param name="tag_scalar_dict">Dictionary storing the tag and corresponding values</param>
            /// <param name="global_step">Global step value to record</param>
            /// <param name="walltime">Optional override default walltime (DateTimeOffset.Now.ToUnixTimeSeconds())</param>
            public void add_scalars(string main_tag, IDictionary<string, float> tag_scalar_dict, int global_step, long? walltime = null)
            {
                SetWalltime(ref walltime);

                foreach (var kv in tag_scalar_dict) {

                    var key = kv.Key;
                    var scalar_value = kv.Value;

                    var summary = new Summary();
                    summary.Value.Add(new Summary.Types.Value() { SimpleValue = scalar_value, Tag = main_tag });
                    var evnt = new Event() { Step = global_step, WallTime = walltime.Value, Summary = summary };

                    if (!_fileNames.TryGetValue(key, out var fileName)) {

                        // We haven't logged this tag from this session, yet. Create a clean file.

                        var fileDir = Path.Combine(_log_dir, $"{main_tag}_{key}");
                        if (!Directory.Exists(fileDir)) {
                            Directory.CreateDirectory(fileDir);
                        }

                        fileName = Path.Combine(fileDir, $"events.out.tfevents.{DateTime.Now.Ticks}.{System.Net.Dns.GetHostName()}.{Process.GetCurrentProcess().Id}.{Interlocked.Increment(ref _global_uid)}");

                        if (File.Exists(fileName)) {
                            File.Delete(fileName);
                        }

                        InitFile(fileName);

                        _fileNames[key] = fileName;
                    }

                    WriteEvent(fileName, evnt);
                }
            }


            /// <summary>
            /// Adds many scalar data points to summary.
            /// </summary>
            /// <param name="main_tag">Data identifier</param>
            /// <param name="tag_scalar_dict">List of tuples storing the tag and corresponding values</param>
            /// <param name="global_step">Global step value to record</param>
            /// <param name="walltime">Optional override default walltime (DateTimeOffset.Now.ToUnixTimeSeconds())</param>
            public void add_scalars(string main_tag, IList<(string, float)> tag_scalar_dict, int global_step, long? walltime = null)
            {
                SetWalltime(ref walltime);

                foreach (var (key, scalar_value) in tag_scalar_dict) {

                    var summary = new Summary();
                    summary.Value.Add(new Summary.Types.Value() { SimpleValue = scalar_value, Tag = main_tag });
                    var evnt = new Event() { Step = global_step, WallTime = walltime.Value, Summary = summary };

                    if (!_fileNames.TryGetValue(key, out var fileName)) {

                        var fileDir = Path.Combine(_log_dir, $"{main_tag}_{key}");
                        if (!Directory.Exists(fileDir)) {
                            Directory.CreateDirectory(fileDir);
                        }

                        fileName = Path.Combine(fileDir, $"events.out.tfevents.{DateTime.Now.Ticks}.{System.Net.Dns.GetHostName()}.{Process.GetCurrentProcess().Id}.{Interlocked.Increment(ref _global_uid)}");

                        if (!File.Exists(fileName)) {
                            InitFile(fileName);
                        }

                        _fileNames[key] = fileName;
                    }

                    WriteEvent(fileName, evnt);
                }
            }

            /// <summary>
            /// Add histogram to summary.
            ///
            /// https://pytorch.org/docs/stable/_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_histogram
            /// </summary>
            /// <param name="tag"> Data identifier </param>
            /// <param name="values"> Values to build histogram </param>
            /// <param name="global_step"> Global step value to record </param>
            /// <param name="bins"> This determines how the bins are made </param>
            /// <param name="walltime"> Optional override default walltime (DateTimeOffset.Now.ToUnixTimeSeconds()) </param>
            /// <param name="max_bins"></param>
            public void add_histogram(string tag,
                torch.Tensor values,
                int global_step,
                Utils.tensorboard.Enums.HistogramBinSelector bins = Utils.tensorboard.Enums.HistogramBinSelector.Tensorflow,
                long? walltime = null,
                long? max_bins = null)
            {
                static torch.Tensor default_bins()
                {
                    double v = 1e-12;
                    var buckets = new List<double>();
                    var neg_buckets = new List<double>();
                    while (v < 1e20) {
                        buckets.Add(v);
                        neg_buckets.Add(-v);
                        v *= 1.1;
                    }
                    neg_buckets.Reverse();
                    var result = new List<double>();
                    result.AddRange(neg_buckets); result.Add(0); result.AddRange(buckets);
                    return torch.tensor(result);
                }

                var fileName = InitDefaultFile();
                SetWalltime(ref walltime);
                Summary summary = bins == Utils.tensorboard.Enums.HistogramBinSelector.Tensorflow ?
                    torch.utils.tensorboard.Summary.histogram(tag, values, default_bins(), max_bins) :
                    torch.utils.tensorboard.Summary.histogram(tag, values, (HistogramBinSelector)(byte)bins, max_bins);
                var evnt = new Event() { Step = global_step, WallTime = walltime.Value, Summary = summary };
                WriteEvent(fileName, evnt);
            }

            /// <summary>
            /// Add batched image data to summary.
            ///
            /// https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image
            /// </summary>
            /// <param name="tag"> Data identifier </param>
            /// <param name="img_tensor"> Image data </param>
            /// <param name="global_step"> Global step value to record </param>
            /// <param name="walltime"> Optional override default walltime (DateTimeOffset.Now.ToUnixTimeSeconds()) </param>
            /// <param name="dataformats"> Image data format specification of the form CHW, HWC, HW, WH, etc. </param>
            public void add_img(string tag, torch.Tensor img_tensor, int global_step, long? walltime = null, string dataformats = "CHW")
            {
                var fileName = InitDefaultFile();
                SetWalltime(ref walltime);
                Summary summary = torch.utils.tensorboard.Summary.image(tag, img_tensor, dataformats: dataformats);
                var evnt = new Event() { Step = global_step, WallTime = walltime.Value, Summary = summary };
                WriteEvent(fileName, evnt);
            }

            /// <summary>
            /// Add batched image data to summary.
            ///
            /// https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image
            /// </summary>
            /// <param name="tag"> Data identifier </param>
            /// <param name="file_name"> Image file </param>
            /// <param name="global_step"> Global step value to record </param>
            /// <param name="walltime"> Optional override default walltime (DateTimeOffset.Now.ToUnixTimeSeconds()) </param>
            public void add_img(string tag, string file_name, int global_step, long? walltime = null)
            {
                var fileName = InitDefaultFile();
                SetWalltime(ref walltime);
                Summary summary = torch.utils.tensorboard.Summary.image(tag, file_name);
                var evnt = new Event() { Step = global_step, WallTime = walltime.Value, Summary = summary };
                WriteEvent(fileName, evnt);
            }

            /// <summary>
            /// Add video data to summary.
            ///
            /// https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video
            /// </summary>
            /// <param name="tag"> Data identifier </param>
            /// <param name="vid_tensor">
            /// Video data
            ///
            /// Shape: (N,T,C,H,W). The values should lie in [0, 255] for type uint8 or [0, 1] for type float.
            /// </param>
            /// <param name="global_step"> Global step value to record </param>
            /// <param name="fps"> Frames per second </param>
            /// <param name="walltime"> Optional override default walltime (DateTimeOffset.Now.ToUnixTimeSeconds()) </param>
            public void add_video(string tag, torch.Tensor vid_tensor, int global_step, int fps = 4, long? walltime = null)
            {
                var fileName = InitDefaultFile();
                SetWalltime(ref walltime);
                Summary summary = torch.utils.tensorboard.Summary.video(tag, vid_tensor, fps);
                var evnt = new Event() { Step = global_step, WallTime = walltime.Value, Summary = summary };
                WriteEvent(fileName, evnt);
            }

            /// <summary>
            /// Add text data to summary.
            /// 
            /// https://pytorch.org/docs/stable/_modules/torch/utils/tensorboard/writer.html#SummaryWriter.add_text
            /// </summary>
            /// <param name="tag"> Data identifier </param>
            /// <param name="text_string"> String to save </param>
            /// <param name="global_step"> Global step value to record </param>
            /// <param name="walltime"> Optional override default walltime (DateTimeOffset.Now.ToUnixTimeSeconds()) </param>
            public void add_text(string tag, string text_string, int global_step, long? walltime = null)
            {
                var fileName = InitDefaultFile();
                SetWalltime(ref walltime);

                var evnt = new Event() { Step = global_step, WallTime = walltime.Value, Summary = torch.utils.tensorboard.Summary.text(tag, text_string) };
                WriteEvent(fileName, evnt);
            }

            private static void InitFile(string fileName)
            {
                var evnt = new Event() { FileVersion = "brain.Event:2", WallTime = DateTime.Now.Ticks };
                WriteEvent(fileName, evnt);
            }

            private string InitDefaultFile()
            {
                var fileName = _fileNames["__default__"];

                if (!File.Exists(fileName)) {
                    InitFile(fileName);
                }

                return fileName;
            }

            private static void SetWalltime(ref long? walltime)
                => walltime ??= DateTimeOffset.Now.ToUnixTimeSeconds();

            private static void WriteEvent(string fileName, Event evnt)
            {
                var bytes = evnt.ToByteArray();

                long header = bytes.LongLength;
                uint header_crc = GetMaskedCrc(header);
                uint footer_crc = GetMaskedCrc(bytes);

                using (var fStream = File.Open(fileName, FileMode.OpenOrCreate, FileAccess.Write, FileShare.Read)) {
                    fStream.Seek(0, SeekOrigin.End);

                    using (var writers = new BinaryWriter(fStream)) {
                        writers.Write(header);
                        writers.Write(header_crc);
                        using (var stream = new Google.Protobuf.CodedOutputStream(fStream, true)) {
                            evnt.WriteTo(stream);
                        }
                        fStream.Seek(0, SeekOrigin.End);
                        writers.Write(footer_crc);
                        fStream.Flush();
                    }
                }
            }

            private string _log_dir = null;
            private Dictionary<string, string> _fileNames = new Dictionary<string, string>();

            private string _suffix = null;

            private int _global_uid;

            private static uint GetMaskedCrc(byte[] data)
            {
                var value = TorchSharp.Utils.CRC32C.process(data);
                // Rotate right by 15 bits and add a constant.
                return ((value >> 15) | (value << 17)) + 0xa282ead8U;
            }

            private static uint GetMaskedCrc(int data)
            {
                var value = TorchSharp.Utils.CRC32C.process(data);
                // Rotate right by 15 bits and add a constant.
                return ((value >> 15) | (value << 17)) + 0xa282ead8U;
            }

            private static uint GetMaskedCrc(long data)
            {
                var value = TorchSharp.Utils.CRC32C.process(data);
                // Rotate right by 15 bits and add a constant.
                return ((value >> 15) | (value << 17)) + 0xa282ead8U;
            }
        }
    }
}