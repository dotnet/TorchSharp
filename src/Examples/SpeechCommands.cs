// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchaudio.datasets;

namespace TorchSharp.Examples
{
    /// <summary>
    /// SpeechCommands model with convolusions.
    /// </summary>
    /// <remarks>
    /// Translated from Python implementation
    /// https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
    /// </remarks>
    public class SpeechCommands
    {
        private static readonly string[] Labels = new string[] {
            "bed", "bird", "backward", "cat", "dog", "down", "eight",
            "five", "follow", "forward", "four", "go", "happy", "house",
            "learn", "left", "marvin", "nine", "no", "off", "on", "one",
            "right", "seven", "six", "sheila", "stop", "three", "tree",
            "two", "up", "visual", "wow", "yes", "zero"
        };

        private static int _epochs = 1;
        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 128;
        private static int _sample_rate = 16000;
        private static int _new_sample_rate = 8000;

        private readonly static int _logInterval = 200;

        private static IDictionary<string, int> _labelToIndex;

        internal static void Main(string[] args)
        {
            var dataset = args.Length > 0 ? args[0] : "speechcommands";
            var datasetPath = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);

            torchaudio.backend.utils.set_audio_backend(new WaveAudioBackend());
            torch.random.manual_seed(1);

            var cwd = Environment.CurrentDirectory;
            Console.WriteLine(datasetPath);

            var device = torch.device(torch.cuda.is_available() ? "cuda" : "cpu");

            Console.WriteLine($"Running SpeechCommands on {device.type.ToString()}");
            Console.WriteLine($"Dataset: {dataset}");

            if (device.type == DeviceType.CUDA) {
                _trainBatchSize *= 4;
                _testBatchSize *= 4;
            }

            var model = new M5("model");
            model.to(device);

            var transform = torchaudio.transforms.Resample(_sample_rate, _new_sample_rate, device: device);

            _labelToIndex = Labels.Select((label, index) => (label, index)).ToDictionary(t => t.label, t => t.index);
            using (var train_data = SPEECHCOMMANDS(datasetPath, subset: "training", download: true))
            using (var test_data = SPEECHCOMMANDS(datasetPath, subset: "testing", download: true)) {
                TrainingLoop("speechcommands", device, model, transform, train_data, test_data);
            }
        }

        private static BatchItem Collate(IEnumerable<SpeechCommandsDatasetItem> items, torch.Device device)
        {
            var audio_sequences = items.Select(item => item.waveform.t());
            var padded_audio = torch.nn.utils.rnn.pad_sequence(audio_sequences, batch_first: true, padding_value: 0.0);
            padded_audio = padded_audio.permute(0, 2, 1);
            var labels = items.Select(item => _labelToIndex[item.label]).ToArray();
            return new BatchItem {
                audio = padded_audio.to(device),
                label = torch.tensor(labels, dtype: torch.int64, device: device)
            };
        }

        internal static void TrainingLoop(string dataset, Device device, M5 model, nn.IModule<Tensor, Tensor> transform, Dataset<SpeechCommandsDatasetItem> train_data, Dataset<SpeechCommandsDatasetItem> test_data)
        {
            using (var train_loader = new DataLoader<SpeechCommandsDatasetItem, BatchItem>(
                train_data, _trainBatchSize, Collate, shuffle: true, device: device))
            using (var test_loader = new DataLoader<SpeechCommandsDatasetItem, BatchItem>(
                test_data, _testBatchSize, Collate, shuffle: false, device: device)) {
                if (device.type == DeviceType.CUDA) {
                    _epochs *= 4;
                }

                var optimizer = optim.Adam(model.parameters(), lr: 0.01, weight_decay: 0.0001);
                var scheduler = optim.lr_scheduler.StepLR(optimizer, step_size: 20, gamma: 0.1);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (var epoch = 1; epoch <= _epochs; epoch++) {
                    Train(model, transform, optimizer, torch.nn.NLLLoss(reduction: torch.nn.Reduction.Mean), train_loader, epoch, train_data.Count);
                    Test(model, transform, torch.nn.NLLLoss(reduction: torch.nn.Reduction.Sum), test_loader, test_data.Count);

                    Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");
                    scheduler.step();
                }

                sw.Stop();
                Console.WriteLine($"Elapsed time: {sw.Elapsed.TotalSeconds:F1} s.");

                Console.WriteLine("Saving model to '{0}'", dataset + ".model.bin");
                model.save(dataset + ".model.bin");
            }
        }

        private static void Train(
            M5 model,
            nn.IModule<Tensor, Tensor> transform,
            torch.optim.Optimizer optimizer,
            Loss<Tensor,Tensor,Tensor> criteria,
            DataLoader<SpeechCommandsDatasetItem, BatchItem> dataLoader,
            int epoch,
            long size)
        {
            int batchId = 1;
            long total = 0;

            Console.WriteLine($"Epoch: {epoch}...");

            using (var d = torch.NewDisposeScope()) {

                model.train();
                foreach (var batch in dataLoader) {
                    var audio = transform.call(batch.audio);
                    var target = batch.label;
                    var output = model.call(batch.audio).squeeze();
                    var loss = criteria.call(output, target);
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();
                    total += target.shape[0];

                    if (batchId % _logInterval == 0 || total == size) {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{total} / {size}] Loss: {loss.ToSingle():F4}");
                    }

                    batchId++;

                    d.DisposeEverything();
                }
            }
        }

        private static void Test(
            M5 model,
            nn.IModule<Tensor, Tensor> transform,
            Loss<Tensor, Tensor, Tensor> criteria,
            DataLoader<SpeechCommandsDatasetItem, BatchItem> dataLoader,
            long size)
        {
            model.eval();

            double testLoss = 0;
            int correct = 0;

            using (var d = torch.NewDisposeScope()) {

                foreach (var batch in dataLoader) {
                    var audio = transform.call(batch.audio);
                    var target = batch.label;
                    var output = model.call(batch.audio).squeeze();
                    var loss = criteria.call(output, target);
                    testLoss += loss.ToSingle();

                    var pred = output.argmax(1);
                    correct += pred.eq(batch.label).sum().ToInt32();

                    d.DisposeEverything();
                }
            }

            Console.WriteLine($"Size: {size}, Total: {size}");

            Console.WriteLine($"\rTest set: Average loss {(testLoss / size):F4} | Accuracy {((double)correct / size):P2}");
        }

        private class WaveAudioBackend : torchaudio.backend.AudioBackend
        {
            public override (torch.Tensor, int) load(string filepath, long frame_offset = 0, long num_frames = -1, bool normalize = true, bool channels_first = true, torchaudio.AudioFormat? format = null)
            {
                byte[] data = File.ReadAllBytes(filepath);
                // In many cases, the first 44 bytes are for RIFF header.
                short[] waveform = MemoryMarshal.Cast<byte, short>(data.AsSpan(11 * 4)).ToArray();
                return (torch.tensor(waveform).unsqueeze(0).to(torch.float32) / short.MaxValue, 16000);
            }

            public override void save(string filepath, torch.Tensor src, int sample_rate, bool channels_first = true, float? compression = null, torchaudio.AudioFormat? format = null, torchaudio.AudioEncoding? encoding = null, int? bits_per_sample = null)
            {
                throw new NotImplementedException();
            }

            public override torchaudio.AudioMetaData info(string filepath, torchaudio.AudioFormat? format = null)
            {
                throw new NotImplementedException();
            }
        }

        private class BatchItem
        {
            public torch.Tensor audio;
            public torch.Tensor label;
        }

        internal class M5 : Module<Tensor, Tensor>
        {
            private readonly Module<Tensor, Tensor> conv1;
            private readonly Module<Tensor, Tensor> bn1;
            private readonly Module<Tensor, Tensor> pool1;
            private readonly Module<Tensor, Tensor> conv2;
            private readonly Module<Tensor, Tensor> bn2;
            private readonly Module<Tensor, Tensor> pool2;
            private readonly Module<Tensor, Tensor> conv3;
            private readonly Module<Tensor, Tensor> bn3;
            private readonly Module<Tensor, Tensor> pool3;
            private readonly Module<Tensor, Tensor> conv4;
            private readonly Module<Tensor, Tensor> bn4;
            private readonly Module<Tensor, Tensor> pool4;
            private readonly Module<Tensor, Tensor> fc1;

            public M5(string name, int n_input = 1, int n_output = 35, int stride = 16, int n_channel = 32) : base(name)
            {
                conv1 = nn.Conv1d(n_input, n_channel, kernelSize: 80, stride: stride);
                bn1 = nn.BatchNorm1d(n_channel);
                pool1 = nn.MaxPool1d(4);
                conv2 = nn.Conv1d(n_channel, n_channel, kernelSize: 3);
                bn2 = nn.BatchNorm1d(n_channel);
                pool2 = nn.MaxPool1d(4);
                conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernelSize: 3);
                bn3 = nn.BatchNorm1d(2 * n_channel);
                pool3 = nn.MaxPool1d(4);
                conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernelSize: 3);
                bn4 = nn.BatchNorm1d(2 * n_channel);
                pool4 = nn.MaxPool1d(4);
                fc1 = nn.Linear(2 * n_channel, n_output);
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                var x = input;
                x = conv1.call(x);
                x = relu(bn1.call(x));
                x = pool1.call(x);
                x = conv2.call(x);
                x = relu(bn2.call(x));
                x = pool2.call(x);
                x = conv3.call(x);
                x = relu(bn3.call(x));
                x = pool3.call(x);
                x = conv4.call(x);
                x = relu(bn4.call(x));
                x = pool4.call(x);
                x = avg_pool1d(x, x.shape[x.dim() - 1]);
                x = x.permute(0, 2, 1);
                x = fc1.call(x);
                return log_softmax(x, dim: 2);
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    pool1.Dispose();
                    pool2.Dispose();
                    pool3.Dispose();
                    pool4.Dispose();
                    conv1.Dispose();
                    conv2.Dispose();
                    conv3.Dispose();
                    conv4.Dispose();
                    bn1.Dispose();
                    bn2.Dispose();
                    bn3.Dispose();
                    bn4.Dispose();
                    fc1.Dispose();
                    ClearModules();
                }
                base.Dispose(disposing);
            }
        }
    }
}
