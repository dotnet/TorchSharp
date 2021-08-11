// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Modified version of original AlexNet to fix CIFAR10 32x32 images.
    /// </summary>
    /// <remarks>
    /// The dataset for this example can be found at: http://yann.lecun.com/exdb/mnist/
    /// Download the binary file, and place it in a dedicated folder, e.g. 'CIFAR10,' then edit
    /// the '_dataLocation' definition below to point at the right folder.
    ///
    /// Note: so far, CIFAR10 is supported, but not CIFAR100.
    /// </remarks>
    class AlexNet
    {
        private readonly static string _dataset = "CIFAR10";
        private readonly static string _dataLocation = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", _dataset);

        private static int _epochs = 1;
        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 128;

        private readonly static int _logInterval = 25;
        private readonly static int _numClasses = 10;

        internal static void Main(string[] args)
        {
            torch.random.manual_seed(1);

            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;

            if (device.type == DeviceType.CUDA) {
                _trainBatchSize *= 8;
                _testBatchSize *= 8;
                _epochs *= 16;
            }

            Console.WriteLine();
            Console.WriteLine($"\tRunning AlexNet with {_dataset} on {device.type.ToString()} for {_epochs} epochs");
            Console.WriteLine();

            var sourceDir = _dataLocation;
            var targetDir = Path.Combine(_dataLocation, "test_data");

            if (!Directory.Exists(targetDir)) {
                Directory.CreateDirectory(targetDir);
                Utils.Decompress.ExtractTGZ(Path.Combine(sourceDir, "cifar-10-binary.tar.gz"), targetDir);
            }

            using (var train = new CIFARReader(targetDir, false, _trainBatchSize, shuffle: true, device: device))
            using (var test = new CIFARReader(targetDir, true, _testBatchSize, device: device))
            using (var model = new Model("model", _numClasses, device))
            using (var optimizer = torch.optim.Adam(model.parameters(), 0.001)) {

                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (var epoch = 1; epoch <= _epochs; epoch++) {
                    Train(model, optimizer, nll_loss(), train, epoch, _trainBatchSize, train.Size);
                    Test(model, nll_loss(), test, test.Size);
                    GC.Collect();

                    if (sw.Elapsed.TotalSeconds > 3600) break;
                }

                sw.Stop();
                Console.WriteLine($"Elapsed time {sw.Elapsed.TotalSeconds} s.");
            }
        }

        private class Model : Module
        {
            private readonly Module features;
            private readonly Module avgPool;
            private readonly Module classifier;

            public Model(string name, int numClasses, torch.Device device = null) : base(name)
            {
                features = Sequential(
                    ("c1", Conv2d(3, 64, kernelSize: 3, stride: 2, padding: 1)),
                    ("r1", ReLU(inPlace: true)),
                    ("mp1", MaxPool2d(kernelSize: new long[] { 2, 2 })),
                    ("c2", Conv2d(64, 192, kernelSize: 3, padding: 1)),
                    ("r2", ReLU(inPlace: true)),
                    ("mp2", MaxPool2d(kernelSize: new long[] { 2, 2 })),
                    ("c3", Conv2d(192, 384, kernelSize: 3, padding: 1)),
                    ("r3", ReLU(inPlace: true)),
                    ("c4", Conv2d(384, 256, kernelSize: 3, padding: 1)),
                    ("r4", ReLU(inPlace: true)),
                    ("c5", Conv2d(256, 256, kernelSize: 3, padding: 1)),
                    ("r5", ReLU(inPlace: true)),
                    ("mp3", MaxPool2d(kernelSize: new long[] { 2, 2 })));

                avgPool = AdaptiveAvgPool2d(new long[] { 2, 2 });

                classifier = Sequential(
                    ("d1", Dropout()),
                    ("l1", Linear(256 * 2 * 2, 4096)),
                    ("r1", ReLU(inPlace: true)),
                    ("d2", Dropout()),
                    ("l2", Linear(4096, 4096)),
                    ("r3", ReLU(inPlace: true)),
                    ("d3", Dropout()),
                    ("l3", Linear(4096, numClasses))
                );

                RegisterModule("features", features);
                RegisterModule("avg", avgPool);
                RegisterModule("classify", classifier);

                if (device != null && device.type == DeviceType.CUDA)
                    this.to(device);
            }

            public override Tensor forward(Tensor input)
            {
                using (var f = features.forward(input))
                using (var avg = avgPool.forward(f))

                using (var x = avg.view(new long[] { avg.shape[0], 256 * 2 * 2 }))
                    return classifier.forward(x);
            }
        }

        private static void Train(
            Model model,
            torch.optim.Optimizer optimizer,
            Loss loss,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            long batchSize,
            long size)
        {
            model.Train();

            int batchId = 1;
            long total = 0;
            long correct = 0;

            Console.WriteLine($"Epoch: {epoch}...");

            foreach (var (data, target) in dataLoader) {
                optimizer.zero_grad();

                using (var prediction = model.forward(data))
                using (var output = loss(log_softmax(prediction, 1), target)) {
                    output.backward();

                    optimizer.step();

                    total += target.shape[0];

                    using (var predicted = prediction.argmax(1))
                    using (var eq = predicted.eq(target))
                    using (var sum = eq.sum()) {
                        correct += sum.ToInt64();
                    }

                    if (batchId % _logInterval == 0) {
                        var count = Math.Min(batchId * batchSize, size);
                        Console.WriteLine($"\rTrain: epoch {epoch} [{count} / {size}] Loss: {output.ToSingle().ToString("0.0000")} Acc: { ((float)correct / total).ToString("0.0000") }");
                    }

                    batchId++;
                }
            }
        }

        private static void Test(
            Model model,
            Loss loss,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            long size)
        {
            model.Eval();

            double testLoss = 0;
            long correct = 0;
            int batchCount = 0;

            foreach (var (data, target) in dataLoader) {
                using (var prediction = model.forward(data))
                using (var output = loss(log_softmax(prediction, 1), target)) {

                    testLoss += output.ToSingle();
                    batchCount += 1;

                    using (var predicted = prediction.argmax(1))
                    using (var eq = predicted.eq(target))
                    using (var sum = eq.sum()) {
                        correct += sum.ToInt64();
                    }
                }

            }

            Console.WriteLine($"\rTest set: Average loss {(testLoss / batchCount).ToString("0.0000")} | Accuracy {((float)correct / size).ToString("0.0000")}");
        }
    }
}
