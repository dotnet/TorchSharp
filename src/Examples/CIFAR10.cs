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
    /// Driver for various models trained and evaluated on the CIFAR10 small (32x32) color image data set.
    /// </summary>
    /// <remarks>
    /// The dataset for this example can be found at: https://www.cs.toronto.edu/~kriz/cifar.html
    /// Download the binary file, and place it in a dedicated folder, e.g. 'CIFAR10,' then edit
    /// the '_dataLocation' definition below to point at the right folder.
    ///
    /// Note: so far, CIFAR10 is supported, but not CIFAR100.
    /// </remarks>
    class CIFAR10
    {
        private readonly static string _dataset = "CIFAR10";
        private readonly static string _dataLocation = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", _dataset);

        private static int _epochs = 8;
        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 128;

        private readonly static int _logInterval = 25;
        private readonly static int _numClasses = 10;

        private readonly static int _timeout = 3600;    // One hour by default.

        internal static void Main(string[] args)
        {
            torch.random.manual_seed(1);

            var device =
                // This worked on a GeForce RTX 2080 SUPER with 8GB, for all the available network architectures.
                // It may not fit with less memory than that, but it's worth modifying the batch size to fit in memory.
                torch.cuda.is_available() ? torch.CUDA :
                torch.CPU;

            if (device.type == DeviceType.CUDA) {
                _trainBatchSize *= 8;
                _testBatchSize *= 8;
                _epochs *= 16;
            }

            var modelName = args.Length > 0 ? args[0] : "AlexNet";
            var epochs = args.Length > 1 ? int.Parse(args[1]) : _epochs;
            var timeout = args.Length > 2 ? int.Parse(args[2]) : _timeout;

            Console.WriteLine();
            Console.WriteLine($"\tRunning {modelName} with {_dataset} on {device.type.ToString()} for {epochs} epochs, terminating after {timeout/3600} hours.");
            Console.WriteLine();

            var sourceDir = _dataLocation;
            var targetDir = Path.Combine(_dataLocation, "test_data");

            if (!Directory.Exists(targetDir)) {
                Directory.CreateDirectory(targetDir);
                Utils.Decompress.ExtractTGZ(Path.Combine(sourceDir, "cifar-10-binary.tar.gz"), targetDir);
            }

            Module model = null;

            switch (modelName.ToLower()) {
            case "alexnet":
                model = new AlexNet(modelName, _numClasses, device);
                break;
            case "mobilenet":
                model = new MobileNet(modelName, _numClasses, device);
                break;
            case "vgg11":
                model = new VGG(modelName, _numClasses, device);
                break;
            case "vgg13":
                model = new VGG(modelName, _numClasses, device);
                break;
            case "vgg16":
                model = new VGG(modelName, _numClasses, device);
                break;
            case "vgg19":
                model = new VGG(modelName, _numClasses, device);
                break;
            }

            using (var train = new CIFARReader(targetDir, false, _trainBatchSize, shuffle: true, device: device))
            using (var test = new CIFARReader(targetDir, true, _testBatchSize, device: device))
            using (var optimizer = torch.optim.Adam(model.parameters(), 0.001)) {

                Stopwatch totalSW = new Stopwatch();
                totalSW.Start();

                for (var epoch = 1; epoch <= epochs; epoch++) {

                    Stopwatch epchSW = new Stopwatch();
                    epchSW.Start();

                    Train(model, optimizer, nll_loss(), train, epoch, _trainBatchSize, train.Size);
                    Test(model, nll_loss(), test, test.Size);
                    GC.Collect();

                    epchSW.Stop();
                    Console.WriteLine($"Elapsed time for this epoch: {epchSW.Elapsed.TotalSeconds} s.");

                    if (totalSW.Elapsed.TotalSeconds > timeout) break;
                }

                totalSW.Stop();
                Console.WriteLine($"Elapsed training time: {totalSW.Elapsed.TotalSeconds} s.");
            }

            model.Dispose();
        }

        private static void Train(
            Module model,
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

                using var prediction = model.forward(data);
                using var lsm = log_softmax(prediction, 1);
                using (var output = loss(lsm, target)) {

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
                        Console.WriteLine($"\rTrain: epoch {epoch} [{count} / {size}] Loss: {output.ToSingle().ToString("0.000000")} | Accuracy: { ((float)correct / total).ToString("0.000000") }");
                    }

                    batchId++;
                }

                GC.Collect();
            }
        }

        private static void Test(
            Module model,
            Loss loss,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            long size)
        {
            model.Eval();

            double testLoss = 0;
            long correct = 0;
            int batchCount = 0;

            foreach (var (data, target) in dataLoader) {

                using var prediction = model.forward(data);
                using var lsm = log_softmax(prediction, 1);
                using (var output = loss(lsm, target)) {

                    testLoss += output.ToSingle();
                    batchCount += 1;

                    using (var predicted = prediction.argmax(1))
                    using (var eq = predicted.eq(target))
                    using (var sum = eq.sum()) {
                        correct += sum.ToInt64();
                    }
                }

                GC.Collect();
            }

            Console.WriteLine($"\rTest set: Average loss {(testLoss / batchCount).ToString("0.0000")} | Accuracy {((float)correct / size).ToString("0.0000")}");
        }
    }
}
