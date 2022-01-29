// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Simple MNIST Convolutional model.
    /// </summary>
    /// <remarks>
    /// There are at least two interesting data sets to use with this example:
    /// 
    /// 1. The classic MNIST set of 60000 images of handwritten digits.
    ///
    ///     It is available at: http://yann.lecun.com/exdb/mnist/
    ///     
    /// 2. The 'fashion-mnist' data set, which has the exact same file names and format as MNIST, but is a harder
    ///    data set to train on. It's just as large as MNIST, and has the same 60/10 split of training and test
    ///    data.
    ///    It is available at: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
    ///
    /// In each case, there are four .gz files to download. Place them in a folder and then point the '_dataLocation'
    /// constant below at the folder location.
    /// </remarks>
    public class MNIST
    {
        private static int _epochs = 1;
        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 128;

        private readonly static int _logInterval = 100;

        internal static void Main(string[] args)
        {
            var dataset = args.Length > 0 ? args[0] : "mnist";
            var datasetPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", dataset);

            torch.random.manual_seed(1);

            var cwd = Environment.CurrentDirectory;

            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Running MNIST on {device.type.ToString()}");
            Console.WriteLine($"Dataset: {dataset}");

            var sourceDir = datasetPath;
            var targetDir = Path.Combine(datasetPath, "test_data");

            if (!Directory.Exists(targetDir)) {
                Directory.CreateDirectory(targetDir);
                Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-images-idx3-ubyte.gz"), targetDir);
                Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-labels-idx1-ubyte.gz"), targetDir);
                Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-images-idx3-ubyte.gz"), targetDir);
                Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-labels-idx1-ubyte.gz"), targetDir);
            }

            if (device.type == DeviceType.CUDA) {
                _trainBatchSize *= 4;
                _testBatchSize *= 4;
            }

            var model = new Model("model", device);

            var normImage = torchvision.transforms.Normalize(new double[] { 0.1307 }, new double[] { 0.3081 }, device: (Device)device);

            using (MNISTReader train = new MNISTReader(targetDir, "train", _trainBatchSize, device: device, shuffle: true, transform: normImage),
                                test = new MNISTReader(targetDir, "t10k", _testBatchSize, device: device, transform: normImage)) {

                TrainingLoop(dataset, device, model, train, test);
            }
        }

        internal static void TrainingLoop(string dataset, Device device, Model model, MNISTReader train, MNISTReader test)
        {
            if (device.type == DeviceType.CUDA) {
                _epochs *= 4;
            }

            var optimizer = torch.optim.Adam(model.parameters());

            var scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.75);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (var epoch = 1; epoch <= _epochs; epoch++) {

                using (var d = torch.NewDisposeScope()) {

                    Train(model, optimizer, nll_loss(reduction: Reduction.Mean), device, train, epoch, train.BatchSize, train.Size);
                    Test(model, nll_loss(reduction: torch.nn.Reduction.Sum), device, test, test.Size);

                    Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");
                    scheduler.step();
                }
            }

            sw.Stop();
            Console.WriteLine($"Elapsed time: {sw.Elapsed.TotalSeconds:F1} s.");

            Console.WriteLine("Saving model to '{0}'", dataset + ".model.bin");
            model.save(dataset + ".model.bin");
        }

        internal class Model : Module
        {
            private Module conv1 = Conv2d(1, 32, 3);
            private Module conv2 = Conv2d(32, 64, 3);
            private Module fc1 = Linear(9216, 128);
            private Module fc2 = Linear(128, 10);

            // These don't have any parameters, so the only reason to instantiate
            // them is performance, since they will be used over and over.
            private Module pool1 = MaxPool2d(kernelSize: new long[] { 2, 2 });

            private Module relu1 = ReLU();
            private Module relu2 = ReLU();
            private Module relu3 = ReLU();

            private Module dropout1 = Dropout(0.25);
            private Module dropout2 = Dropout(0.5);

            private Module flatten = Flatten();
            private Module logsm = LogSoftmax(1);

            public Model(string name, torch.Device device = null) : base(name)
            {
                RegisterComponents();

                if (device != null && device.type == DeviceType.CUDA)
                    this.to(device);
            }

            public override Tensor forward(Tensor input)
            {
                var l11 = conv1.forward(input);
                var l12 = relu1.forward(l11);

                var l21 = conv2.forward(l12);
                var l22 = relu2.forward(l21);
                var l23 = pool1.forward(l22);
                var l24 = dropout1.forward(l23);

                var x = flatten.forward(l24);

                var l31 = fc1.forward(x);
                var l32 = relu3.forward(l31);
                var l33 = dropout2.forward(l32);

                var l41 = fc2.forward(l33);

                return logsm.forward(l41);
            }
        }


        private static void Train(
            Model model,
            torch.optim.Optimizer optimizer,
            Loss loss,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            long batchSize,
            long size)
        {
            model.train();

            int batchId = 1;

            Console.WriteLine($"Epoch: {epoch}...");

            using (var d = torch.NewDisposeScope()) {

                foreach (var (data, target) in dataLoader) {
                    optimizer.zero_grad();

                    var prediction = model.forward(data);
                    var output = loss(prediction, target);

                    output.backward();

                    optimizer.step();

                    if (batchId % _logInterval == 0) {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {output.ToSingle():F4}");
                    }

                    batchId++;

                    d.DisposeEverything();
                }
            }
        }

        private static void Test(
            Model model,
            Loss loss,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            long size)
        {
            model.eval();

            double testLoss = 0;
            int correct = 0;

            using (var d = torch.NewDisposeScope()) {

                foreach (var (data, target) in dataLoader) {
                    var prediction = model.forward(data);
                    var output = loss(prediction, target);
                    testLoss += output.ToSingle();

                    var pred = prediction.argmax(1);
                    correct += pred.eq(target).sum().ToInt32();

                    d.DisposeEverything();
                }
            }

            Console.WriteLine($"Size: {size}, Total: {size}");

            Console.WriteLine($"\rTest set: Average loss {(testLoss / size):F4} | Accuracy {((double)correct / size):P2}");
        }
    }
}
