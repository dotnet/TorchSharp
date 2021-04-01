// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
using System.IO.Compression;
using ICSharpCode.SharpZipLib.Core;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using System.Collections.Generic;
using System.Diagnostics;
using TorchSharp.Tensor;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using static TorchSharp.NN.Functions;

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
        private readonly static string _dataLocation = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "fashion-mnist");

        private static int _epochs = 10;
        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 128;

        private readonly static int _logInterval = 100;

        static void Main(string[] args)

        {
            Torch.SetSeed(1);

            var cwd = Environment.CurrentDirectory;

            //var device = Device.CPU; //Torch.IsCudaAvailable() ? Device.CUDA : Device.CPU;
            var device = Torch.IsCudaAvailable() ? Device.CUDA : Device.CPU;
            Console.WriteLine($"Running on {device.Type.ToString()}");

            if (device.Type == DeviceType.CUDA) {
                _trainBatchSize *= 4;
                _testBatchSize *= 4;
                _epochs *= 16;
            }

            var sourceDir = _dataLocation;
            var targetDir = Path.Combine(_dataLocation, "test_data");

            if (!Directory.Exists(targetDir)) {
                Directory.CreateDirectory(targetDir);
                Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-images-idx3-ubyte.gz"), targetDir);
                Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-labels-idx1-ubyte.gz"), targetDir);
                Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-images-idx3-ubyte.gz"), targetDir);
                Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-labels-idx1-ubyte.gz"), targetDir);
            }

            using (var train = new MNISTReader(targetDir, "train", _trainBatchSize, device: device, shuffle: true))
            using (var test = new MNISTReader(targetDir, "t10k", _testBatchSize, device: device))
            //using (var model = new Model("model", device))
            using (var model = GetModel(device))
            using (var optimizer = NN.Optimizer.SGD(model.parameters(), 0.01, 0.5))
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (var epoch = 1; epoch <= _epochs; epoch++)
                {
                    Train(model, optimizer, nll_loss(), train, epoch, _trainBatchSize, train.Size);
                    Test(model, nll_loss(reduction: NN.Reduction.Sum), test, test.Size);

                    Console.WriteLine($"Pre-GC memory:  {GC.GetTotalMemory(false)}");
                    GC.Collect();
                    Console.WriteLine($"Post-GC memory: {GC.GetTotalMemory(false)}");
                }

                sw.Stop();
                Console.WriteLine($"Elapsed time: {sw.Elapsed.TotalSeconds} s.");
                Console.ReadLine();
            }
        }

        private static Sequential GetModel(Device device)
        {
            var seq = Sequential(
                ("conv1", Conv2D(1, 10, 5)),
                ("pool1", MaxPool2D(kernelSize: new long[] { 2, 2 })),
                ("relu1", ReLU()),
                ("conv2", Conv2D(10, 20, 5)),
                ("dropout1", FeatureAlphaDropout()),
                ("pool2", MaxPool2D(kernelSize: new long[] { 2, 2 })),
                ("relu2", ReLU()),
                ("flatten", Flatten()),
                ("fc1", Linear(320, 64)),
                ("relu3", ReLU()),
                ("dropout2", Dropout()),
                ("fc2", Linear(64, 10)),
                ("logsm", LogSoftmax(1)));

            return (device != null && device.Type == DeviceType.CUDA) ?
                seq.to(device) as Sequential :
                seq;
        }

        private class Model : CustomModule
        {
            private Conv2D conv1 = Conv2D(1, 10, 5);
            private Conv2D conv2 = Conv2D(10, 20, 5);
            private Linear fc1 = Linear(320, 64);
            private Linear fc2 = Linear(64, 10);

            private MaxPool2D pool1 = MaxPool2D(kernelSize: new long[] { 2, 2 });
            private MaxPool2D pool2 = MaxPool2D(kernelSize: new long[] { 2, 2 });

            private ReLU relu1 = ReLU();
            private ReLU relu2 = ReLU();
            private ReLU relu3 = ReLU();

            private FeatureAlphaDropout dropout1 = FeatureAlphaDropout();
            private Dropout dropout2 = Dropout();

            private Flatten flatten = Flatten();
            private LogSoftmax logsm = LogSoftmax(1);


            public Model(string name, Device device = null) : base(name)
            {
                RegisterModule("conv1", conv1);
                RegisterModule("conv2", conv2);
                RegisterModule("lin1", fc1);
                RegisterModule("lin2", fc2);
                RegisterModule("pool1", pool1);
                RegisterModule("pool2", pool2);
                RegisterModule("relu1", relu1);
                RegisterModule("relu2", relu2);
                RegisterModule("relu3", relu3);
                RegisterModule("drop1", dropout1);
                RegisterModule("drop2", dropout2);
                RegisterModule("logsoft", logsm);

                if (device != null && device.Type == DeviceType.CUDA)
                    this.to(device);
            }

            public override TorchTensor forward(TorchTensor input)
            {
                using (var l11 = conv1.forward(input))
                using (var l12 = pool1.forward(l11))
                using (var l13 = relu1.forward(l12))

                using (var l21 = conv2.forward(l13))
                using (var l22 = pool2.forward(l21))
                using (var l23 = dropout1.forward(l22))
                using (var l24 = relu2.forward(l23))

                using (var x = flatten.forward(l24))

                using (var l31 = fc1.forward(x))
                using (var l32 = relu3.forward(l31))
                using (var l33 = dropout2.forward(l32))

                using (var l41 = fc2.forward(l33))
                return logsm.forward(l41);
            }
        }


        private static void Train(
            Sequential model,
            NN.Optimizer optimizer,
            Loss loss,
            IEnumerable<(TorchTensor, TorchTensor)> dataLoader,
            int epoch,
            long batchSize,
            long size)
        {
            model.Train();

            int batchId = 1;

            Console.WriteLine($"Epoch: {epoch}...");
            foreach (var (data, target) in dataLoader)
            {
                optimizer.zero_grad();

                using (var prediction = model.forward(data))
                using (var output = loss(prediction, target))
                {
                    output.backward();

                    optimizer.step();

                    if (batchId % _logInterval == 0)
                    {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {output.ToSingle()}");
                    }

                    batchId++;
                }
            }
        }

        private static void Test(
            Sequential model,
            Loss loss,
            IEnumerable<(TorchTensor, TorchTensor)> dataLoader,
            long size)
        {
            model.Eval();

            double testLoss = 0;
            int correct = 0;

            foreach (var (data, target) in dataLoader)
            {
                using (var prediction = model.forward(data))
                using (var output = loss(prediction, target))
                    {
                    testLoss += output.ToSingle();

                    var pred = prediction.argmax(1);
                    correct += pred.eq(target).sum().ToInt32();

                    pred.Dispose();
                }

            }

            Console.WriteLine($"\rTest set: Average loss {testLoss / size} | Accuracy {(double)correct / size}");
        }
    }
}
