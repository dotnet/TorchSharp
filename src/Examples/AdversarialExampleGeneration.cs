// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
using System.Runtime.Serialization;

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
using static TorchSharp.Tensor.TensorExtensionMethods;

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
    public class AdversarialExampleGeneration
    {
        private readonly static string _dataLocation = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "mnist");

        private static int _epochs = 4;
        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 128;

        static void Main(string[] args)
        {
            var cwd = Environment.CurrentDirectory;

            var dataset = args.Length > 0 ? args[0] : "mnist";
            var datasetPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", dataset);

            Torch.SetSeed(1);

            //var device = Device.CPU;
            var device = Torch.IsCudaAvailable() ? Device.CUDA : Device.CPU;
            Console.WriteLine($"\n  Running AdversarialExampleGeneration on {device.Type.ToString()}\n");
            Console.WriteLine($"Dataset: {dataset}");

            if (device.Type == DeviceType.CUDA) {
                _trainBatchSize *= 4;
                _testBatchSize *= 4;
                _epochs *= 4;
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

            var normImage = TorchVision.Transforms.Normalize(new double[] { 0.1307 }, new double[] { 0.3081 }, device: device);

            using (var train = new MNISTReader(targetDir, "train", _trainBatchSize, device: device, shuffle: true, transform: normImage))
            using (var test = new MNISTReader(targetDir, "t10k", _testBatchSize, device: device, transform: normImage)) {

                var model = new Model("model", Device.CPU);

                var modelFile = dataset + ".model.bin";

                if (!File.Exists(modelFile)) {
                    // We need the model to be trained first, because we want to start with a trained model.
                    Console.WriteLine($"\n  Running MNIST on {device.Type.ToString()} in order to pre-train the model.");
                    MNIST.TrainingLoop(dataset, device, train, test);
                    Console.WriteLine("Moving on to the Adversarial model.\n");
                }

                model.load(modelFile);
                model.to(device);

                // Establish a baseline accuracy.

                Stopwatch sw = new Stopwatch();
                sw.Start();

                var baseline = TestBaseline(model, nll_loss(reduction: NN.Reduction.Sum), test, test.Size);

                Console.WriteLine($"\rBaseline model accuracy: {baseline}");

                sw.Stop();
                Console.WriteLine($"Elapsed time: {sw.Elapsed.TotalSeconds} s.");

                GC.Collect();
            }
        }

        private class Model : CustomModule
        {
            private Conv2d conv1 = Conv2d(1, 32, 3);
            private Conv2d conv2 = Conv2d(32, 64, 3);
            private Linear fc1 = Linear(9216, 128);
            private Linear fc2 = Linear(128, 10);

            // These don't have any parameters, so the only reason to instantiate
            // them is performance, since they will be used over and over.
            private MaxPool2d pool1 = MaxPool2d(kernelSize: new long[] { 2, 2 });

            private ReLU relu1 = ReLU();
            private ReLU relu2 = ReLU();
            private ReLU relu3 = ReLU();

            private FeatureAlphaDropout dropout1 = FeatureAlphaDropout();
            private Dropout dropout2 = Dropout();

            private Flatten flatten = Flatten();
            private LogSoftmax logsm = LogSoftmax(1);


            public Model(string name, Device device = null) : base(name)
            {
                RegisterComponents();

                if (device != null && device.Type == DeviceType.CUDA)
                    this.to(device);
            }

            public override TorchTensor forward(TorchTensor input)
            {
                var l11 = conv1.forward(input);
                var l12 = relu2.forward(l11);

                var l21 = conv2.forward(l12);
                var l22 = pool1.forward(l21);
                var l23 = dropout1.forward(l22);
                var l24 = relu2.forward(l23);

                var x = flatten.forward(l24);

                var l31 = fc1.forward(x);
                var l32 = relu3.forward(l31);
                var l33 = dropout2.forward(l32);

                var l41 = fc2.forward(l33);

                return logsm.forward(l41);
            }
        }

        private static double TestBaseline(
            Model model,
            Loss loss,
            IEnumerable<(TorchTensor, TorchTensor)> dataLoader,
            long size)
        {
            model.Eval();

            double testLoss = 0;
            int correct = 0;

            foreach (var (data, target) in dataLoader)
            {
                var prediction = model.forward(data);
                var output = loss(prediction, target);
                testLoss += output.ToSingle();

                var pred = prediction.argmax(1);
                correct += pred.eq(target).sum().ToInt32();

                pred.Dispose();

                GC.Collect();
            }

            return (double)correct / size;
        }
    }
}
