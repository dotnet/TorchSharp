// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils.data;

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
            var datasetPath = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);

            torch.random.manual_seed(1);

            var cwd = Environment.CurrentDirectory;

            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Running MNIST on {device.type.ToString()}");
            Console.WriteLine($"Dataset: {dataset}");

            if (device.type == DeviceType.CUDA) {
                _trainBatchSize *= 4;
                _testBatchSize *= 4;
            }

            using var model = new Model("model", device);

            var normImage = torchvision.transforms.Normalize(new double[] { 0.1307 }, new double[] { 0.3081 });

            using (Dataset train_data = torchvision.datasets.MNIST(datasetPath, true, download: true, target_transform: normImage),
                           test_data = torchvision.datasets.MNIST(datasetPath, false, download: true, target_transform: normImage)) {

                TrainingLoop("mnist", device, model, train_data, test_data);
            }
        }

        internal static void TrainingLoop(string dataset, Device device, Model model, Dataset train_data, Dataset test_data)
        {
            using var train = new DataLoader(train_data, _trainBatchSize, device: device, shuffle: true);
            using var test = new DataLoader(test_data, _testBatchSize, device: device, shuffle: false);

            if (device.type == DeviceType.CUDA) {
                _epochs *= 4;
            }

            var optimizer = torch.optim.Adam(model.parameters());

            var scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.75);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (var epoch = 1; epoch <= _epochs; epoch++) {

                using (var d = torch.NewDisposeScope()) {

                    Train(model, optimizer, torch.nn.NLLLoss(reduction: Reduction.Mean), train, epoch, train_data.Count);
                    Test(model, torch.nn.NLLLoss(reduction: torch.nn.Reduction.Sum), test, test_data.Count);

                    Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");
                    scheduler.step();
                }
            }

            sw.Stop();
            Console.WriteLine($"Elapsed time: {sw.Elapsed.TotalSeconds:F1} s.");

            Console.WriteLine("Saving model to '{0}'", dataset + ".model.bin");
            model.save(dataset + ".model.bin");
        }

        internal class Model : Module<Tensor, Tensor>
        {
            private Module<Tensor, Tensor> conv1 = Conv2d(1, 32, 3);
            private Module<Tensor, Tensor> conv2 = Conv2d(32, 64, 3);
            private Module<Tensor, Tensor> fc1 = Linear(9216, 128);
            private Module<Tensor, Tensor> fc2 = Linear(128, 10);

            // These don't have any parameters, so the only reason to instantiate
            // them is performance, since they will be used over and over.
            private Module<Tensor, Tensor> pool1 = MaxPool2d(kernelSize: new long[] { 2, 2 });

            private Module<Tensor, Tensor> relu1 = ReLU();
            private Module<Tensor, Tensor> relu2 = ReLU();
            private Module<Tensor, Tensor> relu3 = ReLU();

            private Module<Tensor, Tensor> dropout1 = Dropout(0.25);
            private Module<Tensor, Tensor> dropout2 = Dropout(0.5);

            private Module<Tensor, Tensor> flatten = Flatten();
            private Module<Tensor, Tensor> logsm = LogSoftmax(1);

            public Model(string name, torch.Device device = null) : base(name)
            {
                RegisterComponents();

                if (device != null && device.type == DeviceType.CUDA)
                    this.to(device);
            }

            public override Tensor forward(Tensor input)
            {
                // All these modules are private to the model and won't have hooks,
                // so we can use 'forward' instead of 'call'
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

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    conv1.Dispose();
                    conv2.Dispose();
                    fc1.Dispose();
                    fc2.Dispose();
                    pool1.Dispose();
                    relu1.Dispose();
                    relu2.Dispose();
                    relu3.Dispose();
                    dropout1.Dispose();
                    dropout2.Dispose();
                    flatten.Dispose();
                    logsm.Dispose();
                    ClearModules();
                }
                base.Dispose(disposing);
            }
        }


        private static void Train(
            Model model,
            torch.optim.Optimizer optimizer,
            Loss<torch.Tensor, torch.Tensor, torch.Tensor> loss,
            DataLoader dataLoader,
            int epoch,
            long size)
        {
            model.train();

            int batchId = 1;
            long total = 0;

            Console.WriteLine($"Epoch: {epoch}...");

            using (var d = torch.NewDisposeScope()) {

                foreach (var data in dataLoader) {
                    optimizer.zero_grad();

                    var target = data["label"];
                    var prediction = model.call(data["data"]);
                    var output = loss.call(prediction, target);

                    output.backward();

                    optimizer.step();

                    total += target.shape[0];

                    if (batchId % _logInterval == 0 || total == size) {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{total} / {size}] Loss: {output.ToSingle():F4}");
                    }

                    batchId++;

                    d.DisposeEverything();
                }
            }
        }

        private static void Test(
            Model model,
            Loss<torch.Tensor, torch.Tensor, torch.Tensor> loss,
            DataLoader dataLoader,
            long size)
        {
            model.eval();

            double testLoss = 0;
            int correct = 0;

            using (var d = torch.NewDisposeScope()) {

                foreach (var data in dataLoader) {
                    var prediction = model.call(data["data"]);
                    var output = loss.call(prediction, data["label"]);
                    testLoss += output.ToSingle();

                    var pred = prediction.argmax(1);
                    correct += pred.eq(data["label"]).sum().ToInt32();

                    d.DisposeEverything();
                }
            }

            Console.WriteLine($"Size: {size}, Total: {size}");

            Console.WriteLine($"\rTest set: Average loss {(testLoss / size):F4} | Accuracy {((double)correct / size):P2}");
        }
    }
}
