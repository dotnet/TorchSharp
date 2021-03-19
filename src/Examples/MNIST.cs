// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TorchSharp.Tensor;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using static TorchSharp.NN.Functions;
using static TorchSharp.NN.Losses;

namespace TorchSharp.Examples
{
    public class MNIST
    {
        private readonly static int _epochs = 10;
        private readonly static long _trainBatchSize = 64;
        private readonly static long _testBatchSize = 1000;
        private readonly static string _dataLocation = @"../../../../src/Examples/Data";

        private readonly static int _logInterval = 10;

        static void Main(string[] args)
        {
            Torch.SetSeed(1);

            using (var train = Data.Loader.MNIST(_dataLocation, _trainBatchSize))
            using (var test = Data.Loader.MNIST(_dataLocation, _testBatchSize, false))
            using (var model = new Model("model"))
            using (var optimizer = NN.Optimizer.SGD(model.GetParameters(), 0.01, 0.5))
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (var epoch = 1; epoch <= _epochs; epoch++)
                {
                    Train(model, optimizer, NLL(), train, epoch, _trainBatchSize, train.Size());
                    Test(model, NLL(reduction: NN.Reduction.Sum), test, test.Size());
                }

                sw.Stop();
                Console.WriteLine($"Elapsed time {sw.ElapsedMilliseconds}.");
                Console.ReadLine();
            }
        }

        private class Model : CustomModule
        {
            private Conv2D conv1 = Conv2D(1, 10, 5);
            private Conv2D conv2 = Conv2D(10, 20, 5);
            private Linear fc1 = Linear(320, 50);
            private Linear fc2 = Linear(50, 10);

            public Model(string name) : base(name)
            {
                RegisterModule("conv1", conv1);
                RegisterModule("conv2", conv2);
                RegisterModule("lin1", fc1);
                RegisterModule("lin2", fc2);
            }

            public override TorchTensor forward(TorchTensor input)
            {
                using (var l11 = conv1.forward(input))
                using (var l12 = MaxPool2D (l11, kernelSize: new long[]{ 2 }))
                using (var l13 = Relu(l12))

                using (var l21 = conv2.forward(l13))
                using (var l22 = FeatureAlphaDropout(l21))
                using (var l23 = MaxPool2D (l22, kernelSize: new long[] { 2 }))
                using (var l24 = Relu(l23))

                using (var x = l24.view(new long[] { -1, 320 }))

                using (var l31 = fc1.forward(x))
                using (var l32 = Relu(l31))
                using (var l33 = Dropout(l32))

                using (var l41 = fc2.forward(l33))

                    return LogSoftMax(l41, 1);
            }
        }

        private static void Train(
            Model model,
            NN.Optimizer optimizer,
            Loss loss,
            IEnumerable<(TorchTensor, TorchTensor)> dataLoader,
            int epoch,
            long batchSize,
            long size)
        {
            model.Train();

            int batchId = 1;

            foreach (var (data, target) in dataLoader)
            {
                optimizer.ZeroGrad();

                using (var prediction = model.forward(data))
                using (var output = loss(prediction, target))
                {
                    output.backward();

                    optimizer.Step();

                    if (batchId % _logInterval == 0)
                    {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {output.ToSingle()}");
                    }

                    batchId++;

                    data.Dispose();
                    target.Dispose();
                }
            }
        }

        private static void Test(
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
                using (var prediction = model.forward(data))
                using (var output = loss(prediction, target))
                {
                    testLoss += output.ToSingle();

                    var pred = prediction.argmax(1);

                    correct += pred.eq(target).sum().ToInt32(); // Memory leak here

                    data.Dispose();
                    target.Dispose();
                    pred.Dispose();
                }

            }

            Console.WriteLine($"\rTest set: Average loss {testLoss / size} | Accuracy {(double)correct / size}");
        }
    }
}
