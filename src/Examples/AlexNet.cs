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
    /// <summary>
    /// Modified version of original AlexNet to fix CIFAR10 32x32 images.
    /// </summary>
    class AlexNet
    {
        private readonly static int _epochs = 5;
        private readonly static long _trainBatchSize = 100;
        private readonly static long _testBatchSize = 10000;
        private readonly static string _dataLocation = @"../../../../src/Examples/Data";

        private readonly static int _logInterval = 10;
        private readonly static int _numClasses = 10;

        static void Main(string[] args)
        {
            Torch.SetSeed(1);

            using (var train = Data.Loader.CIFAR10(_dataLocation, _trainBatchSize))
            using (var test = Data.Loader.CIFAR10(_dataLocation, _testBatchSize, false))
            using (var model = new Model(_numClasses))
            using (var optimizer = NN.Optimizer.Adam(model.Parameters(), 0.001))
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (var epoch = 1; epoch <= _epochs; epoch++)
                {
                    Train(model, optimizer, NLL(), train, epoch, _trainBatchSize, train.Size());
                    Test(model, NLL(), test, test.Size());
                }

                sw.Stop();
                Console.WriteLine($"Elapsed time {sw.ElapsedMilliseconds}.");
                Console.ReadLine();
            }
        }

        private class Model : NN.Module
        {
            private readonly NN.Sequential features;
            private readonly NN.AdaptiveAvgPool2D avgPool;
            private readonly NN.Sequential classifier;

            public Model(int numClasses)
            {
                features = Sequential(
                    Conv2D(3, 64, kernelSize: 3, stride: 2, padding: 1),
                    Relu(inPlace: true),
                    MaxPool2D(kernelSize: new long[] { 2 }),
                    Conv2D(64, 192, kernelSize: 3, padding: 1),
                    Relu(inPlace: true),
                    MaxPool2D(kernelSize: new long[] { 2 }),
                    Conv2D(192, 384, kernelSize: 3, padding: 1),
                    Relu(inPlace: true),
                    Conv2D(384, 256, kernelSize: 3, padding: 1),
                    Relu(inPlace: true),
                    Conv2D(256, 256, kernelSize: 3, padding: 1),
                    Relu(inPlace: true),
                    MaxPool2D(kernelSize: new long[] { 2 }));

                avgPool = AdaptiveAvgPool2D(new long[] { 2, 2 });

                classifier = Sequential(
                    Dropout(),
                    Linear(256 * 2 * 2, 4096),
                    Relu(inPlace: true),
                    Dropout(),
                    Linear(4096, 4096),
                    Relu(inPlace: true),
                    Linear(4096, numClasses)
                );

                RegisterModule (features);
                RegisterModule (avgPool);
                RegisterModule(classifier);
            }

            public TorchTensor Forward(TorchTensor input)
            {
                using (var f = features.Forward(input))
                using (var avg = avgPool.Forward(f))

                using (var x = avg.View(new long[] { avg.Shape[0], 256 * 2 * 2 }))
                    return classifier.Forward(x);
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
            long total = 0;
            long correct = 0;

            foreach (var (data, target) in dataLoader)
            {
                optimizer.ZeroGrad();

                using (var prediction = model.Forward(data))
                using (var output = loss(LogSoftMax(prediction, 1), target))
                {
                    output.Backward();

                    optimizer.Step();

                    var predicted = prediction.Argmax(1);
                    total += target.Shape[0];
                    correct += predicted.Eq(target).Sum().DataItem<long>();

                    if (batchId % _logInterval == 0)
                    {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {output.DataItem<float>()} Acc: { (float)correct / total }");
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
            long correct = 0;

            foreach (var (data, target) in dataLoader)
            {
                using (var prediction = model.Forward(data))
                using (var output = loss(LogSoftMax(prediction, 1), target))
                {
                    testLoss += output.DataItem<float>();

                    var pred = prediction.Argmax(1);

                    correct += pred.Eq(target).Sum().DataItem<long>();

                    data.Dispose();
                    target.Dispose();
                    pred.Dispose();
                }

            }

            Console.WriteLine($"\rTest set: Average loss {testLoss} | Accuracy {(float)correct / size}");
        }
    }
}
