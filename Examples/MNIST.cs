using System;
using System.Collections.Generic;
using System.Diagnostics;
using TorchSharp.Tensor;

namespace TorchSharp.Examples
{
    public class MNIST
    {
        private readonly static int _epochs = 10;
        private readonly static long _trainBatchSize = 64;
        private readonly static long _testBatchSize = 1000;
        private readonly static string _dataLocation = @"../../../Data";

        private readonly static int _logInterval = 10;

        static void Main(string[] args)
        {
            Torch.SetSeed(1);

            using (var train = Data.Loader.MNIST(_dataLocation, _trainBatchSize))
            using (var test = Data.Loader.MNIST(_dataLocation, _testBatchSize, false))
            using (var model = new Model())
            using (var optimizer = NN.Optimizer.SGD(model.Parameters(), 0.01, 0.5))
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (var epoch = 1; epoch <= _epochs; epoch++)
                {
                    Train(model, optimizer, train, epoch, _trainBatchSize, train.Size());
                    Test(model, test, test.Size());
                }

                sw.Stop();
                Console.WriteLine($"Elapsed time {sw.ElapsedMilliseconds}.");
                Console.ReadLine();
            }
        }

        private class Model : NN.Module
        {
            private NN.Module conv1 = Conv2D(1, 10, 5);
            private NN.Module conv2 = Conv2D(10, 20, 5);
            private NN.Module fc1 = Linear(320, 50);
            private NN.Module fc2 = Linear(50, 10);

            public Model() : base(IntPtr.Zero)
            {
                RegisterModule(conv1);
                RegisterModule(conv2);
                RegisterModule(fc1);
                RegisterModule(fc2);
            }

            public override ITorchTensor<float> Forward<T>(params ITorchTensor<T>[] tensors)
            {
                using (var l11 = conv1.Forward(tensors))
                using (var l12 = MaxPool2D(l11, 2))
                using (var l13 = Relu(l12))

                using (var l21 = conv2.Forward(l13))
                using (var l22 = FeatureDropout(l21))
                using (var l23 = MaxPool2D(l22, 2))
                using (var l24 = Relu(l23))

                using (var x = l24.View(new long[] { -1, 320 }))

                using (var l31 = fc1.Forward(x))
                using (var l32 = Relu(l31))
                using (var l33 = Dropout(l32, 0.5, _isTraining))

                using (var l41 = fc2.Forward(l33))

                return LogSoftMax(l41, 1);
            }
        }

        private static void Train(
            NN.Module model, 
            NN.Optimizer optimizer,
            IEnumerable<(ITorchTensor<int>, ITorchTensor<int>)> dataLoader,
            int epoch,
            long batchSize, 
            long size)
        {
            model.Train();

            int batchId = 1;

            foreach (var (data, target) in dataLoader)
            {
                optimizer.ZeroGrad();

                using (var output = model.Forward(data))
                using (var loss = NN.LossFunction.NLL(output, target))
                {
                    loss.Backward();

                    optimizer.Step();

                    if (batchId % _logInterval == 0)
                    {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {loss.DataItem}");
                    }

                    batchId++;

                    data.Dispose();
                    target.Dispose();
                }
            }
        }

        private static void Test(
            NN.Module model,
            IEnumerable<(ITorchTensor<int>, ITorchTensor<int>)> dataLoader,
            long size)
        {
            model.Eval();

            double testLoss = 0;
            int correct = 0;

            foreach (var (data, target) in dataLoader)
            {
                using (var output = model.Forward(data))
                using (var loss = NN.LossFunction.NLL(output, target, reduction: NN.Reduction.Sum))
                {
                    testLoss += loss.DataItem;

                    var pred = output.Argmax(1);

                    correct += pred.Eq(target).Sum().DataItem; // Memory leak here

                    data.Dispose();
                    target.Dispose();
                    pred.Dispose();
                }

            }

            Console.WriteLine($"\rTest set: Average loss {testLoss / size} | Accuracy {(double)correct / size}");
        }
    }
}
