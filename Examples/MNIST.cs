using System;
using System.Collections.Generic;
using TorchSharp.Tensor;

namespace TorchSharp.Examples
{
    public class MNIST
    {
        private readonly static int _epochs = 10;
        private readonly static long _batch = 64;
        private readonly static string _trainDataset = @"E:/Source/Repos/LibTorchSharp/MNIST";

        static void Main(string[] args)
        {
            using (var train = Data.Loader.MNIST(_trainDataset, _batch))
            using (var model = new Model())
            using (var optimizer = NN.Optimizer.SGD(model.Parameters(), 0.01, 0.5))
            {
                for (var epoch = 1; epoch <= _epochs; epoch++)
                {
                    Train(model, optimizer, train, epoch, _batch, train.Size());
                }
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

            public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
            {
                using (var l11 = conv1.Forward(tensor))
                using (var l12 = MaxPool2D(l11, 2))
                using (var l13 = Relu(l12))

                using (var l21 = conv2.Forward(l13))
                using (var l22 = FeatureDropout(l21))
                using (var l23 = MaxPool2D(l22, 2))

                using (var x = l23.View(new long[] { -1, 320 }))

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

                    batchId++;

                    Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {loss.Item}");

                    data.Dispose();
                    target.Dispose();
                }
            }
        }
    }
}
