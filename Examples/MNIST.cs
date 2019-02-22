using System;
using System.Collections.Generic;
using TorchSharp.Tensor;

namespace TorchSharp.Examples
{
    public class MNIST
    {
        static void Main(string[] args)
        {
            var train = Data.Loader.MNIST(@"E:/Source/Repos/LibTorchSharp/MNIST", 64, out int size);

            var model = new Model();

            var optimizer = NN.Optimizer.SGD(model.Parameters(), 0.01, 0.5);

            for (var epoch = 1; epoch <= 10; epoch++)
            {
                Train(model, optimizer, train, epoch, size);
            }
        }

        private class Model : NN.Module
        {
            private NN.Module conv1 = NN.Module.Conv2D(1, 10, 5);
            private NN.Module conv2 = NN.Module.Conv2D(10, 20, 5);
            private NN.Module fc1 = NN.Module.Linear(320, 50);
            private NN.Module fc2 = NN.Module.Linear(50, 10);

            public Model() : base(IntPtr.Zero)
            {
                RegisterModule(conv1);
                RegisterModule(conv2);
                RegisterModule(fc1);
                RegisterModule(fc2);
            }

            public override ITorchTensor<float> Forward<T>(ITorchTensor<T> tensor)
            {
                var x = conv1.Forward(tensor);
                x = NN.Module.MaxPool2D(x, 2);
                x = NN.Module.Relu(x);

                x = conv2.Forward(x);
                x = NN.Module.FeatureDropout(x);
                x = NN.Module.MaxPool2D(x, 2);

                x = x.View(new long[] { -1, 320 });

                x = fc1.Forward(x);
                x = NN.Module.Relu(x);
                x = NN.Module.Dropout(x, 0.5, _isTraining);

                x = fc2.Forward(x);

                return NN.Module.LogSoftMax(x, 1);
            }
        }

        private static void Train(NN.Module model, NN.Optimizer optimizer, IEnumerable<(ITorchTensor<float>, ITorchTensor<float>)> dataLoader, int epoch, int size)
        {
            model.Train();

            int batchId = 0;

            foreach (var (data, target) in dataLoader)
            {
                optimizer.ZeroGrad();

                var output = model.Forward(data);

                var loss = NN.LossFunction.NLL(output, target);

                loss.Backward();

                optimizer.Step();

                batchId++;

                Console.WriteLine($"\rTrain Epoch: {epoch} [{batchId} / {size}] Loss: {loss.Item}");
            }
        }
    }
}
