using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp.NN;
using TorchSharp;

using TorchSharp.Tensor;
using static TorchSharp.NN.LossFunction;
using System.Diagnostics;
using TorchSharp.Data;

namespace Examples
{
    public class VGG
    {
        private readonly static int _epochs = 5;
        private readonly static long _trainBatchSize = 100;
        private readonly static long _testBatchSize = 10000;
        private readonly static string _dataLocation = @"../../../../src/Examples/Data";

        private readonly static int _logInterval = 10;
        private readonly static int _numClasses = 10;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="args"></param>
        static void main(String[] args)
        {
            Torch.SetSeed(1);
            using (var train = Loader.CIFAR10(_dataLocation, _trainBatchSize))
            using (var test = Loader.CIFAR10(_dataLocation, _testBatchSize, false))
            using (var model = new Model("VGG11"))
            using (var optimizer = Optimizer.Adam(model.Parameters(), 0.001))
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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="model"></param>
        /// <param name="loss"></param>
        /// <param name="dataLoader"></param>
        /// <param name="size"></param>
        private static void Test(
           TorchSharp.NN.Module model,
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
                using (var output = loss(TorchSharp.NN.Module.LogSoftMax(prediction, 1), target))
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

        /// <summary>
        /// 
        /// </summary>
        private static void Train(Module module, Optimizer optimizer, Loss loss, 
            IEnumerable<(TorchTensor, TorchTensor)> dataLoader, int epoch, long batchSize, long size
            )
        {
            module.Train();

            int batchId = 1;
            long total = 0;
            long correct = 0;

            foreach(var (data, target) in dataLoader)
            {
                optimizer.ZeroGrad();
                using (var prediction = module.Forward(data))
                using (var output = loss(TorchSharp.NN.Module.LogSoftMax(prediction, 1), target))
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

        /// <summary>
        ///     
        /// </summary>
        private class Model : Module
        {
            Sequential sequential;
            Module classifier;
            private readonly static Dictionary<string, object[]> cfgs = new Dictionary<string, object[]>();

            /// <summary>
            /// 
            /// </summary>
            /// <param name="VGGName"></param>
            public Model(string VGGName) : base()
            {
                object[] vgg11 = new object[] { 64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' };
                object[] vgg13 = new object[] { 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' };
                object[] vgg16 = new object[] { 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M' };
                object[] vgg19 = new object[] { 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M' };

                cfgs.Add("VGG11", vgg11);
                cfgs.Add("VGG13", vgg13);
                cfgs.Add("VGG16", vgg16);
                cfgs.Add("VGG19", vgg19);

                // super(VGG, self).__init__()  dont really need it, i think

                object[] cfg = cfgs.GetValueOrDefault(VGGName);
                this.sequential = MakeSequential(cfg);
                classifier = Linear(320, 50);

            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <returns></returns>
            public override TorchTensor Forward(TorchTensor input)
            {
                TorchTensor res = this.sequential.Forward(input);
                res = res.View(new long[] { -1, 0 });
                return classifier.Forward(res);
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="cfg"></param>
            /// <returns></returns>
            private Sequential MakeSequential(object[] cfg)
            {
                List<Module> layers = new List<Module>();
                int in_channels = 3;

                foreach (object x in cfg)
                {
                    if (x.ToString() == "M")
                        layers.Add(Module.MaxPool2D(new long[2], new long[1]));
                    else
                    {
                        layers.Add(Module.Conv2D(in_channels, (long)x, 3, padding: 1));
                        //layers.Add(Module.Bat);
                        layers.Add(Module.Relu(inPlace: true));
                        in_channels = (int)x;
                    }
                }
                // layers.Add(Module.AdaptiveAvgPool2D();

                return Module.Sequential(layers.ToArray());
            }

        }
    }
}
