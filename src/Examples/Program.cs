// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

using System.Diagnostics;

namespace TorchSharp.Examples
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            Benchmark();
            //MNIST.Main(args);
            //AdversarialExampleGeneration.Main(args);
            //CIFAR10.Main(args);
            //SequenceToSequence.Main(args);
            //TextClassification.Main(args);
            //ImageTransforms.Main(args);
            //SpeechCommands.Main(args);
            //IOReadWrite.Main(args);
        }

        private static void Benchmark()
        {
            var sw = new Stopwatch();

            var iter = 1000;

            var seq = TorchSharp.torch.nn.Sequential(
                TorchSharp.torch.nn.Linear(100,100),
                TorchSharp.torch.nn.Linear(100, 100),
                TorchSharp.torch.nn.Linear(100, 100),
                TorchSharp.torch.nn.Linear(100, 100),
                TorchSharp.torch.nn.Linear(100, 100),
                TorchSharp.torch.nn.Linear(100, 100)
                );

            var input = TorchSharp.torch.rand(32, 100);

            sw.Start();

            for (var i = 0; i < iter; i++) {

                seq.forward(input);
            }

            sw.Stop();

            System.Console.WriteLine($"Elapsed time forward(Tensor): {sw.ElapsedMilliseconds}");

            sw.Start();

            var obj = (object)input;

            for (var i = 0; i < iter; i++) {

                seq.forward(obj);
            }

            sw.Stop();

            System.Console.WriteLine($"Elapsed time forward(object): {sw.ElapsedMilliseconds}");

        }
    }
}
