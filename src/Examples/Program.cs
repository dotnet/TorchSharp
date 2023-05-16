// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

namespace TorchSharp.Examples
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            MNIST.Main(args);
            //AdversarialExampleGeneration.Main(args);
            //CIFAR10.Main(args);
            //SequenceToSequence.Main(args);
            //TextClassification.Main(args);
            //ImageTransforms.Main(args);
            //SpeechCommands.Main(args);
            IOReadWrite.Main(args);
            Tensorboard.Main(args).Wait();
        }
    }
}
