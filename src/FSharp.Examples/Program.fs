// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

[<EntryPoint>]
let main argv =

    //TorchSharp.Examples.MNIST.run 2
    //TorchSharp.Examples.AdversarialExampleGeneration.run 4
    //TorchSharp.Examples.AlexNet.run 4
    //TorchSharp.Examples.SequenceToSequence.run 4
    //TorchSharp.Examples.TextClassification.run 15
    TorchSharp.Examples.ImageTransforms.run argv
    0 // return an integer exit code
