// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
module TorchSharp.Examples.AdversarialExampleGeneration

open System
open System.IO
open System.Diagnostics

open TorchSharp
open type TorchSharp.torch.nn
open type TorchSharp.torch.optim
open type TorchSharp.Scalar
open type TorchSharp.torchvision.datasets

open TorchSharp.Examples
open type TorchSharp.torch.utils.data

// FGSM Attack
//
// Based on : https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
//
// There are at least two interesting data sets to use with this example:
//
// 1. The classic MNIST set of 60000 images of handwritten digits.
//
//     It is available at: http://yann.lecun.com/exdb/mnist/
//
// 2. The 'fashion-mnist' data set, which has the exact same file names and format as MNIST, but is a harder
//    data set to train on. It's just as large as MNIST, and has the same 60/10 split of training and test
//    data.
//    It is available at: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
//
// The example is based on the PyTorch tutorial, but the results from attacking the model are very different from
// what the tutorial article notes, at least on the machine where it was developed. There is an order-of-magnitude lower
// drop-off in accuracy in this version. That said, when running the PyTorch tutorial on the same machine, the
// accuracy trajectories are the same between .NET and Python. If the base convulutational model is trained
// using Python, and then used for the FGSM attack in both .NET and Python, the drop-off trajectories are extremenly
// close.

let mutable trainBatchSize = 64
let mutable testBatchSize = 128

let logInterval = 100

let cmdArgs = Environment.GetCommandLineArgs()
let dataset = if cmdArgs.Length = 2 then cmdArgs.[1] else "mnist"

let datasetPath = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData)

torch.random.manual_seed(1L) |> ignore

let hasCUDA = torch.cuda_is_available() //torch.cuda.is_available()

let device = if hasCUDA then torch.CUDA else torch.CPU

let _loss = torch.nn.NLLLoss()
let criterion x y = _loss.call(x,y)

let attack (image:torch.Tensor) (eps:Scalar) (data_grad:torch.Tensor) =
    use sign = data_grad.sign()
    (image + eps * sign).clamp(0.0.ToScalar(), 1.0.ToScalar())

let test (model:MNIST.Model) (eps:float) (data:Dataset) size =

    let mutable correct = 0

    let dataLoader = new DataLoader(data, testBatchSize, false, device=device)
    for dat in dataLoader do

        use d = torch.NewDisposeScope()
        let input = dat.["data"]
        let labels = dat.["label"]

        input.requires_grad <- true

        begin  // This is introduced in order to let a few tensors go out of scope before GC
            use estimate = input --> model
            use loss = criterion estimate labels

            model.zero_grad()
            loss.backward()

            use perturbed = attack input (eps.ToScalar()) (input.grad())
            use final = perturbed --> model
            correct <- correct + final.argmax(1L).eq(labels).sum().ToInt32()
        end

    float correct / size

let run epochs =

    printfn $"Running AdversarialExampleGeneration on {device.``type``.ToString()}"
    printfn $"Dataset: {dataset}"

    if device.``type`` = DeviceType.CUDA then
        trainBatchSize <- trainBatchSize * 4
        testBatchSize <- testBatchSize * 4

    let normImage = torchvision.transforms.Normalize( [|0.1307|], [|0.3081|])
    use testData = MNIST(datasetPath, false, true, target_transform=normImage)

    let modelFile = dataset + ".model.bin"

    let model =
        if not (File.Exists(modelFile)) then
            printfn $"\n  Running MNIST on {device.``type``.ToString()} in order to pre-train the model."

            let model = new MNIST.Model("model",device)

            use train = MNIST(datasetPath, true, true, target_transform=normImage)
            MNIST.trainingLoop model epochs dataset train testData |> ignore

            printfn "Moving on to the Adversarial model.\n"

            model

        else
            let model = new MNIST.Model("model", torch.CPU)
            model.load(modelFile) |> ignore
            model

    model.``to``(device) |> ignore

    model.eval()

    let epsilons = [| 0.0; 0.05; 0.1; 0.15; 0.20; 0.25; 0.30; 0.35; 0.40; 0.45; 0.50|]

    for eps in epsilons do
        let attacked = test model eps testData (float testData.Count)
        printfn $"Epsilon: {eps:F2}, accuracy: {attacked:P2}"
