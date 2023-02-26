// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
module TorchSharp.Examples.AlexNet

open System
open System.IO
open System.Diagnostics

open TorchSharp

open type TorchSharp.torch.nn
open type TorchSharp.torch.optim
open type TorchSharp.torch.utils.data
open type TorchSharp.torchvision.datasets

// Modified version of original AlexNet to fix CIFAR10 32x32 images.
//
// The dataset for this example can be found at: https://www.cs.toronto.edu/~kriz/cifar.html
// Download the binary file, and place it in a dedicated folder, e.g. 'CIFAR10,' then edit
// the '_dataLocation' definition below to point at the right folder.
//
// Note: so far, CIFAR10 is supported, but not CIFAR100.

let mutable trainBatchSize = 64
let mutable testBatchSize = 128

let logInterval = 25
let numClasses = 10L

let cmdArgs = Environment.GetCommandLineArgs()
let dataset = "CIFAR10"

let datasetPath = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData)

torch.random.manual_seed(1L) |> ignore

let hasCUDA = torch.cuda_is_available() //torch.cuda.is_available()

let device = if hasCUDA then torch.CUDA else torch.CPU

let getDataFiles sourceDir targetDir =

    if not (Directory.Exists(targetDir)) then
        Directory.CreateDirectory(targetDir) |> ignore
        Utils.Decompress.ExtractTGZ(Path.Combine(sourceDir, "cifar-10-binary.tar.gz"), targetDir)

type Model(name,device:torch.Device) as this =
    inherit Module<torch.Tensor,torch.Tensor>(name)

    let features = Sequential(("c1", Conv2d(3L, 64L, kernelSize=3L, stride=2L, padding=1L) :> Module<torch.Tensor,torch.Tensor>),
                              ("r1", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                              ("mp1", MaxPool2d(kernelSize=[|2L; 2L|]) :> Module<torch.Tensor,torch.Tensor>),
                              ("c2", Conv2d(64L, 192L, kernelSize=3L, padding=1L) :> Module<torch.Tensor,torch.Tensor>),
                              ("r2", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                              ("mp2", MaxPool2d(kernelSize=[|2L; 2L|]) :> Module<torch.Tensor,torch.Tensor>),
                              ("c3", Conv2d(192L, 384L, kernelSize=3L, padding=1L) :> Module<torch.Tensor,torch.Tensor>),
                              ("r3", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                              ("c4", Conv2d(384L, 256L, kernelSize=3L, padding=1L) :> Module<torch.Tensor,torch.Tensor>),
                              ("r4", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                              ("c5", Conv2d(256L, 256L, kernelSize=3L, padding=1L) :> Module<torch.Tensor,torch.Tensor>),
                              ("r5", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                              ("mp3", MaxPool2d(kernelSize=[|2L; 2L|]) :> Module<torch.Tensor,torch.Tensor>),
                              ("avg", AdaptiveAvgPool2d([|2L; 2L|]) :> Module<torch.Tensor,torch.Tensor>))

    let classifier = Sequential(("d1", Dropout() :> Module<torch.Tensor,torch.Tensor>),
                                ("l1", Linear(256L * 2L * 2L, 4096L) :> Module<torch.Tensor,torch.Tensor>),
                                ("r6", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                                ("d2", Dropout() :> Module<torch.Tensor,torch.Tensor>),
                                ("l2", Linear(4096L, 4096L) :> Module<torch.Tensor,torch.Tensor>),
                                ("r7", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                                ("d3", Dropout() :> Module<torch.Tensor,torch.Tensor>),
                                ("l3", Linear(4096L, numClasses) :> Module<torch.Tensor,torch.Tensor>),
                                ("logsm", LogSoftmax(1L) :> Module<torch.Tensor,torch.Tensor>))

    do
        this.RegisterComponents()

        if device.``type`` = DeviceType.CUDA then
            this.``to``(device) |> ignore

    override _.forward(input) =

        let avg = features.call(input)
        let x = avg.view([|avg.shape.[0]; 256L*2L*2L|])

        classifier.call(x)

let _loss = torch.nn.NLLLoss()
let loss x y = _loss.call(x,y)

let train (model:Model) (optimizer:Optimizer) (dataLoader:DataLoader) epoch (size:int64) =

    model.train()

    let mutable batchID = 1
    let mutable total = 0L
    let mutable correct = 0L

    printfn $"Epoch: {epoch}..."

    for batch in dataLoader do

        use d = torch.NewDisposeScope()

        let input = batch.["data"]
        let labels = batch.["label"]

        optimizer.zero_grad()

        begin
            let estimate = input --> model
            let output = loss estimate labels

            output.backward()
            optimizer.step() |> ignore

            total <- total + labels.shape.[0]

            use sum = estimate.argmax(1L).eq(labels).sum()
            correct <- correct + sum.ToInt64()

            if batchID % logInterval = 0 || total = size then
                let outputString = output.ToSingle().ToString("0.0000")
                let accString = ((float correct) / (float total)).ToString("0.0000")
                printfn $"\rTrain: epoch {epoch} [{total} / {size}] Loss: {outputString} Acc: {accString}"

            batchID <- batchID + 1
        end

let test (model:Model) (dataLoader:DataLoader) (size:int) =
    model.eval()

    let sz = float32 size

    let mutable testLoss = 0.0f
    let mutable correct = 0L
    let mutable batchCount = 0L

    for batch in dataLoader do

        use d = torch.NewDisposeScope()

        let input = batch.["data"]
        let labels = batch.["label"]

        let estimate = input --> model
        let output = loss estimate labels

        testLoss <- testLoss + output.ToSingle()
        batchCount <- batchCount + 1L

        use sum = estimate.argmax(1L).eq(labels).sum()
        correct <- correct + sum.ToInt64()

    let avgLossString = (testLoss / (float32 batchCount)).ToString("0.0000")
    let accString = ((float32 correct) / sz).ToString("0.0000")

    printfn $"\rTest set: Average loss {avgLossString} | Accuracy {accString}"


let trainingLoop (model:Model) epochs trainData testData =

    use trainLoader = new DataLoader(trainData, trainBatchSize, device=device, shuffle=true)
    use testLoader = new DataLoader(testData, testBatchSize, device=device, shuffle=false)

    use optimizer = Adam(model.parameters(), 0.001)

    let sw = Stopwatch()
    sw.Start()

    for epoch = 1 to epochs do
        train model optimizer trainLoader epoch trainData.Count
        test model testLoader (int testData.Count)

    sw.Stop()

    printfn $"Elapsed time: {sw.Elapsed.TotalSeconds:F1} s."

let run epochs =

    if device.``type`` = DeviceType.CUDA then
        trainBatchSize <- trainBatchSize * 8
        testBatchSize <- testBatchSize * 8

    let epochs = if device.``type`` = DeviceType.CUDA then epochs * 4 else epochs

    printfn ""
    printfn $"\tRunning AlexNet with {dataset} on {device.``type``.ToString()} for {epochs} epochs"
    printfn ""

    let targetDir = Path.Combine(datasetPath, "test_data")

    getDataFiles datasetPath targetDir

    use trainData = torchvision.datasets.CIFAR10(datasetPath, true, true)
    use testData = torchvision.datasets.CIFAR10(datasetPath, false, true)

    use model = new Model("model", device)

    trainingLoop model epochs trainData testData

    ()