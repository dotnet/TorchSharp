module TorchSharp.Examples.MNIST

open System
open System.IO
open System.Diagnostics

open TorchSharp
open type TorchSharp.torch.nn
open type TorchSharp.torch.optim
open type TorchSharp.Scalar

open TorchSharp.Examples

// Simple MNIST Convolutional model.
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
// In each case, there are four .gz files to download. Place them in a folder and then point the '_dataLocation'
// constant below at the folder location.

let mutable trainBatchSize = 64
let mutable testBatchSize = 128

let logInterval = 100

let cmdArgs = Environment.GetCommandLineArgs()
let dataset = if cmdArgs.Length = 2 then cmdArgs.[1] else "mnist"

let datasetPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", dataset)

torch.random.manual_seed(1L) |> ignore

let hasCUDA = torch.cuda.is_available()

let device = if hasCUDA then torch.CUDA else torch.CPU

let getDataFiles sourceDir targetDir =

    if not (Directory.Exists(targetDir)) then
        Directory.CreateDirectory(targetDir) |> ignore
        Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-images-idx3-ubyte.gz"), targetDir)
        Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-labels-idx1-ubyte.gz"), targetDir)
        Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-images-idx3-ubyte.gz"), targetDir)
        Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-labels-idx1-ubyte.gz"), targetDir)

type Model(name,device:torch.Device) as this =
    inherit Module(name)

    let conv1 = Conv2d(1L, 32L, 3L)
    let conv2 = Conv2d(32L, 64L, 3L)
    let fc1 = Linear(9216L, 128L)
    let fc2 = Linear(128L, 10L)

    let pool1 = MaxPool2d(kernelSize=[|2L; 2L|])

    let relu = ReLU()

    let dropout1 = Dropout(0.25)
    let dropout2 = Dropout(0.5)
    let flatten = Flatten()

    let logsm = LogSoftmax(1L)

    do
        this.RegisterComponents()

        if device.``type`` = DeviceType.CUDA then
            this.``to``(device) |> ignore

    override _.forward(input) =
        input
        --> conv1 --> relu --> conv2 --> relu --> pool1 --> dropout1
        --> flatten
        --> fc1 --> relu --> dropout2 --> fc2
        --> logsm

let loss x y = functional.nll_loss(reduction=Reduction.Mean).Invoke(x,y)

let train (model:Model) (optimizer:Optimizer) (dataLoader: MNISTReader) epoch =
    model.Train()

    let size = dataLoader.Size
    let batchSize = dataLoader.BatchSize

    let mutable batchID = 1

    printfn $"Epoch: {epoch}..."

    for (input,labels) in dataLoader do
        optimizer.zero_grad()

        begin  // This is introduced in order to let a few tensors go out of scope before GC
            use estimate = input --> model
            use output = loss estimate labels

            output.backward()
            optimizer.step()

            if batchID % logInterval = 0 then
                printfn $"\rTrain: epoch {epoch} [{batchID * batchSize} / {size}] Loss: {output.ToSingle():F4}"

            batchID <- batchID + 1
        end

        GC.Collect()

let test (model:Model) (dataLoader:MNISTReader) =
    model.Eval()

    let sz = float32 dataLoader.Size

    let mutable testLoss = 0.0f
    let mutable correct = 0

    for (input,labels) in dataLoader do

        begin  // This is introduced in order to let a few tensors go out of scope before GC
            use estimate = input --> model
            use output = loss estimate labels
            testLoss <- testLoss + output.ToSingle()

            let pred = estimate.argmax(1L)
            correct <- correct + pred.eq(labels).sum().ToInt32()
        end

        GC.Collect()

    printfn $"Size: {sz}, Total: {sz}"
    printfn $"\rTest set: Average loss {(testLoss / sz):F4} | Accuracy {(float32 correct / sz):P2}"

let trainingLoop (model:Model) epochs dataset trainData testData =

    let epochs = if device.``type`` = DeviceType.CUDA then epochs * 4 else epochs

    let optimizer = Adam(model.parameters())
    lr_scheduler.StepLR(optimizer, 1u, 0.7, last_epoch=5) |> ignore

    let sw = Stopwatch()
    sw.Start()

    for epoch = 1 to epochs do
        train model optimizer trainData epoch
        test model testData
    
    sw.Stop()

    printfn $"Elapsed time: {sw.Elapsed.TotalSeconds:F1} s."
    printfn $"Saving model to '{dataset}'.model.bin"

    model.save(dataset + ".model.bin") |> ignore

let run epochs =
    printfn $"Running MNIST on {device.``type``.ToString()}"
    printfn $"Dataset: {dataset}"

    let targetDir = Path.Combine(datasetPath, "test_data")

    getDataFiles datasetPath targetDir

    if device.``type`` = DeviceType.CUDA then
        trainBatchSize <- trainBatchSize * 4
        testBatchSize <- testBatchSize * 4

    let normImage = torchvision.transforms.Normalize( [|0.1307|], [|0.3081|], device=device)
    use train = new MNISTReader(targetDir, "train", trainBatchSize, device=device, shuffle=true, transform=normImage)
    use test = new MNISTReader(targetDir, "t10k", testBatchSize, device=device, transform=normImage)

    let model = new Model("model", device)
    
    trainingLoop model epochs dataset train test
