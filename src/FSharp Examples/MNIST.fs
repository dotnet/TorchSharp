module TorchSharp.Examples.MNIST

open System
open System.IO
open System.Diagnostics

open TorchSharp
open TorchSharp.Tensor
open TorchSharp.NN

open type TorchSharp.NN.Modules

open TorchSharp.Examples

/// <summary>
/// Simple MNIST Convolutional model.
/// </summary>
/// <remarks>
/// There are at least two interesting data sets to use with this example:
/// 
/// 1. The classic MNIST set of 60000 images of handwritten digits.
///
///     It is available at: http://yann.lecun.com/exdb/mnist/
///     
/// 2. The 'fashion-mnist' data set, which has the exact same file names and format as MNIST, but is a harder
///    data set to train on. It's just as large as MNIST, and has the same 60/10 split of training and test
///    data.
///    It is available at: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
///
/// In each case, there are four .gz files to download. Place them in a folder and then point the '_dataLocation'
/// constant below at the folder location.
/// </remarks>

let mutable trainBatchSize = 64
let mutable testBatchSize = 128

let logInterval = 100

let cmdArgs = Environment.GetCommandLineArgs()
let dataset = if cmdArgs.Length = 2 then cmdArgs.[1] else "mnist"

let datasetPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", dataset)

Torch.SetSeed(1L)

let hasCUDA = Torch.IsCudaAvailable()

let device = if hasCUDA then Device.CUDA else Device.CPU

type Model(name) as this =
    inherit CustomModule(name)

    let conv1 = Conv2d(1L, 32L, 3L)
    let conv2 = Conv2d(32L, 64L, 3L)
    let fc1 = Linear(9216L, 128L)
    let fc2 = Linear(128L, 10L)

    let pool1 = MaxPool2d(kernelSize=[|2L; 2L|])

    let relu = ReLU();

    let dropout1 = Dropout(0.25)
    let dropout2 = Dropout(0.5)
    let flatten = Flatten()

    let logsm = LogSoftmax(1L)

    do
        this.RegisterComponents()

        if device.Type = DeviceType.CUDA then
            this.``to``(device) |> ignore

    override _.forward(input) =
        input
        |> conv1.forward |> relu.forward
        |> conv2.forward |> relu.forward |> pool1.forward
        |> dropout1.forward
        |> flatten.forward
        |> fc1.forward |> relu.forward
        |> dropout2.forward
        |> fc2.forward |> logsm.forward

let loss x y = Functions.nll_loss(reduction=Reduction.Mean).Invoke(x,y)

let train (model: Model) (optimizer: NN.Optimizer) (dataLoader: MNISTReader) epoch batchSize size =
    model.Train()

    let mutable batchID = 1

    printfn $"Epoch: {epoch}..."

    for (X,y) in dataLoader do
        optimizer.zero_grad()

        use ŷ = model.forward(X)
        use output = loss ŷ y

        output.backward()
        optimizer.step()

        if batchID % logInterval = 0 then
            printfn $"\rTrain: epoch {epoch} [{batchID * batchSize} / {size}] Loss: {output.ToSingle():F4}"

        batchID <- batchID + 1

        GC.Collect()

let test (model:Model) (optimizer:NN.Optimizer) (dataLoader:MNISTReader) size =
    model.Eval()

    let sz = float32 size

    let mutable testLoss = 0.0f
    let mutable correct = 0

    for (X,y) in dataLoader do

        do  // This is introduced in order to let a few tensors go out of scope before GC
            use ŷ = model.forward(X)
            use output = loss ŷ y
            testLoss <- testLoss + output.ToSingle()

            let pred = ŷ.argmax(1L)
            correct <- correct + pred.eq(y).sum().ToInt32()

        GC.Collect()

    printfn $"Size: {sz}, Total: {sz}"
    printfn $"\rTest set: Average loss {(testLoss / sz):F4} | Accuracy {(float32 correct / sz):P2}"

let trainingLoop epochs dataset (model:Model) trainData (testData:MNISTReader) =
    let mutable epochs = epochs

    if device.Type = DeviceType.CUDA then
        epochs <- epochs * 4

    use optimizer = NN.Optimizer.Adam(model.parameters())

    let sw = Stopwatch()
    sw.Start()

    for epoch = 1 to epochs do
        train model optimizer trainData epoch trainData.BatchSize trainData.Size
        test model optimizer testData testData.Size
    
    sw.Stop()

    printfn $"Elapsed time: {sw.Elapsed.TotalSeconds:F1} s."
    printfn $"Saving model to '{dataset}'.model.bin"

let run epochs =
    printfn $"Running MNIST on {device.Type.ToString()}"
    printfn $"Dataset: {dataset}"

    let sourceDir = datasetPath
    let targetDir = Path.Combine(datasetPath, "test_data")

    if not (Directory.Exists(targetDir)) then
        Directory.CreateDirectory(targetDir) |> ignore
        Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-images-idx3-ubyte.gz"), targetDir)
        Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-labels-idx1-ubyte.gz"), targetDir)
        Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-images-idx3-ubyte.gz"), targetDir)
        Utils.Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-labels-idx1-ubyte.gz"), targetDir)

    if device.Type = DeviceType.CUDA then
        trainBatchSize <- trainBatchSize * 4
        testBatchSize <- testBatchSize * 4

    let normImage = TorchVision.Transforms.Normalize( [|0.1307|], [|0.3081|], device=device)
    use train = new MNISTReader(targetDir, "train", trainBatchSize, device=device, shuffle=true, transform=normImage)
    use test = new MNISTReader(targetDir, "t10k", testBatchSize, device=device, transform=normImage)

    use model = new Model("model")
    
    trainingLoop epochs dataset model train test

    model.save(dataset + ".model.bin")
