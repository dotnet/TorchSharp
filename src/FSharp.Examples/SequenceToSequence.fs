module TorchSharp.Examples.SequenceToSequence

open System
open System.IO
open System.Linq
open System.Diagnostics
open System.Collections.Generic

open TorchSharp
open TorchSharp.Tensor
open TorchSharp.NN

open type TorchSharp.NN.Modules

open TorchSharp.Examples

// This example is based on the PyTorch tutorial at:
// 
// https://pytorch.org/tutorials/beginner/transformer_tutorial.html
//
// It relies on the WikiText2 dataset, which can be downloaded at:
//
// https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
//

let emsize = 200L
let nhidden = 200L
let nlayers = 2L
let nheads = 2L
let dropout = 0.2
let bptt = 32L

let batch_size = 64L
let eval_batch_size = 256L

let epochs = 50

let logInterval = 200

let cmdArgs = Environment.GetCommandLineArgs()

let datasetPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "wikitext-2-v1")

Torch.SetSeed(1L)

let hasCUDA = Torch.IsCudaAvailable()

let device = if hasCUDA then Device.CUDA else Device.CPU

let criterion x y = Functions.cross_entropy_loss(reduction=Reduction.Mean).Invoke(x,y)

type PositionalEncoding(dmodel, maxLen) as this =
    inherit CustomModule("PositionalEncoding")

    let dropout = Dropout(dropout)
    let mutable pe = Float32Tensor.zeros([| maxLen; dmodel|])

    do
        let position = Float32Tensor.arange(0L.ToScalar(), maxLen.ToScalar(), 1L.ToScalar()).unsqueeze(1L)
        let divTerm = (Float32Tensor.arange(0L.ToScalar(), dmodel.ToScalar(), 2L.ToScalar()) * (-Math.Log(10000.0) / (float dmodel)).ToScalar()).exp()

        let NULL = System.Nullable<int64>()

        // See: https://github.com/dotnet/fsharp/issues/9369 -- for now we have to use an explicit array within the index
        //
        pe.[ [| TorchTensorIndex.Ellipsis; TorchTensorIndex.Slice(0L, NULL, 2L) |] ] <- (position * divTerm).sin()
        pe.[ [| TorchTensorIndex.Ellipsis; TorchTensorIndex.Slice(1L, NULL, 2L) |] ] <- (position * divTerm).cos()

        pe <- pe.unsqueeze(0L).transpose(0L,1L)

        this.RegisterComponents()

    override _.forward(t) =
        let NULL = System.Nullable<int64>()
        use x = t + pe.[TorchTensorIndex.Slice(NULL, t.shape.[0]), TorchTensorIndex.Slice()]
        dropout.forward(x)

type TransformerModel(ntokens, device:Device) as this =
    inherit CustomModule("Transformer")

    let pos_encoder = new PositionalEncoding(emsize, 5000L)
    let encoder_layers = TransformerEncoderLayer(emsize, nheads, nhidden, dropout)
    let transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    let encoder = Embedding(ntokens, emsize)
    let decoder = Linear(emsize, ntokens)

    let sqrEmSz = MathF.Sqrt(float32 emsize).ToScalar()

    do
        let initrange = 0.1

        Init.uniform(encoder.Weight, -initrange, initrange) |> ignore
        Init.zeros(decoder.Bias) |> ignore
        Init.uniform(decoder.Weight, -initrange, initrange) |> ignore

        this.RegisterComponents()

        if device.Type = DeviceType.CUDA then
            this.``to``(device) |> ignore

    override _.forward(input) = raise (NotImplementedException("single-argument forward()"))

    member _.forward(t, mask) =
        let src = pos_encoder.forward(encoder.forward(t) * sqrEmSz)
        let enc = transformer_encoder.forward(src, mask)
        decoder.forward(enc)

    member _.GenerateSquareSubsequentMask(size:int64) =
        use mask = Float32Tensor.ones([|size;size|]).eq(Float32Tensor.from(1.0f)).triu().transpose(0L,1L)
        use maskIsZero = mask.eq(Float32Tensor.from(0.0f))
        use maskIsOne = mask.eq(Float32Tensor.from(1.0f))
        mask.to_type(ScalarType.Float32)
            .masked_fill(maskIsZero, Single.NegativeInfinity.ToScalar())
            .masked_fill(maskIsOne, 0.0f.ToScalar()).``to``(device)

let process_input (iter:string seq) (tokenizer:string->string seq) (vocab:TorchText.Vocab.Vocab) =
    [|
        for item in iter do
            let itemData = [| for token in tokenizer(item) do (int64 vocab.[token]) |]
            let t = Int64Tensor.from(itemData)
            if t.NumberOfElements > 0L then
                t
    |].cat(0L)
    
let batchify (data:TorchTensor) batchSize (device:Device) =
    let nbatch = data.shape.[0] / batchSize
    let d2 = data.narrow(0L, 0L, nbatch * batchSize).view(batchSize, -1L).t()
    d2.contiguous().``to``(device)

let get_batch (source:TorchTensor) (index:int64) =

    let len = min bptt (source.shape.[0]-1L-index)
    let data = source.[TorchTensorIndex.Slice(index, index + len)]
    let target = source.[TorchTensorIndex.Slice(index + 1L, index + 1L + len)].reshape(-1L)
    data,target

let train epoch (model:TransformerModel) (optimizer:Optimizer) (trainData:TorchTensor) ntokens =

    model.Train()

    let mutable total_loss = 0.0f
    let mutable src_mask = model.GenerateSquareSubsequentMask(bptt)

    let mutable batch = 0

    let tdlen = trainData.shape.[0]

    let mutable i = 0L

    while i < tdlen - 2L  do

        begin
            let data,targets = get_batch trainData i
            use data = data
            use targets = targets

            if data.shape.[0] <> bptt then
                src_mask.Dispose()
                src_mask <- model.GenerateSquareSubsequentMask(data.shape.[0])

            optimizer.zero_grad()

            use output = model.forward(data, src_mask)
            use loss = criterion (output.view(-1L, ntokens)) targets
            loss.backward()
            model.parameters().clip_grad_norm(0.5) |> ignore
            optimizer.step()

            total_loss <- total_loss + loss.cpu().DataItem<float32>()
        end 

        GC.Collect()

        if (batch % logInterval = 0) && (batch > 0) then
            let cur_loss = (total_loss / (float32 logInterval)).ToString("0.00")
            printfn $"epoch: {epoch} | batch: {batch} / {tdlen/bptt} | loss: {cur_loss}"
            total_loss <- 0.0f

        batch <- batch + 1
        i <- i + bptt


let evaluate (model:TransformerModel) (evalData:TorchTensor) ntokens =

    model.Eval()

    let mutable total_loss = 0.0f
    let mutable src_mask = model.GenerateSquareSubsequentMask(bptt)

    let mutable batch = 0L

    let tdlen = evalData.shape.[0]

    let mutable i = 0L

    while i < tdlen - 2L  do

        begin
            let data,targets = get_batch evalData i
            use data = data
            use targets = targets

            if data.shape.[0] <> bptt then
                src_mask.Dispose()
                src_mask <- model.GenerateSquareSubsequentMask(data.shape.[0])

            use output = model.forward(data, src_mask)
            use loss = criterion (output.view(-1L, ntokens)) targets
            total_loss <- total_loss + (float32 data.shape.[0]) * loss.cpu().DataItem<float32>()
        end 

        GC.Collect()

        batch <- batch + 1L
        i <- i + bptt

    total_loss / (float32 evalData.shape.[0])

let run epochs =

    printfn $"Running SequenceToSequence on {device.Type.ToString()} for {epochs} epochs."

    let vocabIter = TorchText.Datasets.WikiText2("train", datasetPath)
    let tokenizer = TorchText.Data.Utils.get_tokenizer("basic_english")
    let tokenizer str = tokenizer.Invoke(str)

    let counter = new TorchText.Vocab.Counter<string>()

    for item in vocabIter do
        counter.update(tokenizer(item))

    let vocab = TorchText.Vocab.Vocab(counter)

    let trainIter,validIter,testIter = TorchText.Datasets.WikiText2(datasetPath).ToTuple()

    let train_data = batchify (process_input trainIter tokenizer vocab) batch_size device
    let valid_data = batchify (process_input validIter tokenizer vocab) eval_batch_size device
    let test_data = batchify (process_input testIter tokenizer vocab) eval_batch_size device

    let ntokens = int64 vocab.Count

    use model = new TransformerModel(ntokens, device)
    let lr = 2.50
    let optimizer = NN.Optimizer.SGD(model.parameters(), lr)
    let scheduler = NN.Optimizer.StepLR(optimizer, (uint32 1), 0.95, last_epoch=15)

    let totalTime = Stopwatch()
    totalTime.Start()


    for epoch = 1 to epochs do
        let sw = Stopwatch()
        sw.Start()

        train epoch model optimizer train_data ntokens

        let val_loss = evaluate model valid_data ntokens
        sw.Stop()

        let lrStr = scheduler.LearningRate.ToString("0.00")
        let elapsed = sw.Elapsed.TotalSeconds.ToString("0.0")
        let lossStr = val_loss.ToString("0.00")

        printfn $"\nEnd of epoch: {epoch} | lr: {lrStr} | time: {elapsed}s | loss: {lossStr}\n"

        scheduler.step()

    let tst_loss = evaluate model test_data ntokens

    totalTime.Stop()

    let elapsed = totalTime.Elapsed.TotalSeconds.ToString("0.0")
    let lossStr = tst_loss.ToString("0.00")
    printfn $"\nEnd of training | time: {elapsed} s | loss: {lossStr}\n"