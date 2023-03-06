// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
module TorchSharp.Examples.SequenceToSequence

open System
open System.IO
open System.Linq
open System.Diagnostics
open System.Collections.Generic

open TorchSharp
open TorchSharp.Modules
open type TorchSharp.torch.nn
open type TorchSharp.torch.optim

open TorchSharp.Examples

// This example is based on the PyTorch tutorial at:
//
// https://pytorch.org/tutorials/beginner/transformer_tutorial.html
//
// It relies on the WikiText2 dataset, which can be downloaded at:
//
// https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
//
// After downloading, extract the files using the defaults (Windows only).
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

let datasetPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "wikitext-2-v1")

torch.random.manual_seed(1L) |> ignore

let hasCUDA = torch.cuda_is_available() //torch.cuda.is_available()

let device = if hasCUDA then torch.CUDA else torch.CPU

let loss = torch.nn.CrossEntropyLoss(reduction=Reduction.Mean)
let criterion x y = loss.call(x,y)

type PositionalEncoding(dmodel, maxLen) as this =
    inherit Module<torch.Tensor,torch.Tensor>("PositionalEncoding")

    let dropout = Dropout(dropout)
    let mutable pe = torch.zeros([| maxLen; dmodel|])

    do
        let position = torch.arange(0L.ToScalar(), maxLen.ToScalar(), 1L.ToScalar()).unsqueeze(1L)
        let divTerm = (torch.arange(0L.ToScalar(), dmodel.ToScalar(), 2L.ToScalar()) * (-Math.Log(10000.0) / (float dmodel)).ToScalar()).exp()

        let NULL = System.Nullable<int64>()

        // See: https://github.com/dotnet/fsharp/issues/9369 -- for now we have to use an explicit array within the index
        //
        pe.[ [| torch.TensorIndex.Ellipsis; torch.TensorIndex.Slice(0L, NULL, 2L) |] ] <- (position * divTerm).sin()
        pe.[ [| torch.TensorIndex.Ellipsis; torch.TensorIndex.Slice(1L, NULL, 2L) |] ] <- (position * divTerm).cos()

        pe <- pe.unsqueeze(0L).transpose(0L,1L)

        this.RegisterComponents()

    override _.forward(t) =
        let NULL = System.Nullable<int64>()
        use x = t + pe.[torch.TensorIndex.Slice(NULL, t.shape.[0]), torch.TensorIndex.Slice()]
        dropout.call(x)

type TransformerModel(ntokens, device:torch.Device) as this =
    inherit Module<torch.Tensor,torch.Tensor,torch.Tensor>("Transformer")

    let pos_encoder = new PositionalEncoding(emsize, 5000L)
    let encoder_layers = TransformerEncoderLayer(emsize, nheads, nhidden, dropout)
    let transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    let encoder = Embedding(ntokens, emsize)
    let decoder = Linear(emsize, ntokens)

    let sqrEmSz = Math.Sqrt(float emsize).ToScalar()

    do
        let initrange = 0.1

        init.uniform_(encoder.weight, -initrange, initrange) |> ignore
        init.zeros_(decoder.bias) |> ignore
        init.uniform_(decoder.weight, -initrange, initrange) |> ignore

        this.RegisterComponents()

        if device.``type`` = DeviceType.CUDA then
            this.``to``(device) |> ignore

    override _.forward(t, mask) =
        let src = pos_encoder.call(encoder.call(t) * sqrEmSz)
        let enc = transformer_encoder.call(src, mask)
        decoder.call(enc)

    member _.GenerateSquareSubsequentMask(size:int64) =
        use mask = torch.ones([|size;size|]).eq(torch.tensor(1.0f)).triu().transpose(0L,1L)
        use maskIsZero = mask.eq(torch.tensor(0.0f))
        use maskIsOne = mask.eq(torch.tensor(1.0f))
        mask.to_type(torch.float32)
            .masked_fill(maskIsZero, Single.NegativeInfinity.ToScalar())
            .masked_fill(maskIsOne, 0.0f.ToScalar()).``to``(device)

let process_input (iter:string seq) (tokenizer:string->string seq) (vocab:TorchText.Vocab.Vocab) =
    torch.cat(
        [|
            for item in iter do
                let itemData = [| for token in tokenizer(item) do (int64 vocab.[token]) |]
                let t = torch.tensor(itemData)
                if t.NumberOfElements > 0L then
                    t
        |], 0L)

let batchify (data:torch.Tensor) batchSize (device:torch.Device) =
    let nbatch = data.shape.[0] / batchSize
    let d2 = data.narrow(0L, 0L, nbatch * batchSize).view(batchSize, -1L).t()
    d2.contiguous().``to``(device)

let get_batch (source:torch.Tensor) (index:int64) =

    let len = min bptt (source.shape.[0]-1L-index)
    let data = source.[torch.TensorIndex.Slice(index, index + len)]
    let target = source.[torch.TensorIndex.Slice(index + 1L, index + 1L + len)].reshape(-1L)
    data,target

let train epoch (model:TransformerModel) (optimizer:Optimizer) (trainData:torch.Tensor) ntokens =

    model.train()

    use d = torch.NewDisposeScope()

    let mutable total_loss = 0.0f
    let mutable src_mask = model.GenerateSquareSubsequentMask(bptt)

    let mutable batch = 0

    let tdlen = trainData.shape.[0]

    let mutable i = 0L

    while i < tdlen - 2L  do

        begin
            let data,targets = get_batch trainData i

            if data.shape.[0] <> bptt then
                src_mask <- model.GenerateSquareSubsequentMask(data.shape.[0])

            optimizer.zero_grad()

            let output = model.call(data, src_mask)
            let loss = criterion (output.view(-1L, ntokens)) targets
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) |> ignore
            optimizer.step() |> ignore

            total_loss <- total_loss + loss.cpu().item<float32>()
        end

        if (batch % logInterval = 0) && (batch > 0) then
            let cur_loss = (total_loss / (float32 logInterval)).ToString("0.00")
            printfn $"epoch: {epoch} | batch: {batch} / {tdlen/bptt} | loss: {cur_loss}"
            total_loss <- 0.0f

        batch <- batch + 1
        i <- i + bptt

        d.DisposeEverythingBut(src_mask) |> ignore


let evaluate (model:TransformerModel) (evalData:torch.Tensor) ntokens =

    model.eval()

    use d = torch.NewDisposeScope()

    let mutable total_loss = 0.0f
    let mutable src_mask = model.GenerateSquareSubsequentMask(bptt)

    let mutable batch = 0L

    let tdlen = evalData.shape.[0]

    let mutable i = 0L

    while i < tdlen - 2L  do

        begin
            let data,targets = get_batch evalData i

            if data.shape.[0] <> bptt then
                src_mask <- model.GenerateSquareSubsequentMask(data.shape.[0])

            let output = model.call(data, src_mask)
            let loss = criterion (output.view(-1L, ntokens)) targets
            total_loss <- total_loss + (float32 data.shape.[0]) * loss.cpu().item<float32>()
        end

        batch <- batch + 1L
        i <- i + bptt

        d.DisposeEverythingBut(src_mask) |> ignore

    src_mask.Dispose();

    total_loss / (float32 evalData.shape.[0])


let run epochs =

    printfn $"Running SequenceToSequence on {device.``type``.ToString()} for {epochs} epochs."

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

    let optimizer = SGD(model.parameters(), lr);
    let scheduler = lr_scheduler.StepLR(optimizer, 1, 0.95, last_epoch=15)

    let totalTime = Stopwatch()
    totalTime.Start()


    for epoch = 1 to epochs do
        let sw = Stopwatch()
        sw.Start()

        train epoch model optimizer train_data ntokens

        let val_loss = evaluate model valid_data ntokens
        sw.Stop()

        let pgFirst = optimizer.ParamGroups.First()
        let lrStr = pgFirst.LearningRate.ToString("0.00")
        let elapsed = sw.Elapsed.TotalSeconds.ToString("0.0")
        let lossStr = val_loss.ToString("0.00")

        printfn $"\nEnd of epoch: {epoch} | lr: {lrStr} | time: {elapsed}s | loss: {lossStr}\n"

        scheduler.step() |> ignore

    let tst_loss = evaluate model test_data ntokens

    totalTime.Stop()

    let elapsed = totalTime.Elapsed.TotalSeconds.ToString("0.0")
    let lossStr = tst_loss.ToString("0.00")
    printfn $"\nEnd of training | time: {elapsed} s | loss: {lossStr}\n"