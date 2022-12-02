// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
module TorchSharp.Examples.TextClassification

open System
open System.IO
open System.Linq
open System.Diagnostics
open System.Collections.Generic

open TorchSharp
open type TorchSharp.torch.nn

// This example is based on the PyTorch tutorial at:
//
// https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
//
// It relies on the AG_NEWS dataset, which can be downloaded in CSV form at:
//
// https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv
//
// Download the two files, and place them in a folder called "AG_NEWS" in
// accordance with the file path below (Windows only).

let emsize = 200L

let batch_size = 64L
let eval_batch_size = 256L

let epochs = 16

let lr = 5.0

let logInterval = 200

let cmdArgs = Environment.GetCommandLineArgs()

let datasetPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "AG_NEWS")

torch.random.manual_seed(1L) |> ignore

let hasCUDA = torch.cuda_is_available() //torch.cuda.is_available() 

let device = if hasCUDA then torch.CUDA else torch.CPU

let loss = torch.nn.CrossEntropyLoss()
let criterion x y = loss.call(x,y)

type TextClassificationModel(vocabSize, embedDim, nClasses, device:torch.Device) as this =
    inherit Module<torch.Tensor, torch.Tensor, torch.Tensor>("Transformer")

    let embedding = EmbeddingBag(vocabSize, embedDim, sparse=false)
    let fc = Linear(embedDim, nClasses)

    do
        let initrange = 0.5

        init.uniform_(embedding.weight, -initrange, initrange) |> ignore
        init.uniform_(fc.weight, -initrange, initrange) |> ignore
        init.zeros_(fc.bias) |> ignore

        this.RegisterComponents()

        if device.``type`` = DeviceType.CUDA then
            this.``to``(device) |> ignore

    override _.forward(input, offsets) =
        embedding.call(input, offsets) --> fc

let train epoch (trainData:IEnumerable<torch.Tensor*torch.Tensor*torch.Tensor>) (model:TextClassificationModel) (optimizer:torch.optim.Optimizer) =

    model.train()

    let mutable total_acc = 0.0
    let mutable total_count = 0L
    let mutable batch = 0

    let batch_count = trainData.Count()

    for labels,texts,offsets in trainData do

        use d = torch.NewDisposeScope()

        optimizer.zero_grad()

        let predicted_labels = model.call(texts, offsets)
        let loss = criterion predicted_labels labels

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) |> ignore
        optimizer.step() |> ignore

        total_acc <- total_acc + float ((predicted_labels.argmax(1L).eq(labels)).sum().cpu().item<int64>())
        total_count <- total_count + labels.size(0)

        if (batch % logInterval = 0) && (batch > 0) then
            let accuracy = (total_acc / (float total_count)).ToString("0.00")
            printfn $"epoch: {epoch} | batch: {batch} / {batch_count} | accuracy: {accuracy}"

        batch <- batch + 1

let evaluate (testData:IEnumerable<torch.Tensor*torch.Tensor*torch.Tensor>) (model:TextClassificationModel) =

    model.eval()

    let mutable total_acc = 0.0
    let mutable total_count = 0L

    for labels,texts,offsets in testData do

        use d = torch.NewDisposeScope()

        let predicted_labels = model.call(texts, offsets)
        let loss = criterion predicted_labels labels

        total_acc <- total_acc + float ((predicted_labels.argmax(1L).eq(labels)).sum().cpu().item<int64>())
        total_count <- total_count + labels.size(0)

    total_acc / (float total_count)

let run epochs =

    printfn $"Running TextClassification on {device.``type``.ToString()} for {epochs} epochs."

    use reader = TorchText.Data.AG_NEWSReader.AG_NEWS("train", device, datasetPath)
    let dataloader = reader.Enumerate()

    let tokenizer = TorchText.Data.Utils.get_tokenizer("basic_english")
    let counter = new TorchText.Vocab.Counter<string>()

    for label,text in dataloader do
        counter.update(tokenizer.Invoke(text))

    let vocab = TorchText.Vocab.Vocab(counter)

    let model = new TextClassificationModel((int64 vocab.Count), emsize, 4L, device)

    let optimizer = torch.optim.SGD(model.parameters(), lr)
    let scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.2, last_epoch=5)

    let sw = Stopwatch()

    for epoch = 1 to epochs do

        sw.Restart()

        let batches = [| for b in reader.GetBatches(tokenizer, vocab, batch_size) -> b.ToTuple() |]
        train epoch batches model optimizer

        sw.Stop()

        let pgFirst = optimizer.ParamGroups.First()
        let lrStr = pgFirst.LearningRate.ToString("0.0000")
        let tsStr = sw.Elapsed.TotalSeconds.ToString("0.0")
        printfn $"\nEnd of epoch: {epoch} | lr: {lrStr} | time: {tsStr}s\n"
        scheduler.step() |> ignore

    use test_reader = TorchText.Data.AG_NEWSReader.AG_NEWS("test", device, datasetPath)

    sw.Restart()

    let batches = [| for b in test_reader.GetBatches(tokenizer, vocab, batch_size) -> b.ToTuple() |]
    let accuracy = evaluate batches model

    let accStr = accuracy.ToString("0.00")
    let tsStr = sw.Elapsed.TotalSeconds.ToString("0.0")
    printf $"\nEnd of training: test accuracy: {accStr} | eval time: {tsStr}s\n"

    sw.Stop()
