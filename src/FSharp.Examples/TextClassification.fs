module TorchSharp.Examples.TextClassification

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
// https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
//
// It relies on the AG_NEWS dataset, which can be downloaded in CSV form at:
//
// https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv

let emsize = 200L

let batch_size = 64L
let eval_batch_size = 256L

let epochs = 16

let lr = 5.0

let logInterval = 200

let cmdArgs = Environment.GetCommandLineArgs()

let datasetPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "AG_NEWS")

torch.random.manual_seed(1L) |> ignore

let hasCUDA = torch.cuda.is_available()

let device = if hasCUDA then Device.CUDA else Device.CPU

let criterion x y = Functions.cross_entropy_loss().Invoke(x,y)

type TextClassificationModel(vocabSize, embedDim, nClasses, device:Device) as this =
    inherit CustomModule("Transformer")

    let embedding = EmbeddingBag(vocabSize, embedDim, sparse=false)
    let fc = Linear(embedDim, nClasses)

    do
        let initrange = 0.5

        Init.uniform(embedding.Weight, -initrange, initrange) |> ignore
        Init.uniform(fc.Weight, -initrange, initrange) |> ignore
        Init.zeros(fc.Bias) |> ignore

        this.RegisterComponents()

        if device.Type = DeviceType.CUDA then
            this.``to``(device) |> ignore

    override _.forward(input) = raise (NotImplementedException("single-argument forward()"))

    member _.forward(input, offsets) =
        embedding.forward(input, offsets) --> fc

let train epoch (trainData:IEnumerable<TorchTensor*TorchTensor*TorchTensor>) (model:TextClassificationModel) (optimizer:Optimizer) =

    model.Train()

    let mutable total_acc = 0.0
    let mutable total_count = 0L
    let mutable batch = 0

    let batch_count = trainData.Count()

    for labels,texts,offsets in trainData do

        optimizer.zero_grad()

        let predicted_labels = model.forward(texts, offsets)
        let loss = criterion predicted_labels labels

        loss.backward()
        model.parameters().clip_grad_norm(0.5) |> ignore
        optimizer.step()

        total_acc <- total_acc + float ((predicted_labels.argmax(1L).eq(labels)).sum().cpu().DataItem<int64>())
        total_count <- total_count + labels.size(0)

        if (batch % logInterval = 0) && (batch > 0) then
            let accuracy = (total_acc / (float total_count)).ToString("0.00")
            printfn $"epoch: {epoch} | batch: {batch} / {batch_count} | accuracy: {accuracy}"

        batch <- batch + 1

    GC.Collect()

let evaluate (testData:IEnumerable<TorchTensor*TorchTensor*TorchTensor>) (model:TextClassificationModel) =

    model.Eval()

    let mutable total_acc = 0.0
    let mutable total_count = 0L

    for labels,texts,offsets in testData do

        let predicted_labels = model.forward(texts, offsets)
        let loss = criterion predicted_labels labels

        total_acc <- total_acc + float ((predicted_labels.argmax(1L).eq(labels)).sum().cpu().DataItem<int64>())
        total_count <- total_count + labels.size(0)

    total_acc / (float total_count)

let run epochs =

    printfn $"Running TextClassification on {device.Type.ToString()} for {epochs} epochs."

    use reader = TorchText.Data.AG_NEWSReader.AG_NEWS("train", device, datasetPath)
    let dataloader = reader.Enumerate()

    let tokenizer = TorchText.Data.Utils.get_tokenizer("basic_english")
    let counter = new TorchText.Vocab.Counter<string>()

    for label,text in dataloader do
        counter.update(tokenizer.Invoke(text))

    let vocab = TorchText.Vocab.Vocab(counter)

    let model = new TextClassificationModel((int64 vocab.Count), emsize, 4L, device)

    let optimizer = NN.Optimizer.SGD(model.parameters(), lr)
    let scheduler = NN.Optimizer.StepLR(optimizer, 1u, 0.2, last_epoch=5)

    let sw = Stopwatch()

    for epoch = 1 to epochs do

        sw.Restart()

        let batches = [| for b in reader.GetBatches(tokenizer, vocab, batch_size) -> b.ToTuple() |]
        train epoch batches model optimizer

        sw.Stop()

        let lrStr = scheduler.LearningRate.ToString("0.0000")
        let tsStr = sw.Elapsed.TotalSeconds.ToString("0.0")
        printfn $"\nEnd of epoch: {epoch} | lr: {lrStr} | time: {tsStr}s\n"
        scheduler.step()

    use test_reader = TorchText.Data.AG_NEWSReader.AG_NEWS("test", device, datasetPath)

    sw.Restart()

    let batches = [| for b in test_reader.GetBatches(tokenizer, vocab, batch_size) -> b.ToTuple() |]
    let accuracy = evaluate batches model

    let accStr = accuracy.ToString("0.00")
    let tsStr = sw.Elapsed.TotalSeconds.ToString("0.0")
    printf $"\nEnd of training: test accuracy: {accStr} | eval time: {tsStr}s\n"

    sw.Stop()
