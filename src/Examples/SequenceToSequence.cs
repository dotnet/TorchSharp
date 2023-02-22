// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{

    /// <summary>
    /// This example is based on the PyTorch tutorial at:
    ///
    /// https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    ///
    /// It relies on the WikiText2 dataset, which can be downloaded at:
    ///
    /// https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
    ///
    /// After downloading, extract the files using the defaults (Windows only).
    /// </summary>
    public class SequenceToSequence
    {
        // This path assumes that you're running this on Windows.
#if NET472_OR_GREATER
        private readonly static string _dataLocation = NSPath.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "wikitext-2-v1");
#else
        private readonly static string _dataLocation = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "wikitext-2-v1");
#endif // NET472_OR_GREATER

        private const long emsize = 200;
        private const long nhid = 200;
        private const long nlayers = 2;
        private const long nhead = 2;
        private const double dropout = 0.2;

        private const int batch_size = 64;
        private const int eval_batch_size = 32;

        private const int epochs = 10;

        internal static void Main(string[] args)

        {
            torch.random.manual_seed(1);

            var cwd = Environment.CurrentDirectory;

            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Running SequenceToSequence on {device.type.ToString()} for {epochs} epochs.");

            var vocab_iter = TorchText.Datasets.WikiText2("train", _dataLocation);
            var tokenizer = TorchText.Data.Utils.get_tokenizer("basic_english");

            var counter = new TorchText.Vocab.Counter<string>();
            foreach (var item in vocab_iter) {
                counter.update(tokenizer(item));
            }

            var vocab = new TorchText.Vocab.Vocab(counter);

            var (train_iter, valid_iter, test_iter) = TorchText.Datasets.WikiText2(_dataLocation);

            var train_data = Batchify(ProcessInput(train_iter, tokenizer, vocab), batch_size).to((Device)device);
            var valid_data = Batchify(ProcessInput(valid_iter, tokenizer, vocab), eval_batch_size).to((Device)device);
            var test_data = Batchify(ProcessInput(test_iter, tokenizer, vocab), eval_batch_size).to((Device)device);

            var bptt = 32;

            var ntokens = vocab.Count;

            var model = new TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to((Device)device);
            var loss = CrossEntropyLoss();
            var lr = 2.50;
            var optimizer = torch.optim.SGD(model.parameters(), lr);
            var scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.95, last_epoch: 15);

            var totalTime = new Stopwatch();
            totalTime.Start();

            foreach (var epoch in Enumerable.Range(1, epochs)) {

                var sw = new Stopwatch();
                sw.Start();

                train(epoch, train_data, model, loss, bptt, ntokens, optimizer);

                var val_loss = evaluate(valid_data, model, loss, bptt, ntokens, optimizer);
                sw.Stop();

                var pgFirst = optimizer.ParamGroups.First();
                Console.WriteLine($"\nEnd of epoch: {epoch} | lr: {pgFirst.LearningRate:0.00} | time: {sw.Elapsed.TotalSeconds:0.0}s | loss: {val_loss:0.00}\n");
                scheduler.step();
            }

            var tst_loss = evaluate(test_data, model, loss, bptt, ntokens, optimizer);
            totalTime.Stop();

            Console.WriteLine($"\nEnd of training | time: {totalTime.Elapsed.TotalSeconds:0.0}s | loss: {tst_loss:0.00}\n");
        }

        private static void train(int epoch, Tensor train_data, TransformerModel model, Loss<Tensor, Tensor, Tensor> criterion, int bptt, int ntokens, torch.optim.Optimizer optimizer)
        {
            model.train();

            using (var d = torch.NewDisposeScope()) {

                var total_loss = 0.0f;

                var batch = 0;
                var log_interval = 200;

                var src_mask = model.GenerateSquareSubsequentMask(bptt);

                var tdlen = train_data.shape[0];


                for (int i = 0; i < tdlen - 1; batch++, i += bptt) {

                    var (data, targets) = GetBatch(train_data, i, bptt);
                    optimizer.zero_grad();

                    if (data.shape[0] != bptt) {
                        src_mask = model.GenerateSquareSubsequentMask(data.shape[0]);
                    }

                    using (var output = model.call(data, src_mask)) {
                        var loss = criterion.call(output.view(-1, ntokens), targets);
                        loss.backward();
                        torch.nn.utils.clip_grad_norm_(model.parameters().ToArray(), 0.5);
                        optimizer.step();

                        total_loss += loss.to(torch.CPU).item<float>();
                    }

                    if ((batch % log_interval == 0 && batch > 0) || (batch == tdlen / bptt)) {
                        var cur_loss = total_loss / log_interval;
                        Console.WriteLine($"epoch: {epoch} | batch: {batch} / {tdlen / bptt} | loss: {cur_loss:0.00}");
                        total_loss = 0;
                    }

                    d.DisposeEverythingBut(src_mask);
                }
            }
        }

        private static double evaluate(Tensor eval_data, TransformerModel model, Loss<Tensor, Tensor, Tensor> criterion, int bptt, int ntokens, torch.optim.Optimizer optimizer)
        {
            model.eval();

            using (var d = torch.NewDisposeScope()) {

                var src_mask = model.GenerateSquareSubsequentMask(bptt);

                var total_loss = 0.0f;
                var batch = 0;


                for (int i = 0; i < eval_data.shape[0] - 1; batch++, i += bptt) {

                    var (data, targets) = GetBatch(eval_data, i, bptt);
                    if (data.shape[0] != bptt) {
                        src_mask = model.GenerateSquareSubsequentMask(data.shape[0]);
                    }
                    using (var output = model.call(data, src_mask)) {
                        var loss = criterion.call(output.view(-1, ntokens), targets);
                        total_loss += data.shape[0] * loss.to(torch.CPU).item<float>();
                    }

                    data.Dispose();
                    targets.Dispose();

                    d.DisposeEverythingBut(src_mask);
                }

                return total_loss / eval_data.shape[0];
            }
        }

        static Tensor ProcessInput(IEnumerable<string> iter, Func<string, IEnumerable<string>> tokenizer, TorchText.Vocab.Vocab vocab)
        {
            List<Tensor> data = new List<Tensor>();
            foreach (var item in iter) {
                List<long> itemData = new List<long>();
                foreach (var token in tokenizer(item)) {
                    itemData.Add(vocab[token]);
                }
                data.Add(torch.tensor(itemData.ToArray(), torch.int64));
            }

            var result = torch.cat(data.Where(t => t.NumberOfElements > 0).ToList(), 0);
            return result;
        }

        static Tensor Batchify(Tensor data, int batch_size)
        {
            var nbatch = data.shape[0] / batch_size;
            return data.narrow(0, 0, nbatch * batch_size).view(batch_size, -1).t().contiguous();
        }

        static (Tensor, Tensor) GetBatch(Tensor source, int index, int bptt)
        {
            var len = Math.Min(bptt, source.shape[0] - 1 - index);
            var data = source[TensorIndex.Slice(index, index + len)];
            var target = source[TensorIndex.Slice(index + 1, index + 1 + len)].reshape(-1);
            return (data, target);
        }

        class TransformerModel : Module<Tensor, Tensor, Tensor>
        {
            private Modules.TransformerEncoder transformer_encoder;
            private PositionalEncoding pos_encoder;
            private Modules.Embedding encoder;
            private Modules.Linear decoder;

            private long ninputs;
            private Device device;

            public TransformerModel(long ntokens, long ninputs, long nheads, long nhidden, long nlayers, double dropout = 0.5) : base("Transformer")
            {
                this.ninputs = ninputs;

                pos_encoder = new PositionalEncoding(ninputs, dropout);
                var encoder_layers = TransformerEncoderLayer(ninputs, nheads, nhidden, dropout);
                transformer_encoder = TransformerEncoder(encoder_layers, nlayers);
                encoder = Embedding(ntokens, ninputs);
                decoder = Linear(ninputs, ntokens);
                InitWeights();

                RegisterComponents();
            }

            public Tensor GenerateSquareSubsequentMask(long size)
            {
                var mask = (torch.ones(new long[] { size, size }) == 1).triu().transpose(0, 1);
                return mask.to_type(ScalarType.Float32)
                    .masked_fill(mask == 0, float.NegativeInfinity)
                    .masked_fill(mask == 1, 0.0f).to(device);
            }

            private void InitWeights()
            {
                var initrange = 0.1;

                init.uniform_(encoder.weight, -initrange, initrange);
                init.zeros_(decoder.bias);
                init.uniform_(decoder.weight, -initrange, initrange);
            }

            public override Tensor forward(Tensor t, Tensor mask)
            {
                var src = pos_encoder.call(encoder.call(t) * MathF.Sqrt(ninputs));
                var enc = transformer_encoder.call(src, mask);
                return decoder.call(enc);
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    transformer_encoder.Dispose();
                    pos_encoder.Dispose();
                    encoder.Dispose();
                    decoder.Dispose();
                    ClearModules();
                }
                base.Dispose(disposing);
            }

            protected override Module _to(DeviceType deviceType, int deviceIndex = -1)
            {
                this.device = new Device(deviceType, deviceIndex);
                return base._to(deviceType, deviceIndex);
            }
        }

        class PositionalEncoding : Module<Tensor, Tensor>
        {
            private Module<Tensor, Tensor> dropout;
            private Tensor pe;

            public PositionalEncoding(long dmodel, double dropout, int maxLen = 5000) : base("PositionalEncoding")
            {
                this.dropout = Dropout(dropout);
                var pe = torch.zeros(new long[] { maxLen, dmodel });
                var position = torch.arange(0, maxLen, 1).unsqueeze(1);
                var divTerm = (torch.arange(0, dmodel, 2) * (-Math.Log(10000.0) / dmodel)).exp();
                pe[TensorIndex.Ellipsis, TensorIndex.Slice(0, null, 2)] = (position * divTerm).sin();
                pe[TensorIndex.Ellipsis, TensorIndex.Slice(1, null, 2)] = (position * divTerm).cos();
                this.pe = pe.unsqueeze(0).transpose(0, 1);

                RegisterComponents();
            }

            public override Tensor forward(Tensor t)
            {
                using var x = t + pe[TensorIndex.Slice(null, t.shape[0]), TensorIndex.Slice()];
                return dropout.call(x);
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    dropout.Dispose();
                    pe.Dispose();
                    ClearModules();
                }
                base.Dispose(disposing);
            }
        }
    }
}
