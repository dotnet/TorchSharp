using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples
{

    /// <summary>
    /// This example is based on the PyTorch tutorial at:
    /// 
    /// https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    ///
    /// It relies on the AG_NEWS dataset, which can be downloaded in CSV form at:
    ///
    /// https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv
    ///
    /// </summary>
    public class TextClassification
    {
        private const long emsize = 200;

        private const long batch_size = 128;
        private const long eval_batch_size = 128;

        private const int epochs = 15;

        // This path assumes that you're running this on Windows.
        private readonly static string _dataLocation = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "AG_NEWS");
        static void Main(string[] args)

        {
            torch.random.manual_seed(1);

            var cwd = Environment.CurrentDirectory;

            var device = torch.cuda.is_available() ? torch.device.CUDA : torch.device.CPU;
            Console.WriteLine($"Running TextClassification on {device.Type.ToString()}");

            using (var reader = TorchText.Data.AG_NEWSReader.AG_NEWS("train", device, _dataLocation)) {

                var dataloader = reader.Enumerate();

                var tokenizer = TorchText.Data.Utils.get_tokenizer("basic_english");

                var counter = new TorchText.Vocab.Counter<string>();
                foreach (var (label, text) in dataloader) {
                    counter.update(tokenizer(text));
                }

                var vocab = new TorchText.Vocab.Vocab(counter);

                var model = new TextClassificationModel(vocab.Count, emsize, 4).to(device);

                var loss = cross_entropy_loss();
                var lr = 5.0;
                var optimizer = torch.optim.Optimizer.SGD(model.parameters(), lr);
                var scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.2, last_epoch: 5);

                foreach (var epoch in Enumerable.Range(1, epochs)) {

                    var sw = new Stopwatch();
                    sw.Start();

                    train(epoch, reader.GetBatches(tokenizer, vocab, batch_size), model, loss, optimizer);

                    sw.Stop();

                    Console.WriteLine($"\nEnd of epoch: {epoch} | lr: {scheduler.LearningRate:0.0000} | time: {sw.Elapsed.TotalSeconds:0.0}s\n");
                    scheduler.step();
                }

                using (var test_reader = TorchText.Data.AG_NEWSReader.AG_NEWS("test", device, _dataLocation)) {

                    var sw = new Stopwatch();
                    sw.Start();

                    var accuracy = evaluate(test_reader.GetBatches(tokenizer, vocab, eval_batch_size), model, loss);

                    sw.Stop();

                    Console.WriteLine($"\nEnd of training: test accuracy: {accuracy:0.00} | eval time: {sw.Elapsed.TotalSeconds:0.0}s\n");
                    scheduler.step();
                }
            }

        }

        static void train(int epoch, IEnumerable<(Tensor, Tensor, Tensor)> train_data, TextClassificationModel model, Loss criterion, torch.optim.Optimizer optimizer)
        {
            model.Train();

            double total_acc = 0.0;
            long total_count = 0;
            long log_interval = 250;

            var batch = 0;

            var batch_count = train_data.Count();

            foreach (var (labels, texts, offsets) in train_data) {

                optimizer.zero_grad();
                var predicted_labels = model.forward(texts, offsets);

                var loss = criterion(predicted_labels, labels);
                loss.backward();
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5);
                optimizer.step();

                total_acc += (predicted_labels.argmax(1) == labels).sum().to(device.CPU).DataItem<long>();
                total_count += labels.size(0);

                if (batch % log_interval == 0 && batch > 0) {
                    var accuracy = total_acc / total_count;
                    Console.WriteLine($"epoch: {epoch} | batch: {batch} / {batch_count} | accuracy: {accuracy:0.00}");
                }
                batch += 1;
            }

            // This data set is small enough that we can get away with
            // collecting memory only once per epoch.

            GC.Collect();
        }

        static double evaluate(IEnumerable<(Tensor, Tensor, Tensor)> test_data, TextClassificationModel model, Loss criterion)
        {
            model.Eval();

            double total_acc = 0.0;
            long total_count = 0;

            foreach (var (labels, texts, offsets) in test_data) {

                var predicted_labels = model.forward(texts, offsets);
                var loss = criterion(predicted_labels, labels);

                total_acc += (predicted_labels.argmax(1) == labels).sum().to(device.CPU).DataItem<long>();
                total_count += labels.size(0);
            }

            return total_acc / total_count;
        }
    }

    class TextClassificationModel : CustomModule
    {
        private impl.EmbeddingBag embedding;
        private impl.Linear fc;

        public TextClassificationModel(long vocab_size, long embed_dim, long num_class) : base("TextClassification")
        {
            embedding = EmbeddingBag(vocab_size, embed_dim, sparse: false);
            fc = Linear(embed_dim, num_class);
            InitWeights();

            RegisterComponents();
        }

        private void InitWeights()
        {
            var initrange = 0.5;

            init.uniform(embedding.Weight, -initrange, initrange);
            init.uniform(fc.Weight, -initrange, initrange);
            init.zeros(fc.Bias);
        }

        public override Tensor forward(Tensor t)
        {
            throw new NotImplementedException();
        }

        public Tensor forward(Tensor input, Tensor offsets)
        {
            return fc.forward(embedding.forward(input, offsets));
        }

        public new TextClassificationModel to(device device)
        {
            base.to(device);
            return this;
        }
    }
}
