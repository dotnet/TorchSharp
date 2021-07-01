using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using TorchSharp.Tensor;
using static TorchSharp.torch;

namespace TorchText.Data
{
    public class AG_NEWSReader : IDisposable
    {
        public static AG_NEWSReader AG_NEWS(string split, device device, string root = ".data")
        {
            var dataPath = Path.Combine(root, $"{split}.csv");
            return new AG_NEWSReader(dataPath, device);
        }

        private AG_NEWSReader(string path, device device)
        {
            _path = path;
            _device = device;
        }

        private string _path;
        private device _device;

        public IEnumerable<(int, string)> Enumerate()
        {
            return File.ReadLines(_path).Select(line => ParseLine(line));
        }

        public IEnumerable<(TorchTensor, TorchTensor, TorchTensor)> GetBatches(Func<string, IEnumerable<string>> tokenizer, Vocab.Vocab vocab, long batch_size)
        {
            // This data set fits in memory, so we will simply load it all and cache it between epochs.

            var inputs = new List<(int, string)>();

            if (_data == null) {

                _data = new List<(TorchTensor, TorchTensor, TorchTensor)>();

                var counter = 0;
                var lines = Enumerate().ToList();
                var left = lines.Count;

                foreach (var line in lines) {

                    inputs.Add(line);
                    left -= 1;

                    if (++counter == batch_size || left == 0) {
                        _data.Add(Batchifier(inputs, tokenizer, vocab));
                        inputs.Clear();
                        counter = 0;
                    }
                }
            }

            return _data;
        }

        private List<(TorchTensor, TorchTensor, TorchTensor)> _data;
        private bool disposedValue;

        private (TorchTensor, TorchTensor, TorchTensor) Batchifier(IEnumerable<(int, string)> input, Func<string, IEnumerable<string>> tokenizer, Vocab.Vocab vocab)
        {
            var label_list = new List<long>();
            var text_list = new List<TorchTensor>();
            var offsets = new List<long>();
            offsets.Add(0);

            long last = 0;

            foreach (var (label, text) in input) {
                label_list.Add(label);
                var processed_text = Int64Tensor.from(tokenizer(text).Select(t => (long)vocab[t]).ToArray());
                text_list.Add(processed_text);
                last += processed_text.size(0);
                offsets.Add(last);
            }

            var labels = Int64Tensor.from(label_list.ToArray()).to(_device);
            var texts = text_list.ToArray().cat(0).to(_device);
            var offs = Int64Tensor.from(offsets.Take(label_list.Count).ToArray()).to(_device);

            return (labels, texts, offs);
        }

        public (int, string) ParseLine(string line)
        {
            int label = 0;
            string text = "";

            int firstComma = line.IndexOf("\",\"");
            label = int.Parse(line.Substring(1, firstComma - 1));
            text = line.Substring(firstComma + 2, line.Length - firstComma - 2);
            int secondComma = text.IndexOf("\",\"");
            text = text.Substring(secondComma + 2, text.Length - secondComma - 2);
            int thirdComma = text.IndexOf("\",\"");

            text = text.Substring(thirdComma + 2, text.Length - thirdComma - 3);

            return (label-1, text);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue) {
                if (disposing && _data != null) {
                    foreach (var (l, t, o) in _data) {
                        l.Dispose();
                        t.Dispose();
                        o.Dispose();
                    }
                }

                disposedValue = true;
            }
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
