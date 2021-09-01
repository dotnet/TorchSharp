// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace TorchText.Data
{
    public class AG_NEWSReader : IDisposable
    {
        public static AG_NEWSReader AG_NEWS(string split, Device device, string root = ".data")
        {
            var dataPath = Path.Combine(root, $"{split}.csv");
            return new AG_NEWSReader(dataPath, device);
        }

        private AG_NEWSReader(string path, Device device)
        {
            _path = path;
            _device = device;
        }

        private string _path;
        private Device _device;

        public IEnumerable<(int, string)> Enumerate()
        {
            return File.ReadLines(_path).Select(line => ParseLine(line));
        }

        public IEnumerable<(Tensor, Tensor, Tensor)> GetBatches(Func<string, IEnumerable<string>> tokenizer, Vocab.Vocab vocab, long batch_size)
        {
            // This data set fits in memory, so we will simply load it all and cache it between epochs.

            var inputs = new List<(int, string)>();

            if (_data == null) {

                _data = new List<(Tensor, Tensor, Tensor)>();

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

        private List<(Tensor, Tensor, Tensor)> _data;
        private bool disposedValue;

        private (Tensor, Tensor, Tensor) Batchifier(IEnumerable<(int, string)> input, Func<string, IEnumerable<string>> tokenizer, Vocab.Vocab vocab)
        {
            var label_list = new List<long>();
            var text_list = new List<Tensor>();
            var offsets = new List<long>();
            offsets.Add(0);

            long last = 0;

            foreach (var (label, text) in input) {
                label_list.Add(label);
                var processed_text = torch.tensor(tokenizer(text).Select(t => (long)vocab[t]).ToArray(),dtype:torch.int64);
                text_list.Add(processed_text);
                last += processed_text.size(0);
                offsets.Add(last);
            }

            var labels = torch.tensor(label_list.ToArray(), dtype: torch.int64).to(_device);
            var texts = torch.cat(text_list.ToArray(), 0).to(_device);
            var offs = torch.tensor(offsets.Take(label_list.Count).ToArray(), dtype:torch.int64).to(_device);

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
