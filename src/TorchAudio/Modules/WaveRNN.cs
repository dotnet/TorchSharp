// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchaudio,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/audio/blob/c15eee23964098f88ab0afe25a8d5cd9d728af54/torchaudio/models/wavernn.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/audio/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

#nullable enable
namespace TorchSharp.Modules
{
    /// <summary>
    /// This class is used to represent a WaveRNN module.
    /// </summary>
    public class WaveRNN : nn.Module<Tensor, Tensor, Tensor>
    {
        private readonly int _pad;
        public readonly nn.Module<Tensor, Tensor> fc;
        public readonly nn.Module<Tensor, Tensor> fc1;
        public readonly nn.Module<Tensor, Tensor> fc2;
        public readonly nn.Module<Tensor, Tensor> fc3;
        public readonly int hop_length;
        public readonly int kernel_size;
        public readonly int n_aux;
        public readonly int n_bits;
        public readonly int n_classes;
        public readonly int n_rnn;
        public readonly nn.Module<Tensor, Tensor> relu1;
        public readonly nn.Module<Tensor, Tensor> relu2;
        public readonly GRU rnn1;
        public readonly GRU rnn2;
        internal readonly UpsampleNetwork upsample;

        internal WaveRNN(
            string name,
            long[] upsample_scales,
            int n_classes,
            int hop_length,
            int n_res_block = 10,
            int n_rnn = 512,
            int n_fc = 512,
            int kernel_size = 5,
            int n_freq = 128,
            int n_hidden = 128,
            int n_output = 128) : base(name)
        {
            this.kernel_size = kernel_size;
            this._pad = (kernel_size % 2 == 1 ? kernel_size - 1 : kernel_size) / 2;
            this.n_rnn = n_rnn;
            this.n_aux = n_output / 4;
            this.hop_length = hop_length;
            this.n_classes = n_classes;
            this.n_bits = (int)(Math.Log(this.n_classes) / Math.Log(2) + 0.5);

            long total_scale = 1;
            foreach (var upsample_scale in upsample_scales) {
                total_scale *= upsample_scale;
            }
            if (total_scale != this.hop_length) {
                throw new ArgumentException($"Expected: total_scale == hop_length, but found {total_scale} != {hop_length}");
            }

            this.upsample = new UpsampleNetwork("upsamplenetwork", upsample_scales, n_res_block, n_freq, n_hidden, n_output, kernel_size);
            this.fc = nn.Linear(n_freq + this.n_aux + 1, n_rnn);

            this.rnn1 = nn.GRU(n_rnn, n_rnn, batchFirst: true);
            this.rnn2 = nn.GRU(n_rnn + this.n_aux, n_rnn, batchFirst: true);

            this.relu1 = nn.ReLU(inplace: true);
            this.relu2 = nn.ReLU(inplace: true);

            this.fc1 = nn.Linear(n_rnn + this.n_aux, n_fc);
            this.fc2 = nn.Linear(n_fc + this.n_aux, n_fc);
            this.fc3 = nn.Linear(n_fc, this.n_classes);

            this.RegisterComponents();
        }

        /// <summary>
        /// Pass the input through the WaveRNN model.
        /// </summary>
        /// <param name="waveform">The input waveform to the WaveRNN layer</param>
        /// <param name="specgram">The input spectrogram to the WaveRNN layer</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public override Tensor forward(Tensor waveform, Tensor specgram)
        {
            if (waveform.size(1) != 1) {
                throw new ArgumentException("Require the input channel of waveform is 1");
            }
            if (specgram.size(1) != 1) {
                throw new ArgumentException("Require the input channel of specgram is 1");
            }
            // remove channel dimension until the end
            waveform = waveform.squeeze(1);
            specgram = specgram.squeeze(1);

            var batch_size = waveform.size(0);
            var h1 = torch.zeros(1, batch_size, this.n_rnn, dtype: waveform.dtype, device: waveform.device);
            var h2 = torch.zeros(1, batch_size, this.n_rnn, dtype: waveform.dtype, device: waveform.device);
            // output of upsample:
            // specgram: (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale)
            // aux: (n_batch, n_output, (n_time - kernel_size + 1) * total_scale)
            Tensor aux;
            (specgram, aux) = this.upsample.call(specgram);
            specgram = specgram.transpose(1, 2);
            aux = aux.transpose(1, 2);

            var aux_idx = new long[5];
            for (int i = 0; i < aux_idx.Length; i++) {
                aux_idx[i] = this.n_aux * i;
            }
            var a1 = aux[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(aux_idx[0], aux_idx[1])];
            var a2 = aux[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(aux_idx[1], aux_idx[2])];
            var a3 = aux[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(aux_idx[2], aux_idx[3])];
            var a4 = aux[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(aux_idx[3], aux_idx[4])];

            var x = torch.cat(new Tensor[] { waveform.unsqueeze(-1), specgram, a1 }, dim: -1);
            x = this.fc.call(x);
            var res = x;
            (x, _) = this.rnn1.call(x, h1);

            x = x + res;
            res = x;
            x = torch.cat(new Tensor[] { x, a2 }, dim: -1);
            (x, _) = this.rnn2.call(x, h2);

            x = x + res;
            x = torch.cat(new Tensor[] { x, a3 }, dim: -1);
            x = this.fc1.call(x);
            x = this.relu1.call(x);

            x = torch.cat(new Tensor[] { x, a4 }, dim: -1);
            x = this.fc2.call(x);
            x = this.relu2.call(x);
            x = this.fc3.call(x);

            // bring back channel dimension
            return x.unsqueeze(1);
        }

        /// <summary>
        /// Inference method of WaveRNN.
        /// </summary>
        /// <param name="specgram">Batch of spectrograms.</param>
        /// <param name="lengths">Indicates the valid length of each audio in the batch.</param>
        /// <returns>The inferred waveform and the valid length in time axis of the output Tensor.</returns>
        public virtual (Tensor, Tensor?) infer(Tensor specgram, Tensor? lengths = null)
        {
            var device = specgram.device;
            var dtype = specgram.dtype;

            specgram = torch.nn.functional.pad(specgram, (this._pad, this._pad));
            Tensor aux;
            (specgram, aux) = this.upsample.call(specgram);
            if (lengths is not null) {
                lengths = lengths * this.upsample.total_scale;
            }

            var output = new List<Tensor>();
            long b_size = specgram.size(0);
            long seq_len = specgram.size(2);

            var h1 = torch.zeros(new long[] { 1, b_size, this.n_rnn }, device: device, dtype: dtype);
            var h2 = torch.zeros(new long[] { 1, b_size, this.n_rnn }, device: device, dtype: dtype);
            var x = torch.zeros(new long[] { b_size, 1 }, device: device, dtype: dtype);

            var aux_split = new Tensor[4];
            for (int i = 0; i < 4; i++) {
                aux_split[i] = aux[TensorIndex.Colon, TensorIndex.Slice(this.n_aux * i, this.n_aux * (i + 1)), TensorIndex.Colon];
            }

            for (int i = 0; i < seq_len; i++) {

                var m_t = specgram[TensorIndex.Colon, TensorIndex.Colon, i];

                var a1_t = aux_split[0][TensorIndex.Colon, TensorIndex.Colon, i];
                var a2_t = aux_split[1][TensorIndex.Colon, TensorIndex.Colon, i];
                var a3_t = aux_split[2][TensorIndex.Colon, TensorIndex.Colon, i];
                var a4_t = aux_split[3][TensorIndex.Colon, TensorIndex.Colon, i];

                x = torch.cat(new Tensor[] { x, m_t, a1_t }, dim: 1);
                x = this.fc.call(x);
                (_, h1) = this.rnn1.call(x.unsqueeze(1), h1);

                x = x + h1[0];
                var inp = torch.cat(new Tensor[] { x, a2_t }, dim: 1);
                (_, h2) = this.rnn2.call(inp.unsqueeze(1), h2);

                x = x + h2[0];
                x = torch.cat(new Tensor[] { x, a3_t }, dim: 1);
                x = F.relu(this.fc1.call(x));

                x = torch.cat(new Tensor[] { x, a4_t }, dim: 1);
                x = F.relu(this.fc2.call(x));

                var logits = this.fc3.call(x);

                var posterior = F.softmax(logits, dim: 1);

                x = torch.multinomial(posterior, 1).@float();
                // Transform label [0, 2 ** n_bits - 1] to waveform [-1, 1]

                x = 2 * x / ((1 << this.n_bits) - 1.0) - 1.0;

                output.Add(x);
            }
            return (torch.stack(output).permute(1, 2, 0), lengths);
        }

        private class ResBlock : nn.Module<Tensor, Tensor>
        {
            public nn.Module<Tensor, Tensor> resblock_model;

            public ResBlock(string name, int n_freq = 128) : base(name)
            {
                this.resblock_model = nn.Sequential(
                    nn.Conv1d(inputChannel: n_freq, outputChannel: n_freq, kernelSize: 1, bias: false),
                    nn.BatchNorm1d(n_freq),
                    nn.ReLU(inplace: true),
                    nn.Conv1d(inputChannel: n_freq, outputChannel: n_freq, kernelSize: 1, bias: false),
                    nn.BatchNorm1d(n_freq));
                RegisterComponents();
            }

            public override Tensor forward(Tensor specgram)
            {
                return this.resblock_model.call(specgram) + specgram;
            }
        }

        internal class MelResNet : nn.Module<Tensor, Tensor>
        {
            public readonly nn.Module<Tensor, Tensor> melresnet_model;

            public MelResNet(
                string name,
                int n_res_block = 10,
                int n_freq = 128,
                int n_hidden = 128,
                int n_output = 128,
                int kernel_size = 5) : base(name)
            {
                var modules = new List<nn.Module<Tensor, Tensor>>();
                modules.Add(nn.Conv1d(inputChannel: n_freq, outputChannel: n_hidden, kernelSize: kernel_size, bias: false));
                modules.Add(nn.BatchNorm1d(n_hidden));
                modules.Add(nn.ReLU(inplace: true));
                for (int i = 0; i < n_res_block; i++) {
                    modules.Add(new ResBlock("resblock", n_hidden));
                }
                modules.Add(nn.Conv1d(inputChannel: n_hidden, outputChannel: n_output, kernelSize: 1));
                this.melresnet_model = nn.Sequential(modules);
                RegisterComponents();
            }

            public override Tensor forward(Tensor specgram)
            {
                return this.melresnet_model.call(specgram);
            }
        }

        public class Stretch2d : nn.Module<Tensor, Tensor>
        {
            public long freq_scale;
            public long time_scale;

            public Stretch2d(string name, long time_scale, long freq_scale) : base(name)
            {
                this.freq_scale = freq_scale;
                this.time_scale = time_scale;
                this.RegisterComponents();
            }

            public override Tensor forward(Tensor specgram)
            {
                return specgram.repeat_interleave(this.freq_scale, -2).repeat_interleave(this.time_scale, -1);
            }
        }

        internal class UpsampleNetwork : nn.Module<Tensor, (Tensor,Tensor)>
        {
            public readonly long indent;
            public readonly MelResNet resnet;
            public readonly Stretch2d resnet_stretch;
            public readonly long total_scale;
            public readonly nn.Module<Tensor, Tensor> upsample_layers;

            public UpsampleNetwork(
                string name,
                long[] upsample_scales,
                int n_res_block = 10,
                int n_freq = 128,
                int n_hidden = 128,
                int n_output = 128,
                int kernel_size = 5) : base(name)
            {
                long total_scale = 1;
                foreach (var upsample_scale in upsample_scales) {
                    total_scale *= upsample_scale;
                }
                this.total_scale = total_scale;

                this.indent = (kernel_size - 1) / 2 * total_scale;
                this.resnet = new MelResNet("melresnet", n_res_block, n_freq, n_hidden, n_output, kernel_size);
                this.resnet_stretch = new Stretch2d("stretch2d", total_scale, 1);

                var up_layers = new List<nn.Module<Tensor, Tensor>>();
                foreach (var scale in upsample_scales) {
                    var stretch = new Stretch2d("stretch2d", scale, 1);
                    var conv = nn.Conv2d(inputChannel: 1, outputChannel: 1, kernelSize: (1, scale * 2 + 1), padding: (0, scale), bias: false);
                    torch.nn.init.constant_(conv.weight, 1.0 / (scale * 2 + 1));
                    up_layers.Add(stretch);
                    up_layers.Add(conv);
                }
                this.upsample_layers = nn.Sequential(up_layers);
                this.RegisterComponents();
            }

            public override (Tensor, Tensor) forward(Tensor specgram)
            {
                var resnet_output = this.resnet.call(specgram).unsqueeze(1);
                resnet_output = this.resnet_stretch.call(resnet_output);
                resnet_output = resnet_output.squeeze(1);

                specgram = specgram.unsqueeze(1);
                var upsampling_output = this.upsample_layers.call(specgram);
                upsampling_output = upsampling_output.squeeze(1)[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(this.indent, -this.indent)];

                return (upsampling_output, resnet_output);
            }
        }
    }
}
