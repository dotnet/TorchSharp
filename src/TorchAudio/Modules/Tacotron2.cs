// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchaudio,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/audio/blob/e502df0106403f7666f89fee09715256ea2e0df3/torchaudio/models/tacotron2.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/audio/blob/main/LICENSE
//

// *****************************************************************************
//  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//      * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//      * Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//      * Neither the name of the NVIDIA CORPORATION nor the
//        names of its contributors may be used to endorse or promote products
//        derived from this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
//  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// *****************************************************************************

using System;
using System.Collections.Generic;
using System.Diagnostics;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

#nullable enable
namespace TorchSharp.Modules
{
    /// <summary>
    /// Tacotron2 model based on the implementation from
    /// Nvidia https://github.com/NVIDIA/DeepLearningExamples/.
    /// </summary>
    public class Tacotron2 : nn.Module<Tensor,Tensor,Tensor,Tensor, (Tensor, Tensor, Tensor, Tensor)>
    {
        public readonly bool mask_padding;
        public readonly int n_mels;
        public readonly int n_frames_per_step;

        private readonly Modules.Embedding embedding;
        private readonly Encoder encoder;
        private readonly Decoder decoder;
        private readonly Postnet postnet;

        internal Tacotron2(
            string name,
            bool mask_padding = false,
            int n_mels = 80,
            int n_symbol = 148,
            int n_frames_per_step = 1,
            int symbol_embedding_dim = 512,
            int encoder_embedding_dim = 512,
            int encoder_n_convolution = 3,
            int encoder_kernel_size = 5,
            int decoder_rnn_dim = 1024,
            int decoder_max_step = 2000,
            double decoder_dropout = 0.1,
            bool decoder_early_stopping = true,
            int attention_rnn_dim = 1024,
            int attention_hidden_dim = 128,
            int attention_location_n_filter = 32,
            int attention_location_kernel_size = 31,
            double attention_dropout = 0.1,
            int prenet_dim = 256,
            int postnet_n_convolution = 5,
            int postnet_kernel_size = 5,
            int postnet_embedding_dim = 512,
            double gate_threshold = 0.5) : base(name)
        {
            this.mask_padding = mask_padding;
            this.n_mels = n_mels;
            this.n_frames_per_step = n_frames_per_step;
            this.embedding = nn.Embedding(n_symbol, symbol_embedding_dim);
            torch.nn.init.xavier_uniform_(this.embedding.weight);
            this.encoder = new Encoder("encoder", encoder_embedding_dim, encoder_n_convolution, encoder_kernel_size);
            this.decoder = new Decoder(
                "decoder",
                n_mels,
                n_frames_per_step,
                encoder_embedding_dim,
                decoder_rnn_dim,
                decoder_max_step,
                decoder_dropout,
                decoder_early_stopping,
                attention_rnn_dim,
                attention_hidden_dim,
                attention_location_n_filter,
                attention_location_kernel_size,
                attention_dropout,
                prenet_dim,
                gate_threshold);
            this.postnet = new Postnet("postnet", n_mels, postnet_embedding_dim, postnet_kernel_size, postnet_n_convolution);
            RegisterComponents();
        }

        public override (Tensor, Tensor, Tensor, Tensor) forward(
            Tensor tokens,
            Tensor token_lengths,
            Tensor mel_specgram,
            Tensor mel_specgram_lengths)
        {
            var embedded_inputs = this.embedding.call(tokens).transpose(1, 2);

            var encoder_outputs = this.encoder.call(embedded_inputs, token_lengths);
            var (x, gate_outputs, alignments) = this.decoder.call(
                encoder_outputs, mel_specgram, memory_lengths: token_lengths
            );
            mel_specgram = x;

            var mel_specgram_postnet = this.postnet.call(mel_specgram);
            mel_specgram_postnet = mel_specgram + mel_specgram_postnet;

            if (this.mask_padding) {
                var mask = _get_mask_from_lengths(mel_specgram_lengths);
                mask = mask.expand(this.n_mels, mask.size(0), mask.size(1));
                mask = mask.permute(1, 0, 2);

                mel_specgram = mel_specgram.masked_fill(mask, 0.0);
                mel_specgram_postnet = mel_specgram_postnet.masked_fill(mask, 0.0);
                gate_outputs = gate_outputs.masked_fill(mask[TensorIndex.Colon, 0, TensorIndex.Colon], 1e3);
            }

            return (mel_specgram, mel_specgram_postnet, gate_outputs, alignments);
        }

        public (Tensor, Tensor, Tensor) infer(
            Tensor tokens, Tensor? lengths = null)
        {
            var n_batch = tokens.shape[0];
            var max_length = tokens.shape[1];
            if (lengths is null) {
                lengths = torch.tensor(new long[] { max_length }).expand(n_batch).to(tokens.device).to(tokens.dtype);
            }

            if (lengths is null) {
                throw new ArgumentNullException();
            }

            var embedded_inputs = this.embedding.call(tokens).transpose(1, 2);
            var encoder_outputs = this.encoder.call(embedded_inputs, lengths);
            var (mel_specgram, mel_specgram_lengths, _, alignments) = this.decoder.infer(encoder_outputs, lengths);

            var mel_outputs_postnet = this.postnet.call(mel_specgram);
            mel_outputs_postnet = mel_specgram + mel_outputs_postnet;

            alignments = alignments.unfold(1, n_batch, n_batch).transpose(0, 2);

            return (mel_outputs_postnet, mel_specgram_lengths, alignments);
        }

        private static Modules.Linear _get_linear_layer(long in_dim, long out_dim, bool bias = true, init.NonlinearityType w_init_gain = init.NonlinearityType.Linear)
        {
            var linear = torch.nn.Linear(in_dim, out_dim, hasBias: bias);
            torch.nn.init.xavier_uniform_(linear.weight, gain: torch.nn.init.calculate_gain(w_init_gain));
            return linear;
        }

        private static Modules.Conv1d _get_conv1d_layer(
            int in_channels,
            int out_channels,
            int kernel_size = 1,
            int stride = 1,
            long? padding = null,
            int dilation = 1,
            bool bias = true,
            init.NonlinearityType w_init_gain = init.NonlinearityType.Linear)
        {
            if (padding is null) {
                if (kernel_size % 2 != 1) {
                    throw new ArgumentException("kernel_size must be odd");
                }
                padding = dilation * (kernel_size - 1) / 2;
            }

            var conv1d = torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernelSize: kernel_size,
                stride: stride,
                padding: padding.Value,
                dilation: dilation,
                bias: bias);

            torch.nn.init.xavier_uniform_(conv1d.weight, gain: torch.nn.init.calculate_gain(w_init_gain));

            return conv1d;
        }

        private static Tensor _get_mask_from_lengths(Tensor lengths)
        {
            var max_len = torch.max(lengths).item<int>();
            var ids = torch.arange(0, max_len, device: lengths.device, dtype: lengths.dtype);
            var mask = (ids < lengths.unsqueeze(1)).to(torch.uint8);
            mask = torch.le(mask, 0);
            return mask;
        }

        private class LocationLayer : nn.Module<Tensor, Tensor>
        {
            private readonly Modules.Conv1d location_conv;
            private readonly Modules.Linear location_dense;

            public LocationLayer(
                string name,
                int attention_n_filter,
                int attention_kernel_size,
                int attention_hidden_dim) : base(name)
            {
                var padding = (attention_kernel_size - 1) / 2;
                this.location_conv = _get_conv1d_layer(
                    2,
                    attention_n_filter,
                    kernel_size: attention_kernel_size,
                    padding: padding,
                    bias: false,
                    stride: 1,
                    dilation: 1);
                this.location_dense = _get_linear_layer(
                    attention_n_filter,
                    attention_hidden_dim,
                    bias: false,
                    w_init_gain: init.NonlinearityType.Tanh);
                RegisterComponents();
            }

            public override Tensor forward(Tensor attention_weights_cat)
            {
                // (n_batch, attention_n_filter, text_lengths.max())
                var processed_attention = this.location_conv.call(attention_weights_cat);
                processed_attention = processed_attention.transpose(1, 2);
                // (n_batch, text_lengths.max(), attention_hidden_dim)
                processed_attention = this.location_dense.call(processed_attention);
                return processed_attention;
            }
        }

        private class Attention : nn.Module<Tensor,Tensor,Tensor,Tensor,Tensor, (Tensor, Tensor)>
        {
            private readonly LocationLayer location_layer;
            public readonly Modules.Linear memory_layer;
            private readonly Modules.Linear query_layer;
            public readonly float score_mask_value;
            private readonly Modules.Linear v;

            public Attention(
                string name,
                int attention_rnn_dim,
                int encoder_embedding_dim,
                int attention_hidden_dim,
                int attention_location_n_filter,
                int attention_location_kernel_size) : base(name)
            {
                this.query_layer = _get_linear_layer(attention_rnn_dim, attention_hidden_dim, bias: false, w_init_gain: init.NonlinearityType.Tanh);
                this.memory_layer = _get_linear_layer(
                    encoder_embedding_dim, attention_hidden_dim, bias: false, w_init_gain: init.NonlinearityType.Tanh);
                this.v = _get_linear_layer(attention_hidden_dim, 1, bias: false);
                this.location_layer = new LocationLayer(
                    "location_layer",
                    attention_location_n_filter,
                    attention_location_kernel_size,
                    attention_hidden_dim);
                this.score_mask_value = float.NegativeInfinity;
                RegisterComponents();
            }

            private Tensor _get_alignment_energies(Tensor query, Tensor processed_memory, Tensor attention_weights_cat)
            {
                var processed_query = this.query_layer.call(query.unsqueeze(1));
                var processed_attention_weights = this.location_layer.call(attention_weights_cat);
                var energies = this.v.call(torch.tanh(processed_query + processed_attention_weights + processed_memory));

                var alignment = energies.squeeze(2);
                return alignment;
            }

            public override (Tensor, Tensor) forward(
                Tensor attention_hidden_state,
                Tensor memory,
                Tensor processed_memory,
                Tensor attention_weights_cat,
                Tensor mask)
            {
                var alignment = this._get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat);

                alignment = alignment.masked_fill(mask, this.score_mask_value);

                var attention_weights = F.softmax(alignment, dim: 1);
                var attention_context = torch.bmm(attention_weights.unsqueeze(1), memory);
                attention_context = attention_context.squeeze(1);

                return (attention_context, attention_weights);
            }
        }

        public class Prenet : nn.Module<Tensor, Tensor>
        {
            private readonly Modules.ModuleList<Module<Tensor, Tensor>> layers;

            public Prenet(string name, int in_dim, long[] out_sizes) : base(name)
            {
                this.layers = nn.ModuleList<Module<Tensor, Tensor>>();
                long prev_size = in_dim;
                for (int i = 0; i < out_sizes.Length; i++) {
                    long in_size = prev_size;
                    long out_size = out_sizes[i];
                    this.layers.Add(
                        _get_linear_layer(in_size, out_size, bias: false));
                    prev_size = out_size;
                }
                RegisterComponents();
            }

            public override Tensor forward(Tensor x)
            {
                foreach (var linear in this.layers) {
                    x = F.dropout(F.relu(linear.call(x)), p: 0.5, training: true);
                }
                return x;
            }
        }

        private class Postnet : nn.Module<Tensor, Tensor>
        {
            private readonly Modules.ModuleList<Module<Tensor, Tensor>> convolutions;
            public readonly int n_convs;

            public Postnet(
                string name,
                int n_mels,
                int postnet_embedding_dim,
                int postnet_kernel_size,
                int postnet_n_convolution) : base(name)
            {
                this.convolutions = nn.ModuleList<Module<Tensor, Tensor>>();

                for (int i = 0; i < postnet_n_convolution; i++) {
                    var in_channels = i == 0 ? n_mels : postnet_embedding_dim;
                    var out_channels = i == postnet_n_convolution - 1 ? n_mels : postnet_embedding_dim;
                    var init_gain = i == postnet_n_convolution - 1 ? init.NonlinearityType.Linear : init.NonlinearityType.Tanh;
                    var num_features = i == postnet_n_convolution - 1 ? n_mels : postnet_embedding_dim;
                    this.convolutions.append(
                        nn.Sequential(
                            _get_conv1d_layer(
                                in_channels,
                                out_channels,
                                kernel_size: postnet_kernel_size,
                                stride: 1,
                                padding: (postnet_kernel_size - 1) / 2,
                                dilation: 1,
                                w_init_gain: init_gain),
                            nn.BatchNorm1d(num_features)));
                }

                this.n_convs = this.convolutions.Count;
                RegisterComponents();
            }

            public override Tensor forward(Tensor x)
            {
                for (int i = 0; i < this.convolutions.Count; i++) {
                    var conv = this.convolutions[i];
                    if (i < this.n_convs - 1) {
                        x = F.dropout(torch.tanh(conv.call(x)), 0.5, training: this.training);
                    } else {
                        x = F.dropout(conv.call(x), 0.5, training: this.training);
                    }
                }
                return x;
            }
        }

        private class Encoder : nn.Module<Tensor, Tensor, Tensor>
        {
            private readonly Modules.ModuleList<Module<Tensor, Tensor>> convolutions;
            private readonly Modules.LSTM lstm;

            public Encoder(
                string name,
                int encoder_embedding_dim,
                int encoder_n_convolution,
                int encoder_kernel_size) : base(name)
            {
                this.convolutions = nn.ModuleList<Module<Tensor, Tensor>>();
                for (int i = 0; i < encoder_n_convolution; i++) {
                    var conv_layer = nn.Sequential(
                        _get_conv1d_layer(
                            encoder_embedding_dim,
                            encoder_embedding_dim,
                            kernel_size: encoder_kernel_size,
                            stride: 1,
                            padding: (encoder_kernel_size - 1) / 2,
                            dilation: 1,
                            w_init_gain: init.NonlinearityType.ReLU
                        ),
                        nn.BatchNorm1d(encoder_embedding_dim)
                    );
                    this.convolutions.append(conv_layer);
                }

                this.lstm = nn.LSTM(
                    encoder_embedding_dim,
                    encoder_embedding_dim / 2,
                    1,
                    batchFirst: true,
                    bidirectional: true
                );
                this.lstm.flatten_parameters();
                RegisterComponents();
            }

            public override Tensor forward(Tensor x, Tensor input_lengths)
            {
                foreach (var conv in this.convolutions) {
                    x = F.dropout(F.relu(conv.call(x)), 0.5, training: this.training);
                }

                x = x.transpose(1, 2);

                input_lengths = input_lengths.cpu();
                var packed_x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first: true);

                var (packed_outputs, _, _) = this.lstm.call(packed_x);
                var (outputs, _) = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first: true);

                return outputs;
            }
        }

        private class Decoder : nn.Module<Tensor, Tensor, Tensor, (Tensor, Tensor, Tensor)>
        {
            public readonly double attention_dropout;
            private readonly Attention attention_layer;
            private readonly Modules.LSTMCell attention_rnn;
            public readonly long attention_rnn_dim;
            public readonly double decoder_dropout;
            public readonly bool decoder_early_stopping;
            public readonly long decoder_max_step;
            private readonly Modules.LSTMCell decoder_rnn;
            public readonly long decoder_rnn_dim;
            public readonly long encoder_embedding_dim;
            private readonly Modules.Linear gate_layer;
            public readonly double gate_threshold;
            private readonly Modules.Linear linear_projection;
            public readonly int n_frames_per_step;
            public readonly int n_mels;
            public readonly Prenet prenet;
            public readonly long prenet_dim;

            public Decoder(
                string name,
                int n_mels,
                int n_frames_per_step,
                int encoder_embedding_dim,
                int decoder_rnn_dim,
                int decoder_max_step,
                double decoder_dropout,
                bool decoder_early_stopping,
                int attention_rnn_dim,
                int attention_hidden_dim,
                int attention_location_n_filter,
                int attention_location_kernel_size,
                double attention_dropout,
                long prenet_dim,
                double gate_threshold) : base(name)
            {
                this.n_mels = n_mels;
                this.n_frames_per_step = n_frames_per_step;
                this.encoder_embedding_dim = encoder_embedding_dim;
                this.attention_rnn_dim = attention_rnn_dim;
                this.decoder_rnn_dim = decoder_rnn_dim;
                this.prenet_dim = prenet_dim;
                this.decoder_max_step = decoder_max_step;
                this.gate_threshold = gate_threshold;
                this.attention_dropout = attention_dropout;
                this.decoder_dropout = decoder_dropout;
                this.decoder_early_stopping = decoder_early_stopping;

                this.prenet = new Prenet("prenet", n_mels * n_frames_per_step, new[] {
                    prenet_dim,
                    prenet_dim
                });

                this.attention_rnn = nn.LSTMCell(prenet_dim + encoder_embedding_dim, attention_rnn_dim);

                this.attention_layer = new Attention(
                    "attention",
                    attention_rnn_dim,
                    encoder_embedding_dim,
                    attention_hidden_dim,
                    attention_location_n_filter,
                    attention_location_kernel_size);

                this.decoder_rnn = nn.LSTMCell(attention_rnn_dim + encoder_embedding_dim, decoder_rnn_dim, true);

                this.linear_projection = _get_linear_layer(decoder_rnn_dim + encoder_embedding_dim, n_mels * n_frames_per_step);

                this.gate_layer = _get_linear_layer(
                    decoder_rnn_dim + encoder_embedding_dim, 1, bias: true, w_init_gain: init.NonlinearityType.Sigmoid);
                RegisterComponents();
            }

            private Tensor _get_initial_frame(Tensor memory)
            {
                var n_batch = memory.size(0);
                var dtype = memory.dtype;
                var device = memory.device;
                var decoder_input = torch.zeros(n_batch, this.n_mels * this.n_frames_per_step, dtype: dtype, device: device);
                return decoder_input;
            }

            private (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) _initialize_decoder_states(Tensor memory)
            {
                var n_batch = memory.size(0);
                var max_time = memory.size(1);
                var dtype = memory.dtype;
                var device = memory.device;

                var attention_hidden = torch.zeros(n_batch, this.attention_rnn_dim, dtype: dtype, device: device);
                var attention_cell = torch.zeros(n_batch, this.attention_rnn_dim, dtype: dtype, device: device);

                var decoder_hidden = torch.zeros(n_batch, this.decoder_rnn_dim, dtype: dtype, device: device);
                var decoder_cell = torch.zeros(n_batch, this.decoder_rnn_dim, dtype: dtype, device: device);

                var attention_weights = torch.zeros(n_batch, max_time, dtype: dtype, device: device);
                var attention_weights_cum = torch.zeros(n_batch, max_time, dtype: dtype, device: device);
                var attention_context = torch.zeros(n_batch, this.encoder_embedding_dim, dtype: dtype, device: device);

                var processed_memory = this.attention_layer.memory_layer.call(memory);

                return (
                    attention_hidden,
                    attention_cell,
                    decoder_hidden,
                    decoder_cell,
                    attention_weights,
                    attention_weights_cum,
                    attention_context,
                    processed_memory);
            }

            private Tensor _parse_decoder_inputs(Tensor decoder_inputs)
            {
                // (n_batch, n_mels, mel_specgram_lengths.max()) -> (n_batch, mel_specgram_lengths.max(), n_mels)
                decoder_inputs = decoder_inputs.transpose(1, 2);
                decoder_inputs = decoder_inputs.view(
                    decoder_inputs.size(0),
                    decoder_inputs.size(1) / this.n_frames_per_step,
                    -1);
                // (n_batch, mel_specgram_lengths.max(), n_mels) -> (mel_specgram_lengths.max(), n_batch, n_mels)
                decoder_inputs = decoder_inputs.transpose(0, 1);
                return decoder_inputs;
            }

            private (Tensor, Tensor, Tensor) _parse_decoder_outputs(Tensor mel_specgram, Tensor gate_outputs, Tensor alignments)
            {
                // (mel_specgram_lengths.max(), n_batch, text_lengths.max())
                // -> (n_batch, mel_specgram_lengths.max(), text_lengths.max())
                alignments = alignments.transpose(0, 1).contiguous();
                // (mel_specgram_lengths.max(), n_batch) -> (n_batch, mel_specgram_lengths.max())
                gate_outputs = gate_outputs.transpose(0, 1).contiguous();
                // (mel_specgram_lengths.max(), n_batch, n_mels) -> (n_batch, mel_specgram_lengths.max(), n_mels)
                mel_specgram = mel_specgram.transpose(0, 1).contiguous();
                // decouple frames per step
                var shape = new long[] { mel_specgram.shape[0], -1, this.n_mels };
                mel_specgram = mel_specgram.view(shape);
                // (n_batch, mel_specgram_lengths.max(), n_mels) -> (n_batch, n_mels, T_out)
                mel_specgram = mel_specgram.transpose(1, 2);

                return (mel_specgram, gate_outputs, alignments);
            }

            public (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) decode(
                Tensor decoder_input,
                Tensor attention_hidden,
                Tensor attention_cell,
                Tensor decoder_hidden,
                Tensor decoder_cell,
                Tensor attention_weights,
                Tensor attention_weights_cum,
                Tensor attention_context,
                Tensor memory,
                Tensor processed_memory,
                Tensor mask)
            {
                var cell_input = torch.cat(new[] { decoder_input, attention_context }, -1);

                (attention_hidden, attention_cell) = this.attention_rnn.call(cell_input, (attention_hidden, attention_cell));
                attention_hidden = F.dropout(attention_hidden, this.attention_dropout, training: this.training);

                var attention_weights_cat = torch.cat(new[] { attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1) }, dim: 1);
                (attention_context, attention_weights) = this.attention_layer.call(
                    attention_hidden, memory, processed_memory, attention_weights_cat, mask);

                attention_weights_cum += attention_weights;
                decoder_input = torch.cat(new[] { attention_hidden, attention_context }, -1);

                (decoder_hidden, decoder_cell) = this.decoder_rnn.call(decoder_input, (decoder_hidden, decoder_cell));
                decoder_hidden = F.dropout(decoder_hidden, this.decoder_dropout, training: this.training);

                var decoder_hidden_attention_context = torch.cat(new[] { decoder_hidden, attention_context }, dim: 1);
                var decoder_output = this.linear_projection.call(decoder_hidden_attention_context);

                var gate_prediction = this.gate_layer.call(decoder_hidden_attention_context);

                return (
                    decoder_output,
                    gate_prediction,
                    attention_hidden,
                    attention_cell,
                    decoder_hidden,
                    decoder_cell,
                    attention_weights,
                    attention_weights_cum,
                    attention_context);
            }

            // Decoder forward pass for training.
            public override (Tensor, Tensor, Tensor) forward(Tensor memory, Tensor mel_specgram_truth, Tensor memory_lengths)
            {
                var decoder_input = this._get_initial_frame(memory).unsqueeze(0);
                var decoder_inputs = this._parse_decoder_inputs(mel_specgram_truth);
                decoder_inputs = torch.cat(new[] { decoder_input, decoder_inputs }, dim: 0);
                decoder_inputs = this.prenet.call(decoder_inputs);

                var mask = _get_mask_from_lengths(memory_lengths);
                var (
                    attention_hidden,
                    attention_cell,
                    decoder_hidden,
                    decoder_cell,
                    attention_weights,
                    attention_weights_cum,
                    attention_context,
                    processed_memory
                    ) = this._initialize_decoder_states(memory);

                var mel_output_list = new List<Tensor>();
                var gate_output_list = new List<Tensor>();
                var alignment_list = new List<Tensor>();
                while (mel_output_list.Count < decoder_inputs.size(0) - 1) {
                    decoder_input = decoder_inputs[mel_output_list.Count];
                    Tensor mel_output, gate_output;
                    (
                        mel_output,
                        gate_output,
                        attention_hidden,
                        attention_cell,
                        decoder_hidden,
                        decoder_cell,
                        attention_weights,
                        attention_weights_cum,
                        attention_context
                        ) = this.decode(
                            decoder_input,
                            attention_hidden,
                            attention_cell,
                            decoder_hidden,
                            decoder_cell,
                            attention_weights,
                            attention_weights_cum,
                            attention_context,
                            memory,
                            processed_memory,
                            mask);

                    mel_output_list.Add(mel_output.squeeze(1));
                    gate_output_list.Add(gate_output.squeeze(1));
                    alignment_list.Add(attention_weights);
                }

                var (mel_specgram, gate_outputs, alignments) = this._parse_decoder_outputs(
                    torch.stack(mel_output_list), torch.stack(gate_output_list), torch.stack(alignment_list));

                return (mel_specgram, gate_outputs, alignments);
            }

            public new (Tensor, Tensor, Tensor) call(Tensor memory, Tensor mel_specgram_truth, Tensor memory_lengths) => base.call(memory, mel_specgram_truth, memory_lengths);

            private Tensor _get_go_frame(Tensor memory)
            {
                var n_batch = memory.size(0);
                var dtype = memory.dtype;
                var device = memory.device;
                var decoder_input = torch.zeros(n_batch, this.n_mels * this.n_frames_per_step, dtype: dtype, device: device);
                return decoder_input;
            }

            public (Tensor, Tensor, Tensor, Tensor) infer(Tensor memory, Tensor memory_lengths)
            {
                var batch_size = memory.size(0);
                var device = memory.device;

                var decoder_input = this._get_go_frame(memory);

                var mask = _get_mask_from_lengths(memory_lengths);
                var (
                    attention_hidden,
                    attention_cell,
                    decoder_hidden,
                    decoder_cell,
                    attention_weights,
                    attention_weights_cum,
                    attention_context,
                    processed_memory
                    ) = this._initialize_decoder_states(memory);

                var mel_specgram_lengths = torch.zeros(new[] {
                    batch_size
                }, dtype: torch.int32, device: device);
                var finished = torch.zeros(new[] {
                    batch_size
                }, dtype: torch.@bool, device: device);
                var mel_specgram_list = new List<Tensor>();
                var gate_output_list = new List<Tensor>();
                var alignment_list = new List<Tensor>();
                for (long i = 0; i < this.decoder_max_step; i++) {
                    decoder_input = this.prenet.call(decoder_input);
                    Tensor mel_specgram, gate_output;
                    (
                        mel_specgram,
                        gate_output,
                        attention_hidden,
                        attention_cell,
                        decoder_hidden,
                        decoder_cell,
                        attention_weights,
                        attention_weights_cum,
                        attention_context
                        ) = this.decode(
                            decoder_input,
                            attention_hidden,
                            attention_cell,
                            decoder_hidden,
                            decoder_cell,
                            attention_weights,
                            attention_weights_cum,
                            attention_context,
                            memory,
                            processed_memory,
                            mask);

                    mel_specgram_list.Add(mel_specgram.unsqueeze(0));
                    gate_output_list.Add(gate_output.transpose(0, 1));
                    alignment_list.Add(attention_weights);
                    mel_specgram_lengths[~finished] += 1;

                    finished |= torch.sigmoid(gate_output.squeeze(1)) > this.gate_threshold;
                    if (this.decoder_early_stopping && torch.all(finished).item<bool>()) {
                        break;
                    }

                    decoder_input = mel_specgram;
                }

                if (mel_specgram_list.Count == this.decoder_max_step) {
                    Debug.WriteLine("Reached max decoder steps. The generated spectrogram might not cover the whole transcript.");
                }

                var mel_specgrams = torch.cat(mel_specgram_list, dim: 0);
                var gate_outputs = torch.cat(gate_output_list, dim: 0);
                var alignments = torch.cat(alignment_list, dim: 0);

                (mel_specgrams, gate_outputs, alignments) = this._parse_decoder_outputs(mel_specgrams, gate_outputs, alignments);

                return (mel_specgrams, mel_specgram_lengths, gate_outputs, alignments);
            }
        }
    }
}