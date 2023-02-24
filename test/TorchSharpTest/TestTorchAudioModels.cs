using Xunit;

namespace TorchSharp
{
    public class TestTorchAudioModels
    {
        private Modules.Tacotron2 CreateTacotron2(int n_symbols)
        {
            return torchaudio.models.Tacotron2(
                mask_padding: false,
                n_mels: 80,
                n_frames_per_step: 1,
                symbol_embedding_dim: 512,
                encoder_embedding_dim: 512,
                encoder_n_convolution: 3,
                encoder_kernel_size: 5,
                decoder_rnn_dim: 1024,
                decoder_max_step: 50, // This is very small to run the test quickly.
                decoder_dropout: 0.1,
                decoder_early_stopping: true,
                attention_rnn_dim: 1024,
                attention_hidden_dim: 128,
                attention_location_n_filter: 32,
                attention_location_kernel_size: 31,
                attention_dropout: 0.1,
                prenet_dim: 256,
                postnet_n_convolution: 5,
                postnet_kernel_size: 5,
                postnet_embedding_dim: 512,
                gate_threshold: 0.5,
                n_symbol: n_symbols
                );
        }

        private Modules.WaveRNN CreateWaveRNN()
        {
            return torchaudio.models.WaveRNN(
                upsample_scales: new long[] { 5, 5, 11 },
                n_classes: 1 << 8,  // n_bits = 8
                hop_length: 275,
                n_res_block: 10,
                n_rnn: 512,
                n_fc: 512,
                kernel_size: 5,
                n_freq: 80,
                n_hidden: 128,
                n_output: 128);
        }

        [Fact]
        public void Tacotron2ModelForward()
        {
            using (var scope = torch.NewDisposeScope()) {
                var n_symbols = 96;
                var n_mels = 80;
                var tacotron2 = CreateTacotron2(n_symbols);
                tacotron2.train();
                Assert.Equal(n_mels, tacotron2.n_mels);

                var batch_size = 3;
                var max_token_length = 10;
                var token = torch.randint(0, n_symbols, new long[] { batch_size, max_token_length });
                var token_lengths = torch.tensor(new int[] { 10, 7, 6 });

                var max_spec_length = 100;
                var spec = torch.randn(new long[] { batch_size, n_mels, max_spec_length });
                var spec_lengths = torch.tensor(new int[] { 30, 25, 17 });

                var (spec_output, postnet_output, gate_output, alignments) = tacotron2.call(token, token_lengths, spec, spec_lengths);

                Assert.Equal(spec.shape, spec_output.shape);
                Assert.Equal(spec.shape, postnet_output.shape);
                Assert.Equal(new long[] { batch_size, spec.shape[2] }, gate_output.shape);
                Assert.Equal(new long[] { batch_size, spec.shape[2], token.shape[1] }, alignments.shape);
            }
        }

        [Fact]
        public void Tacotron2ModelInfer()
        {
            using (var scope = torch.NewDisposeScope()) {
                var n_symbols = 96;
                var n_mels = 80;
                var tacotron2 = CreateTacotron2(n_symbols);
                tacotron2.eval();
                Assert.Equal(n_mels, tacotron2.n_mels);

                using (torch.no_grad()) {
                    var batch_size = 3;
                    var max_length = 4;
                    var token = torch.randint(0, n_symbols, new long[] { batch_size, max_length });
                    var token_lengths = torch.tensor(new int[] { 4, 3, 2 });
                    var (spec, spec_lengths, alignments) = tacotron2.infer(token, token_lengths);

                    Assert.Equal(3, spec.shape.Length);
                    Assert.Equal(batch_size, spec.shape[0]);
                    Assert.Equal(new long[] { batch_size }, spec_lengths.shape);
                    Assert.Equal(new long[] { batch_size, spec.shape[2], token.shape[1] }, alignments.shape);
                }
            }
        }

        [Fact]
        public void WaveRNNModelForward()
        {
            using (var scope = torch.NewDisposeScope()) {
                var wavernn = CreateWaveRNN();
                long batch_size = 2;
                var specgram = torch.randn(new long[] { batch_size, 1, 80, 6 });
                var waveform_len = (specgram.shape[3] - 5 + 1) * (5 * 5 * 11);
                var waveform = torch.randn(new long[] { batch_size, 1, waveform_len });
                // specgram: (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale)
                var output = wavernn.call(waveform, specgram);
                Assert.Equal(new long[] { batch_size, 1, waveform.shape[2], 1 << 8 }, output.shape);
            }
        }

        [Fact]
        public void Wav2Vec2ModelForward()
        {
            using (var scope = torch.NewDisposeScope()) {
                var model = torchaudio.models.wav2vec2_model(
                    extractor_mode: torchaudio.models.FeatureExtractorNormMode.group_norm,
                    extractor_conv_layer_config: new long[][] {
                        new long[] { 512, 10, 5 },

                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },

                        new long[] { 512, 2, 2 },
                        new long[] { 512, 2, 2 }
                    },
                    extractor_conv_bias: false,
                    encoder_embed_dim: 768,
                    encoder_projection_dropout: 0.1,
                    encoder_pos_conv_kernel: 128,
                    encoder_pos_conv_groups: 16,
                    encoder_num_layers: 12,
                    encoder_num_heads: 12,
                    encoder_attention_dropout: 0.1,
                    encoder_ff_interm_features: 3072,
                    encoder_ff_interm_dropout: 0.0,
                    encoder_dropout: 0.1,
                    encoder_layer_norm_first: false,
                    encoder_layer_drop: 0.05,
                    aux_num_out: 29);
                model.eval();
                var waveform = torch.randn(new long[] { 2, 1000 }, torch.float32);
                var (output, length) = model.call(waveform);
                Assert.Equal(2, output.size(0));
            }
        }
    }
}