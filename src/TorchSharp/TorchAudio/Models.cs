// Copyright</param>

using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class models
        {
            /// <summary>
            /// Tacotron2 model based on the implementation from
            /// Nvidia https://github.com/NVIDIA/DeepLearningExamples/.
            /// </summary>
            /// <param name="mask_padding">Use mask padding</param>
            /// <param name="n_mels">Number of mel bins</param>
            /// <param name="n_symbol">Number of symbols for the input text</param>
            /// <param name="n_frames_per_step">Number of frames processed per step, only 1 is supported</param>
            /// <param name="symbol_embedding_dim">Input embedding dimension</param>
            /// <param name="encoder_n_convolution">Number of encoder convolutions</param>
            /// <param name="encoder_kernel_size">Encoder kernel size</param>
            /// <param name="encoder_embedding_dim">Encoder embedding dimension</param>
            /// <param name="decoder_rnn_dim">Number of units in decoder LSTM</param>
            /// <param name="decoder_max_step">Maximum number of output mel spectrograms</param>
            /// <param name="decoder_dropout">Dropout probability for decoder LSTM</param>
            /// <param name="decoder_early_stopping">Continue decoding after all samples are finished</param>
            /// <param name="attention_rnn_dim">Number of units in attention LSTM</param>
            /// <param name="attention_hidden_dim">Dimension of attention hidden representation</param>
            /// <param name="attention_location_n_filter">Number of filters for attention model</param>
            /// <param name="attention_location_kernel_size">Kernel size for attention model</param>
            /// <param name="attention_dropout">Dropout probability for attention LSTM</param>
            /// <param name="prenet_dim">Number of ReLU units in prenet layers</param>
            /// <param name="postnet_n_convolution">Number of postnet convolutions</param>
            /// <param name="postnet_kernel_size">Postnet kernel size</param>
            /// <param name="postnet_embedding_dim">Postnet embedding dimension</param>
            /// <param name="gate_threshold">Probability threshold for stop token</param>
            /// <returns>Tacotron2 model</returns>
            public static Modules.Tacotron2 Tacotron2(
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
                double gate_threshold = 0.5)
            {
                return new Modules.Tacotron2(
                    "tacotron2",
                    mask_padding,
                    n_mels,
                    n_symbol,
                    n_frames_per_step,
                    symbol_embedding_dim,
                    encoder_embedding_dim,
                    encoder_n_convolution,
                    encoder_kernel_size,
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
                    postnet_n_convolution,
                    postnet_kernel_size,
                    postnet_embedding_dim,
                    gate_threshold);
            }
        }
    }
}