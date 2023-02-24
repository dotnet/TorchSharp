// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchaudio,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/audio/blob/bb77cbebb620a46fdc0dc7e6dae2253eef3f37e2/torchaudio/transforms/_transforms.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/audio/blob/main/LICENSE
//

using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchaudio;

namespace TorchSharp
{
    namespace Transforms
    {
        public class MelSpectrogram : Module<Tensor, Tensor>
        {
            public readonly long sample_rate;
            public readonly long n_fft;
            public readonly long win_length;
            public readonly long hop_length;
            public readonly long pad;
            public readonly double power;
            public readonly bool normalized;
            public readonly long n_mels;
            public readonly double? f_max;
            public readonly double f_min;
            public readonly Spectrogram spectrogram;
            public readonly MelScale mel_scale;


            internal MelSpectrogram(
                string name,
                long sample_rate = 16000,
                long n_fft = 400,
                long? win_length = null,
                long? hop_length = null,
                double f_min = 0.0,
                double? f_max = null,
                long pad = 0,
                long n_mels = 128,
                WindowFunction window_fn = null,
                double power = 2.0,
                bool normalized = false,
                bool center = true,
                PaddingModes pad_mode = PaddingModes.Reflect,
                bool onesided = true,
                MelNorm norm = MelNorm.none,
                torchaudio.MelScale mel_scale = torchaudio.MelScale.htk) : base(name)
            {
                this.sample_rate = sample_rate;
                this.n_fft = n_fft;
                this.win_length = win_length ?? n_fft;
                this.hop_length = hop_length ?? (this.win_length / 2);
                this.pad = pad;
                this.power = power;
                this.normalized = normalized;
                this.n_mels = n_mels;  // number of mel frequency bins
                this.f_max = f_max;
                this.f_min = f_min;
                this.spectrogram = torchaudio.transforms.Spectrogram(
                    n_fft: this.n_fft,
                    win_length: this.win_length,
                    hop_length: this.hop_length,
                    pad: this.pad,
                    window_fn: window_fn,
                    power: this.power,
                    normalized: this.normalized,
                    center: center,
                    pad_mode: pad_mode,
                    onesided: onesided);
                this.mel_scale = torchaudio.transforms.MelScale(
                    this.n_mels, this.sample_rate, this.f_min, this.f_max, this.n_fft / 2 + 1, norm, mel_scale
                );
                this.RegisterComponents();
            }

            /// <param name="waveform">Tensor of audio of dimension (..., time).</param>
            /// <returns>Mel frequency spectrogram of size (..., ``n_mels``, time).</returns>
            public override Tensor forward(Tensor waveform)
            {
                var specgram = this.spectrogram.call(waveform);
                var mel_specgram = this.mel_scale.call(specgram);
                return mel_specgram;
            }
        }
    }

    public partial class torchaudio
    {
        public partial class transforms
        {
            /// <summary>
            /// Create MelSpectrogram for a raw audio signal.
            /// </summary>
            /// <param name="sample_rate">Sample rate of audio signal.</param>
            /// <param name="n_fft">Size of FFT, creates ``n_fft / 2 + 1`` bins.</param>
            /// <param name="win_length">Window size.</param>
            /// <param name="hop_length">Length of hop between STFT windows.</param>
            /// <param name="f_min">Minimum frequency.</param>
            /// <param name="f_max">Maximum frequency.</param>
            /// <param name="pad">Two sided padding of signal.</param>
            /// <param name="n_mels">Number of mel filterbanks.</param>
            /// <param name="window_fn">A function to create a window tensor that is applied/multiplied to each frame/window.</param>
            /// <param name="power">Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for energy, 2 for power, etc.</param>
            /// <param name="normalized">Whether to normalize by magnitude after stft.</param>
            /// <param name="center">whether to pad :attr:`waveform` on both sides so that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.</param>
            /// <param name="pad_mode">controls the padding method used when :attr:`center` is ``true``.</param>
            /// <param name="onesided">controls whether to return half of results to avoid redundancy.</param>
            /// <param name="norm">If "slaney", divide the triangular mel weights by the width of the mel band (area normalization).</param>
            /// <param name="mel_scale">Scale to use: ``htk`` or ``slaney``.</param>
            /// <returns></returns>

            public static Transforms.MelSpectrogram MelSpectrogram(
                long sample_rate = 16000,
                long n_fft = 400,
                long? win_length = null,
                long? hop_length = null,
                double f_min = 0.0,
                double? f_max = null,
                long pad = 0,
                long n_mels = 128,
                WindowFunction window_fn = null,
                double power = 2.0,
                bool normalized = false,
                bool center = true,
                PaddingModes pad_mode = PaddingModes.Reflect,
                bool onesided = true,
                MelNorm norm = MelNorm.none,
                torchaudio.MelScale mel_scale = torchaudio.MelScale.htk)
            {
                return new Transforms.MelSpectrogram(
                    "MelSpectrogram",
                    sample_rate,
                    n_fft,
                    win_length,
                    hop_length,
                    f_min,
                    f_max,
                    pad,
                    n_mels,
                    window_fn,
                    power,
                    normalized,
                    center,
                    pad_mode,
                    onesided,
                    norm,
                    mel_scale);
            }
        }
    }
}