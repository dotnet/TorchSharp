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
using F = TorchSharp.torchaudio.functional;

namespace TorchSharp
{
    namespace Transforms
    {
        public class MelScale : Module<Tensor, Tensor>
        {
            public readonly long n_mels;
            public readonly long sample_rate;
            public readonly double f_max;
            public readonly double f_min;
            public readonly MelNorm norm;
            public readonly torchaudio.MelScale mel_scale;
            public readonly Tensor fb;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    fb.Dispose();
                }
                base.Dispose(disposing);
            }

            internal MelScale(
                string name,
                long n_mels = 128,
                long sample_rate = 16000,
                double f_min = 0.0,
                double? f_max = null,
                long n_stft = 201,
                MelNorm norm = MelNorm.none,
                torchaudio.MelScale mel_scale = torchaudio.MelScale.htk) : base(name)
            {
                this.n_mels = n_mels;
                this.sample_rate = sample_rate;
                this.f_max = f_max ?? (sample_rate / 2);
                this.f_min = f_min;
                this.norm = norm;
                this.mel_scale = mel_scale;

                if (f_min > this.f_max) {
                    throw new ArgumentOutOfRangeException($"Require f_min: {f_min} < f_max: {this.f_max}");
                }
                this.fb = F.melscale_fbanks(n_stft, this.f_min, this.f_max, this.n_mels, this.sample_rate, this.norm, this.mel_scale);
                this.register_buffer("fb", fb);
            }

            /// <param name="specgram">A spectrogram STFT of dimension (..., freq, time).</param>
            /// <returns>Mel frequency spectrogram of size (..., ``n_mels``, time).</returns>
            public override Tensor forward(Tensor specgram)
            {
                var mel_specgram = torch.matmul(specgram.transpose(-1, -2), this.fb).transpose(-1, -2);

                return mel_specgram;
            }
        }
    }

    public partial class torchaudio
    {
        public partial class transforms
        {
            /// <summary>
            /// Turn a normal STFT into a mel frequency STFT with triangular filter banks.
            /// </summary>
            /// <param name="n_mels">Number of mel filterbanks.</param>
            /// <param name="sample_rate">Sample rate of audio signal.</param>
            /// <param name="f_min">Minimum frequency.</param>
            /// <param name="f_max">Maximum frequency.</param>
            /// <param name="n_stft">Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`.</param>
            /// <param name="norm">If 'slaney', divide the triangular mel weights by the width of the mel band (area normalization).</param>
            /// <param name="mel_scale">Scale to use: ``htk`` or ``slaney``.</param>
            /// <returns></returns>
            public static Transforms.MelScale MelScale(
                long n_mels = 128,
                long sample_rate = 16000,
                double f_min = 0.0,
                double? f_max = null,
                long n_stft = 201,
                MelNorm norm = MelNorm.none,
                torchaudio.MelScale mel_scale = torchaudio.MelScale.htk)
            {
                return new Transforms.MelScale(
                    "MelScale",
                    n_mels,
                    sample_rate,
                    f_min,
                    f_max,
                    n_stft,
                    norm,
                    mel_scale);
            }
        }
    }
}