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
using System.Collections.Generic;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchaudio;
using F = TorchSharp.torchaudio.functional;

namespace TorchSharp
{
    namespace Transforms
    {
        public sealed class InverseMelScale : Module<Tensor, Tensor>, ITransform
        {
            public readonly double f_max;
            public readonly double f_min;
            public readonly long max_iter;
            public readonly long n_mels;
            public readonly long sample_rate;
            public readonly double tolerance_change;
            public readonly double tolerance_loss;
            public readonly Tensor fb;

            protected override void Dispose(bool disposing)
            {
                if (disposing) {
                    fb.Dispose();
                }
                base.Dispose(disposing);
            }

            internal InverseMelScale(
                string name,
                long n_stft,
                long n_mels = 128,
                long sample_rate = 16000,
                double f_min = 0.0,
                double? f_max = null,
                long max_iter = 100000,
                double tolerance_loss = 1e-05,
                double tolerance_change = 1e-08,
                MelNorm norm = MelNorm.none,
                torchaudio.MelScale mel_scale = torchaudio.MelScale.htk) : base(name)
            {
                this.n_mels = n_mels;
                this.sample_rate = sample_rate;
                this.f_max = f_max ?? (sample_rate / 2);
                this.f_min = f_min;
                this.max_iter = max_iter;
                this.tolerance_loss = tolerance_loss;
                this.tolerance_change = tolerance_change;

                if (f_min > this.f_max) {
                    throw new ArgumentException($"Require f_min: {this.f_min} <= f_max: {this.f_max}");
                }

                this.fb = F.melscale_fbanks(n_stft, this.f_min, this.f_max, this.n_mels, this.sample_rate, norm, mel_scale);
                this.register_buffer("fb", this.fb);
            }

            /// <param name="melspec">A Mel frequency spectrogram of dimension (..., ``n_mels``, time)</param>
            /// <returns>Linear scale spectrogram of size (..., freq, time)</returns>
            /// <exception cref="ArgumentException"></exception>
            public override Tensor forward(Tensor melspec)
            {
                using var d = torch.NewDisposeScope();

                // pack batch
                var shape = melspec.size();
                melspec = melspec.view(-1, shape[shape.Length - 2], shape[shape.Length - 1]);

                var n_mels = shape[shape.Length - 2];
                var time = shape[shape.Length - 1];
                var freq = this.fb.size(0); // (freq, n_mels)
                melspec = melspec.transpose(-1, -2);
                if (this.n_mels != n_mels) {
                    throw new ArgumentException($"Expected an input with {this.n_mels} mel bins. Found: {n_mels}");
                }

                var specgram = nn.Parameter(torch.rand(
                    melspec.size(0), time, freq, requires_grad: true, dtype: melspec.dtype, device: melspec.device));

                var optim = torch.optim.SGD(
                    new List<Modules.Parameter> { specgram },
                    learningRate: 0.1, momentum: 0.9);

                var loss = float.PositiveInfinity;
                for (long i = 0; i < this.max_iter; i++) {
                    using var d2 = torch.NewDisposeScope();

                    optim.zero_grad();
                    var diff = melspec - specgram.matmul(this.fb);
                    var new_loss = diff.pow(2).sum(dim: -1).mean();
                    // take sum over mel-frequency then average over other dimensions
                    // so that loss threshold is applied par unit timeframe
                    new_loss.backward();
                    optim.step();
                    using (torch.no_grad())
                        specgram.set_(specgram.clamp(min: 0));

                    var new_loss_value = new_loss.item<float>();
                    if (new_loss_value < this.tolerance_loss || Math.Abs(loss - new_loss_value) < this.tolerance_change) {
                        break;
                    }
                    loss = new_loss_value;
                }

                specgram.requires_grad_(false);
                var specgram_tensor = specgram.clamp(min: 0).transpose(-1, -2);

                // unpack batch
                shape[shape.Length - 2] = freq;
                shape[shape.Length - 1] = time;
                specgram_tensor = specgram_tensor.view(shape);
                return specgram_tensor.MoveToOuterDisposeScope();
            }
        }
    }

    public partial class torchaudio
    {
        public partial class transforms
        {
            /// <summary>
            /// Estimate a STFT in normal frequency domain from mel frequency domain.
            /// It minimizes the euclidian norm between the input mel-spectrogram and the product between
            /// the estimated spectrogram and the filter banks using SGD.
            /// </summary>
            /// <param name="n_stft">Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`.</param>
            /// <param name="n_mels">Number of mel filterbanks.</param>
            /// <param name="sample_rate">Sample rate of audio signal.</param>
            /// <param name="f_min">Minimum frequency.</param>
            /// <param name="f_max">Maximum frequency.</param>
            /// <param name="max_iter">Maximum number of optimization iterations.</param>
            /// <param name="tolerance_loss">Value of loss to stop optimization at.</param>
            /// <param name="tolerance_change">Difference in losses to stop optimization at.</param>
            /// <param name="norm">If 'slaney', divide the triangular mel weights by the width of the mel band (area normalization).</param>
            /// <param name="mel_scale">Scale to use: ``htk`` or ``slaney``.</param>
            /// <returns></returns>
            public static Transforms.InverseMelScale InverseMelScale(
                long n_stft,
                long n_mels = 128,
                long sample_rate = 16000,
                double f_min = 0.0,
                double? f_max = null,
                long max_iter = 100000,
                double tolerance_loss = 1e-05,
                double tolerance_change = 1e-08,
                MelNorm norm = MelNorm.none,
                torchaudio.MelScale mel_scale = torchaudio.MelScale.htk)
            {
                return new Transforms.InverseMelScale(
                    "InverseMelScale",
                    n_stft,
                    n_mels,
                    sample_rate,
                    f_min,
                    f_max,
                    max_iter,
                    tolerance_loss,
                    tolerance_change,
                    norm,
                    mel_scale);
            }
        }
    }
}