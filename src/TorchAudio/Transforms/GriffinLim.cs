// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.torchaudio;

// A number of implementation details in this file have been translated from the Python version of torchaudio,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/audio/blob/bb77cbebb620a46fdc0dc7e6dae2253eef3f37e2/torchaudio/transforms/_transforms.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/audio/blob/main/LICENSE
//

namespace TorchSharp.Transforms
{
    public sealed class GriffinLim : nn.Module<Tensor, Tensor>, ITransform
    {
        public readonly long n_fft;
        public readonly int n_iter;
        public readonly long win_length;
        public readonly long hop_length;
        public readonly Tensor window;
        public readonly long? length;
        public readonly double power;
        public readonly double momentum;
        public readonly bool rand_init;

        protected override void Dispose(bool disposing)
        {
            if (disposing) {
                window.Dispose();
            }
            base.Dispose(disposing);
        }

        internal GriffinLim(
            string name,
            long n_fft = 400,
            int n_iter = 32,
            long? win_length = null,
            long? hop_length = null,
            WindowFunction window_fn = null,
            double power = 2.0,
            double momentum = 0.99,
            long? length = null,
            bool rand_init = true) : base(name)
        {
            if (momentum < 0 || 1 <= momentum) {
                throw new ArgumentOutOfRangeException($"momentum must be in the range [0, 1). Found: {momentum}");
            }

            this.n_fft = n_fft;
            this.n_iter = n_iter;
            this.win_length = win_length ?? n_fft;
            this.hop_length = hop_length ?? this.win_length / 2;
            if (window_fn != null) {
                this.window = window_fn(this.win_length);
            } else {
                this.window = torch.hann_window(this.win_length);
            }
            this.register_buffer("window", this.window);
            this.length = length;
            this.power = power;
            this.momentum = momentum;
            this.rand_init = rand_init;
        }

        public override Tensor forward(Tensor specgram)
        {
            return torchaudio.functional.griffinlim(
                specgram,
                this.window,
                this.n_fft,
                this.hop_length,
                this.win_length,
                this.power,
                this.n_iter,
                this.momentum,
                this.length,
                this.rand_init
            );
        }
    }
}
