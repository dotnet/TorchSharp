// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

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
    public sealed class Resample : nn.Module<Tensor, Tensor>, ITransform
    {
        private readonly int orig_freq;
        private readonly int new_freq;
        private readonly int gcd;
        private readonly ResamplingMethod resampling_method;
        private readonly int lowpass_filter_width;
        private readonly double rolloff;
        private readonly double? beta;
        public readonly torch.Tensor kernel;
        private readonly int width;

        internal Resample(
            string name,
            int orig_freq = 16000,
            int new_freq = 16000,
            ResamplingMethod resampling_method = ResamplingMethod.sinc_interpolation,
            int lowpass_filter_width = 6,
            double rolloff = 0.99,
            double? beta = null,
            torch.Device device = null,
            torch.ScalarType? dtype = null) : base(name)
        {
            this.orig_freq = orig_freq;
            this.new_freq = new_freq;
            this.gcd = functional.Gcd(this.orig_freq, this.new_freq);
            this.resampling_method = resampling_method;
            this.lowpass_filter_width = lowpass_filter_width;
            this.rolloff = rolloff;
            this.beta = beta;

            if (this.orig_freq != this.new_freq) {
                (this.kernel, this.width) = functional._get_sinc_resample_kernel(
                    this.orig_freq,
                    this.new_freq,
                    this.gcd,
                    this.lowpass_filter_width,
                    this.rolloff,
                    this.resampling_method,
                    beta,
                    device: device,
                    dtype: dtype);
            }
        }

        public override Tensor forward(Tensor waveform)
        {
            using (var d = torch.NewDisposeScope()) {

                if (this.orig_freq == this.new_freq) {
                    return d.MoveToOuter(waveform.alias());
                }
                var resampled = functional._apply_sinc_resample_kernel(waveform, this.orig_freq, this.new_freq, this.gcd, this.kernel, this.width);
                return d.MoveToOuter(resampled);
            }
        }
    }
}
