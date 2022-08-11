// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using static TorchSharp.torch;

// A number of implementation details in this file have been translated from the Python version of torchaudio,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/audio/blob/bb77cbebb620a46fdc0dc7e6dae2253eef3f37e2/torchaudio/transforms/_transforms.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/audio/blob/main/LICENSE
//

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public delegate torch.Tensor WindowFunction(long win_length);

        internal class Spectrogram : ITransform
        {
            private readonly long n_fft;
            private readonly long win_length;
            private readonly long hop_length;
            private readonly long pad;
            private readonly Tensor window;
            private readonly double? power;
            private readonly bool normalized;
            private readonly bool center;
            private readonly PaddingModes pad_mode;
            private readonly bool onesided;

            public Spectrogram(
                long n_fft = 400,
                long? win_length = null,
                long? hop_length = null,
                long pad = 0,
                WindowFunction window_fn = null,
                Tensor window = null,
                double? power = 2.0,
                bool normalized = false,
                bool center = true,
                PaddingModes pad_mode = PaddingModes.Reflect,
                bool onesided = true,
                bool? return_complex = null)
            {
                this.n_fft = n_fft;
                if (win_length.HasValue) {
                    this.win_length = win_length.Value;
                } else {
                    this.win_length = n_fft;
                }
                if (hop_length.HasValue) {
                    this.hop_length = hop_length.Value;
                } else {
                    this.hop_length = this.win_length / 2;
                }
                this.pad = pad;
                if (window is not null) {
                    this.window = window;
                } else if (window_fn != null) {
                    this.window = window_fn(this.win_length);
                } else {
                    this.window = torch.hann_window(this.win_length);
                }
                this.power = power;
                this.normalized = normalized;
                this.center = center;
                this.pad_mode = pad_mode;
                this.onesided = onesided;
                if (return_complex.HasValue) {
                    Console.WriteLine(
                        "`return_complex` argument is now deprecated and is not effective." +
                        "`torchaudio.transforms.Spectrogram(power=null)` always returns a tensor with " +
                        "complex dtype. Please remove the argument in the function call."
                    );
                }
            }

            public Tensor forward(Tensor input)
            {
                return torchaudio.functional.spectrogram(
                    waveform: input,
                    pad: pad,
                    window: window,
                    n_fft: n_fft,
                    hop_length: hop_length,
                    win_length: win_length,
                    power: power,
                    normalized: normalized,
                    center: center,
                    pad_mode: pad_mode,
                    onesided: onesided);
            }
        }

        internal class InverseSpectrogram : ITransform
        {
            private readonly long n_fft;
            private readonly long win_length;
            private readonly long hop_length;
            private readonly long pad;
            private readonly Tensor window;
            private readonly bool normalized;
            private readonly bool center;
            private readonly PaddingModes pad_mode;
            private readonly bool onesided;

            public InverseSpectrogram(
                long n_fft = 400,
                long? win_length = null,
                long? hop_length = null,
                long pad = 0,
                WindowFunction window_fn = null,
                Tensor window = null,
                bool normalized = false,
                bool center = true,
                PaddingModes pad_mode = PaddingModes.Reflect,
                bool onesided = true)
            {
                this.n_fft = n_fft;
                if (win_length.HasValue) {
                    this.win_length = win_length.Value;
                } else {
                    this.win_length = n_fft;
                }
                if (hop_length.HasValue) {
                    this.hop_length = hop_length.Value;
                } else {
                    this.hop_length = this.win_length / 2;
                }
                this.pad = pad;
                if (window is not null) {
                    this.window = window;
                } else if (window_fn != null) {
                    this.window = window_fn(this.win_length);
                } else {
                    this.window = torch.hann_window(this.win_length);
                }
                this.normalized = normalized;
                this.center = center;
                this.pad_mode = pad_mode;
                this.onesided = onesided;
            }

            public Tensor forward(Tensor input)
            {
                return forward(input, null);
            }

            public Tensor forward(Tensor input, long? length = null)
            {
                return torchaudio.functional.inverse_spectrogram(
                    spectrogram: input,
                    length: length,
                    pad: pad,
                    window: window,
                    n_fft: n_fft,
                    hop_length: hop_length,
                    win_length: win_length,
                    normalized: normalized,
                    center: center,
                    pad_mode: pad_mode,
                    onesided: onesided);
            }
        }

        internal sealed class Resample : ITransform
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

            public Resample(
                int orig_freq = 16000,
                int new_freq = 16000,
                ResamplingMethod resampling_method = ResamplingMethod.sinc_interpolation,
                int lowpass_filter_width = 6,
                double rolloff = 0.99,
                double? beta = null,
                torch.Device device = null,
                torch.ScalarType? dtype = null)
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

            public torch.Tensor forward(torch.Tensor waveform)
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

        public static partial class transforms
        {
            /// <summary>
            /// Compute spectrograms from audio signals.
            /// </summary>
            /// <param name="n_fft">The size of Fourier transform</param>
            /// <param name="hop_length">The hop length</param>
            /// <param name="win_length">The window length</param>
            /// <param name="pad">Padding on the sides</param>
            /// <param name="window_fn">The callback to create a window function</param>
            /// <param name="window">The window function</param>
            /// <param name="power">Exponent for the magnitude spectrogram</param>
            /// <param name="normalized">Whether the output is normalized, or not.</param>
            /// <param name="center">Whether the t-th frame is centered around t * hop_window, or not.</param>
            /// <param name="pad_mode">The padding mode used when center is true.</param>
            /// <param name="onesided">Whether the output is onesided or not.</param>
            /// <param name="return_complex">Deprecated and not used.</param>
            /// <returns>ITransform to compute spectrograms of audio signals</returns>
            public static ITransform Spectrogram(
                long n_fft = 400,
                long? win_length = null,
                long? hop_length = null,
                long pad = 0,
                WindowFunction window_fn = null,
                Tensor window = null,
                double? power = 2.0,
                bool normalized = false,
                bool center = true,
                PaddingModes pad_mode = PaddingModes.Reflect,
                bool onesided = true,
                bool? return_complex = null)
            {
                return new Spectrogram(
                    n_fft: n_fft,
                    hop_length: hop_length,
                    win_length: win_length,
                    pad: pad,
                    window_fn: window_fn,
                    window: window,
                    power: power,
                    normalized: normalized,
                    center: center,
                    pad_mode: pad_mode,
                    onesided: onesided,
                    return_complex: return_complex);
            }

            /// <summary>
            /// Compute inverse of spectrogram.
            /// </summary>
            /// <param name="n_fft">The size of Fourier transform</param>
            /// <param name="hop_length">The hop length</param>
            /// <param name="win_length">The window length</param>
            /// <param name="pad">Padding on the sides</param>
            /// <param name="window_fn">The callback to create a window function</param>
            /// <param name="window">The window function</param>
            /// <param name="normalized">Whether the output is normalized, or not.</param>
            /// <param name="center">Whether the t-th frame is centered around t * hop_window, or not.</param>
            /// <param name="pad_mode">The padding mode used when center is true.</param>
            /// <param name="onesided">Whether the output is onesided or not.</param>
            /// <returns>ITransform to compute inverse of spectrogram</returns>
            public static ITransform InverseSpectrogram(
                long n_fft = 400,
                long? win_length = null,
                long? hop_length = null,
                long pad = 0,
                WindowFunction window_fn = null,
                Tensor window = null,
                bool normalized = false,
                bool center = true,
                PaddingModes pad_mode = PaddingModes.Reflect,
                bool onesided = true)
            {
                return new InverseSpectrogram(
                    n_fft: n_fft,
                    hop_length: hop_length,
                    win_length: win_length,
                    pad: pad,
                    window_fn: window_fn,
                    window: window,
                    normalized: normalized,
                    center: center,
                    pad_mode: pad_mode,
                    onesided: onesided);
            }

            /// <summary>
            /// Resample the waveform
            /// </summary>
            /// <param name="orig_freq">The source sampling rate</param>
            /// <param name="new_freq">The destination sampling rate</param>
            /// <param name="lowpass_filter_width">The width of the filter</param>
            /// <param name="rolloff">The roll-off frequency</param>
            /// <param name="resampling_method">The resampling method</param>
            /// <param name="beta">Beta for Keizer window</param>
            /// <param name="device">The device</param>
            /// <param name="dtype">The scalar type</param>
            /// <returns>The resampled waveform</returns>
            /// <exception cref="ArgumentOutOfRangeException"></exception>
            public static ITransform Resample(
                int orig_freq = 16000,
                int new_freq = 16000,
                ResamplingMethod resampling_method = ResamplingMethod.sinc_interpolation,
                int lowpass_filter_width = 6,
                double rolloff = 0.99,
                double? beta = null,
                torch.Device device = null,
                torch.ScalarType? dtype = null)
            {
                return new Resample(
                    orig_freq,
                    new_freq,
                    resampling_method,
                    lowpass_filter_width,
                    rolloff,
                    beta,
                    device,
                    dtype);
            }
        }
    }
}