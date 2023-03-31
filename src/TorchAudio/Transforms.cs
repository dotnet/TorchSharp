// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using TorchSharp.Transforms;

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
            public static Spectrogram Spectrogram(
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
                    "Spectrogram",
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
            public static InverseSpectrogram InverseSpectrogram(
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
                    "InverseSpectrogram",
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
            public static Resample Resample(
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
                    "Resample",
                    orig_freq,
                    new_freq,
                    resampling_method,
                    lowpass_filter_width,
                    rolloff,
                    beta,
                    device,
                    dtype);
            }

            /// <summary>
            /// Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.
            /// </summary>
            /// <param name="n_fft">Size of FFT, creates ``n_fft // 2 + 1`` bins.</param>
            /// <param name="n_iter">Number of iteration for phase recovery process.</param>
            /// <param name="win_length">Window size.</param>
            /// <param name="hop_length">Length of hop between STFT windows.</param>
            /// <param name="window_fn">A function to create a window tensor
            /// that is applied/multiplied to each frame/window.</param>
            /// <param name="power">Exponent for the magnitude spectrogram,
            /// (must be > 0) e.g., 1 for energy, 2 for power, etc.</param>
            /// <param name="momentum">The momentum parameter for fast Griffin-Lim.</param>
            /// <param name="length">Array length of the expected output.</param>
            /// <param name="rand_init">Initializes phase randomly if True and to zero otherwise.</param>
            /// <returns></returns>
            public static GriffinLim GriffinLim(
                int n_fft = 400,
                int n_iter = 32,
                long? win_length = null,
                long? hop_length = null,
                WindowFunction window_fn = null,
                double power = 2.0,
                double momentum = 0.99,
                int? length = null,
                bool rand_init = true)
            {
                return new GriffinLim(
                    "GriffinLim",
                    n_fft,
                    n_iter,
                    win_length,
                    hop_length,
                    window_fn,
                    power,
                    momentum,
                    length,
                    rand_init);
            }
        }
    }
}