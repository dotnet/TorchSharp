// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using static TorchSharp.torch;

// A number of implementation details in this file have been translated from the Python version or torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/audio/blob/39c2c0a77e11b82ce152d60df3a92f6854d5f52b/torchaudio/functional/functional.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/audio/blob/main/LICENSE
//

namespace TorchSharp
{
    public static partial class torchaudio
    {
        public static partial class functional
        {
            /// <summary>
            /// Compute spectrograms from audio signals.
            /// </summary>
            /// <param name="waveform">The audio signal tensor</param>
            /// <param name="pad">Padding on the sides</param>
            /// <param name="window">The window function</param>
            /// <param name="n_fft">The size of Fourier transform</param>
            /// <param name="hop_length">The hop length</param>
            /// <param name="win_length">The window length</param>
            /// <param name="power">Exponent for the magnitude spectrogram</param>
            /// <param name="normalized">Whether the output is normalized, or not.</param>
            /// <param name="center">Whether the t-th frame is centered around t * hop_window, or not.</param>
            /// <param name="pad_mode">The padding mode used when center is true.</param>
            /// <param name="onesided">Whether the output is onesided or not.</param>
            /// <param name="return_complex">Deprecated and not used.</param>
            /// <returns>Spectrograms of audio signals</returns>
            public static torch.Tensor spectrogram(torch.Tensor waveform, long pad, torch.Tensor window, long n_fft, long hop_length, long win_length, double? power, bool normalized, bool center = true, PaddingModes pad_mode = PaddingModes.Reflect, bool onesided = true, bool? return_complex = null)
            {
                using (var d = torch.NewDisposeScope()) {

                    if (pad > 0) {
                        // The original torchaudio doesn't have `torch.no_grad()' here to
                        // avoid issues with JIT.
                        // https://github.com/pytorch/audio/commit/a420cced7e60fcf9ba6efcff3a2e8bee3ac67d05#diff-ab14255549624af556aa0d1dfaf83241a95e05c3dd6a668cd2607655839f7a09
                        using (torch.no_grad()) {
                            waveform = torch.nn.functional.pad(waveform, new long[] { pad, pad }, PaddingModes.Constant);
                        }
                    }

                    // pack batch
                    var shape = waveform.size();
                    waveform = waveform.reshape(-1, shape[shape.Length - 1]);

                    // default values are consistent with librosa.core.spectrum._spectrogram
                    var spec_f = torch.stft(
                        input: waveform,
                        n_fft: n_fft,
                        hop_length: hop_length,
                        win_length: win_length,
                        window: window,
                        center: center,
                        pad_mode: pad_mode,
                        normalized: false,
                        onesided: onesided,
                        return_complex: true);

                    // unpack batch
                    var spec_shape = new long[shape.Length + spec_f.dim() - 2];
                    Array.Copy(shape, spec_shape, shape.Length - 1);
                    Array.Copy(spec_f.shape, 1, spec_shape, shape.Length - 1, spec_f.dim() - 1);
                    spec_f = spec_f.reshape(spec_shape);

                    if (normalized) {
                        spec_f /= window.pow(2.0).sum().sqrt();
                    }

                    if (power.HasValue) {
                        if (power.Value == 1.0) {
                            spec_f = spec_f.abs();
                        } else {
                            spec_f = spec_f.abs().pow(power.Value);
                        }
                    }

                    return d.MoveToOuter(spec_f);
                }
            }

            /// <summary>
            /// Compute inverse of spectrogram.
            /// </summary>
            /// <param name="spectrogram">The spectrogram tensor</param>
            /// <param name="length">The length of the output tensor</param>
            /// <param name="pad">Padding on the sides</param>
            /// <param name="window">The window function</param>
            /// <param name="n_fft">The size of Fourier transform</param>
            /// <param name="hop_length">The hop length</param>
            /// <param name="win_length">The window length</param>
            /// <param name="normalized">Whether the output is normalized, or not.</param>
            /// <param name="center">Whether the t-th frame is centered around t * hop_window, or not.</param>
            /// <param name="pad_mode">The padding mode used when center is true.</param>
            /// <param name="onesided">Whether the output is onesided or not.</param>
            /// <returns>Inverse of spectrogram</returns>
            public static torch.Tensor inverse_spectrogram(torch.Tensor spectrogram, long? length, long pad, torch.Tensor window, long n_fft, long hop_length, long win_length, bool normalized, bool center = true, PaddingModes pad_mode = PaddingModes.Reflect, bool onesided = true)
            {
                if (!spectrogram.is_complex()) {
                    throw new ArgumentException("Expected `spectrogram` to be complex dtype.");
                }

                using (var d = torch.NewDisposeScope()) {

                    if (normalized) {
                        spectrogram = spectrogram * window.pow(2.0).sum().sqrt();
                    }

                    // pack batch
                    var shape = spectrogram.size();
                    spectrogram = spectrogram.reshape(-1, shape[shape.Length - 2], shape[shape.Length - 1]);

                    // default values are consistent with librosa.core.spectrum._spectrogram
                    var waveform = torch.istft(
                        input: spectrogram,
                        n_fft: n_fft,
                        hop_length: hop_length,
                        win_length: win_length,
                        window: window,
                        center: center,
                        normalized: false,
                        onesided: onesided,
                        length: length.HasValue ? length.Value + 2 * pad : -1,
                        return_complex: false
                    );

                    if (length.HasValue && pad > 0) {
                        // remove padding from front and back
                        waveform = waveform[TensorIndex.Colon, TensorIndex.Slice(pad, -pad)];
                    }

                    // unpack batch
                    var waveform_shape = new long[shape.Length - 1];
                    Array.Copy(shape, waveform_shape, shape.Length - 2);
                    waveform_shape[waveform_shape.Length - 1] = waveform.shape[waveform.dim() - 1];
                    waveform = waveform.reshape(waveform_shape);

                    return d.MoveToOuter(waveform);
                }
            }

            /// <summary>
            /// Resample the waveform
            /// </summary>
            /// <param name="waveform">The input waveform</param>
            /// <param name="orig_freq">The source sampling rate</param>
            /// <param name="new_freq">The destination sampling rate</param>
            /// <param name="lowpass_filter_width">The width of the filter</param>
            /// <param name="rolloff">The roll-off frequency</param>
            /// <param name="resampling_method">The resampling method</param>
            /// <param name="beta">Beta for Keizer window</param>
            /// <returns>The resampled waveform</returns>
            /// <exception cref="ArgumentOutOfRangeException"></exception>
            public static torch.Tensor resample(torch.Tensor waveform, int orig_freq, int new_freq, int lowpass_filter_width = 6, double rolloff = 0.99, ResamplingMethod resampling_method = ResamplingMethod.sinc_interpolation, double? beta = null)
            {
                if (orig_freq <= 0 || new_freq <= 0) {
                    throw new ArgumentOutOfRangeException();
                }

                using (var d = torch.NewDisposeScope()) {
                    if (orig_freq == new_freq) {
                        return d.MoveToOuter(waveform.alias());
                    }

                    int gcd = Gcd(orig_freq, new_freq);

                    var (kernel, width) = _get_sinc_resample_kernel(
                        orig_freq,
                        new_freq,
                        gcd,
                        lowpass_filter_width,
                        rolloff,
                        resampling_method,
                        beta,
                        waveform.device,
                        waveform.dtype);
                    var resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width);
                    return d.MoveToOuter(resampled);
                }
            }

            internal static int Gcd(int a, int b)
            {
                if (a <= 0 || b <= 0) {
                    throw new ArgumentOutOfRangeException();
                }

                while (b > 1) {
                    (a, b) = (b, a % b);
                }

                return a;
            }

            internal static (torch.Tensor, int) _get_sinc_resample_kernel(int orig_freq, int new_freq, int gcd, int lowpass_filter_width = 6, double rolloff = 0.99, ResamplingMethod resampling_method = ResamplingMethod.sinc_interpolation, double? beta = null, torch.Device device = null, torch.ScalarType? dtype = null)
            {
                orig_freq = orig_freq / gcd;
                new_freq = new_freq / gcd;

                if (lowpass_filter_width <= 0) {
                    throw new ArgumentOutOfRangeException();
                }

                var kernels_list = new List<torch.Tensor>();
                double base_freq = Math.Min(orig_freq, new_freq);
                base_freq *= rolloff;

                var width = (int)Math.Ceiling(((double)lowpass_filter_width) * orig_freq / base_freq);
                var idx_dtype = dtype ?? torch.float64;
                var idx = torch.arange(-width, width + orig_freq, device: device, dtype: idx_dtype);

                for (int i = 0; i < new_freq; i++) {
                    var t = (-i / new_freq + idx / orig_freq) * base_freq;
                    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width);

                    torch.Tensor window;
                    if (resampling_method == ResamplingMethod.sinc_interpolation) {
                        window = torch.square(torch.cos(t * Math.PI / lowpass_filter_width / 2));
                    } else {
                        // kaiser_window
                        if (!beta.HasValue) {
                            beta = 14.769656459379492;
                        }
                        var beta_tensor = torch.tensor(beta.Value);
                        window = torch.special.i0(beta_tensor * torch.sqrt(1 - torch.square(t / lowpass_filter_width))) / torch.special.i0(beta_tensor);
                    }
                    t *= Math.PI;
                    // Tensor.to(Tensor) of TorchSharp desn't change dtype.
                    var kernel = torch.where(t == 0, torch.tensor(1.0).to(t).type_as(t), torch.sin(t) / t);
                    kernel.mul_(window);
                    kernels_list.Add(kernel);
                }

                var scale = ((double)base_freq) / orig_freq;
                var kernels = torch.stack(kernels_list.ToArray()).view(new_freq, 1, -1).mul_(scale);
                if (dtype == null) {
                    kernels = kernels.to(torch.float32);
                }
                return (kernels, width);
            }

            internal static torch.Tensor _apply_sinc_resample_kernel(torch.Tensor waveform, int orig_freq, int new_freq, int gcd, torch.Tensor kernel, int width)
            {
                if (!waveform.is_floating_point()) {
                    throw new ArgumentException($"Expected floating point type for waveform tensor, but received {waveform.dtype}.");
                }

                orig_freq = orig_freq / gcd;
                new_freq = new_freq / gcd;

                // pack batch
                var shape = waveform.size();
                waveform = waveform.view(-1, shape[waveform.dim() - 1]);

                var num_wavs = waveform.shape[0];
                var length = waveform.shape[1];

                waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq));
                var resampled = torch.nn.functional.conv1d(waveform.unsqueeze(1), kernel, stride: orig_freq);
                resampled = resampled.transpose(1, 2).reshape(num_wavs, -1);
                int target_length = (int)Math.Ceiling(((double)new_freq) * length / orig_freq);
                resampled = resampled[TensorIndex.Ellipsis, TensorIndex.Slice(0, target_length)];

                // unpack batch
                shape[shape.Length - 1] = resampled.shape[resampled.dim() - 1];
                resampled = resampled.view(shape);
                return resampled;
            }
        }
    }
}