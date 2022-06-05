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
        }
    }
}