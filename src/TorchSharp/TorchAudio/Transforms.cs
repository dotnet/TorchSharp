// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using static TorchSharp.torch;

// A number of implementation details in this file have been translated from the Python version or torchvision,
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
        public interface ITransform
        {
            Tensor forward(Tensor input);
        }

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

        public static partial class transforms
        {
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
        }
    }
}