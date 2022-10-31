// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using static TorchSharp.torch;
using System.Diagnostics;

// A number of implementation details in this file have been translated from the Python version of torchaudio,
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

            private static ScalarType _get_complex_dtype(torch.ScalarType real_dtype)
            {
                if (real_dtype == ScalarType.Float64)
                    return ScalarType.ComplexFloat64;
                if (real_dtype == ScalarType.Float32)
                    return ScalarType.ComplexFloat32;
                if (real_dtype == ScalarType.Float16)
                    return ScalarType.ComplexFloat32;
                throw new ArgumentException($"Unexpected dtype {real_dtype}");
            }

            /// <summary>
            /// Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.
            /// </summary>
            /// <param name="specgram">A magnitude-only STFT spectrogram of dimension `(..., freq, frames)` where freq is ``n_fft // 2 + 1``.</param>
            /// <param name="window">Window tensor that is applied/multiplied to each frame/window</param>
            /// <param name="n_fft">Size of FFT, creates ``n_fft // 2 + 1`` bins</param>
            /// <param name="hop_length">Length of hop between STFT windows.</param>
            /// <param name="win_length">Window size.</param>
            /// <param name="power">Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for energy, 2 for power, etc.</param>
            /// <param name="n_iter">Number of iteration for phase recovery process.</param>
            /// <param name="momentum">The momentum parameter for fast Griffin-Lim.</param>
            /// <param name="length">Array length of the expected output.</param>
            /// <param name="rand_init">Initializes phase randomly if True, to zero otherwise.</param>
            /// <returns></returns>
            /// <exception cref="ArgumentOutOfRangeException"></exception>
            public static Tensor griffinlim(Tensor specgram, Tensor window, long n_fft, long hop_length, long win_length, double power, int n_iter, double momentum, long? length, bool rand_init)
            {
                if (momentum < 0.0 || 1.0 <= momentum) {
                    throw new ArgumentOutOfRangeException($"momentum must be in range [0, 1). Found: {momentum}");
                }
                momentum = momentum / (1 + momentum);

                // pack batch
                var shape = specgram.size();
                specgram = specgram.reshape(new long[] { -1, shape[shape.Length - 2], shape[shape.Length - 1] });

                specgram = specgram.pow(1 / power);

                // initialize the phase
                Tensor angles;
                if (rand_init) {
                    angles = torch.rand(specgram.size(), dtype: _get_complex_dtype(specgram.dtype), device: specgram.device);
                } else {
                    angles = torch.full(specgram.size(), 1, dtype: _get_complex_dtype(specgram.dtype), device: specgram.device);
                }

                // And initialize the previous iterate to 0
                var tprev = torch.tensor(0.0, dtype: specgram.dtype, device: specgram.device);
                for (int i = 0; i < n_iter; i++) {
                    // Invert with our current estimate of the phases
                    var inverse = torch.istft(
                        specgram * angles, n_fft: n_fft, hop_length: hop_length, win_length: win_length, window: window, length: length ?? -1
                    );

                    // Rebuild the spectrogram
                    var rebuilt = torch.stft(
                        input: inverse,
                        n_fft: n_fft,
                        hop_length: hop_length,
                        win_length: win_length,
                        window: window,
                        center: true,
                        pad_mode: PaddingModes.Reflect,
                        normalized: false,
                        onesided: true,
                        return_complex: true);

                    // Update our phase estimates
                    angles = rebuilt;
                    if (momentum > 0.0) {
                        angles = angles - tprev.mul_(momentum);
                    }
                    angles = angles.div(angles.abs().add(1e-16));

                    // Store the previous iterate
                    tprev = rebuilt;
                }

                // Return the final phase estimates
                var waveform = torch.istft(
                    specgram * angles, n_fft: n_fft, hop_length: hop_length, win_length: win_length, window: window, length: length ?? -1
                );

                // unpack batch
                var new_shape = new long[shape.Length - 1];
                Array.Copy(shape, new_shape, shape.Length - 2);
                new_shape[new_shape.Length - 1] = waveform.shape[waveform.dim() - 1];
                waveform = waveform.reshape(new_shape);

                return waveform;
            }

            /// <summary>
            /// Turn a spectrogram from the power/amplitude scale to the decibel scale.
            /// </summary>
            /// <param name="x">Input spectrogram(s) before being converted to decibel scale.</param>
            /// <param name="multiplier">Use 10. for power and 20. for amplitude</param>
            /// <param name="amin">Number to clamp x</param>
            /// <param name="db_multiplier">Log10(max(reference value and amin))</param>
            /// <param name="top_db">Minimum negative cut-off in decibels.</param>
            /// <returns>Output tensor in decibel scale</returns>
            public static Tensor amplitude_to_DB(Tensor x, double multiplier, double amin, double db_multiplier, double? top_db = null)
            {
                var x_db = multiplier * torch.log10(torch.clamp(x, min: amin));
                x_db -= multiplier * db_multiplier;

                if (top_db != null) {
                    // Expand batch
                    var shape = x_db.size();
                    var packed_channels = x_db.dim() > 2 ? shape[shape.Length - 3] : 1;
                    x_db = x_db.reshape(-1, packed_channels, shape[shape.Length - 2], shape[shape.Length - 1]);

                    x_db = torch.maximum(x_db, (x_db.amax(dims: new long[] { -3, -2, -1 }) - top_db).view(-1, 1, 1, 1));

                    // Repack batch
                    x_db = x_db.reshape(shape);
                }
                return x_db;
            }

            /// <summary>
            /// Turn a tensor from the decibel scale to the power/amplitude scale.
            /// </summary>
            /// <param name="x">Input tensor before being converted to power/amplitude scale.</param>
            /// <param name="ref">Reference which the output will be scaled by.</param>
            /// <param name="power">If power equals 1, will compute DB to power. If 0.5, will compute DB to amplitude.</param>
            /// <returns>Output tensor in power/amplitude scale.</returns>
            public static Tensor DB_to_amplitude(Tensor x, double @ref, double power)
            {
                return @ref * torch.pow(torch.pow(10.0, 0.1 * x), power);
            }

            private static double _hz_to_mel(double freq, MelScale mel_scale = MelScale.htk)
            {
                if (mel_scale == MelScale.htk) {
                    return 2595.0 * Math.Log10(1.0 + freq / 700.0);
                }

                // Fill in the linear part
                var f_min = 0.0;
                var f_sp = 200.0 / 3;

                var mels = (freq - f_min) / f_sp;

                // Fill in the log-scale part
                var min_log_hz = 1000.0;
                var min_log_mel = (min_log_hz - f_min) / f_sp;
                var logstep = Math.Log(6.4) / 27.0;

                if (freq >= min_log_hz) {
                    mels = min_log_mel + Math.Log(freq / min_log_hz) / logstep;
                }

                return mels;
            }

            private static Tensor _mel_to_hz(Tensor mels, MelScale mel_scale = MelScale.htk)
            {
                if (mel_scale == MelScale.htk) {
                    return 700.0 * (torch.pow(10.0, mels / 2595.0) - 1.0);
                }

                // Fill in the linear scale
                var f_min = 0.0;
                var f_sp = 200.0 / 3;

                var freqs = f_min + f_sp * mels;

                // And now the nonlinear scale
                var min_log_hz = 1000.0;
                var min_log_mel = (min_log_hz - f_min) / f_sp;
                var logstep = Math.Log(6.4) / 27.0;

                var log_t = mels >= min_log_mel;
                freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel));

                return freqs;
            }

            private static Tensor _create_triangular_filterbank(Tensor all_freqs, Tensor f_pts)
            {
                // Adopted from Librosa
                // calculate the difference between each filter mid point and each stft freq point in hertz
                var f_diff = f_pts[TensorIndex.Slice(1, null)] - f_pts[TensorIndex.Slice(null, -1)];  // (n_filter + 1)
                var slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1);  // (n_freqs, n_filter + 2)
                // create overlapping triangles
                var zero = torch.zeros(1);
                var down_slopes = (-1.0 * slopes[TensorIndex.Colon, TensorIndex.Slice(null, -2)]) / f_diff[TensorIndex.Slice(null, -1)];  // (n_freqs, n_filter)
                var up_slopes = slopes[TensorIndex.Colon, TensorIndex.Slice(2, null)] / f_diff[TensorIndex.Slice(1, null)];  // (n_freqs, n_filter)
                var fb = torch.maximum(zero, torch.minimum(down_slopes, up_slopes));

                return fb;
            }

            /// <summary>
            /// Create a frequency bin conversion matrix.
            /// </summary>
            /// <param name="n_freqs">Number of frequencies to highlight/apply</param>
            /// <param name="f_min">Minimum frequency(Hz)</param>
            /// <param name="f_max">Maximum frequency(Hz)</param>
            /// <param name="n_mels">Number of mel filterbanks</param>
            /// <param name="sample_rate">Sample rate of the audio waveform</param>
            /// <param name="norm">If MelNorm.slaney, divide the triangular mel weights by the width of the mel band</param>
            /// <param name="mel_scale">Scale to use</param>
            /// <returns>Triangular filter banks</returns>
            public static Tensor melscale_fbanks(long n_freqs, double f_min, double f_max, long n_mels, long sample_rate, MelNorm norm = MelNorm.none, MelScale mel_scale = MelScale.htk)
            {
                // freq bins
                var all_freqs = torch.linspace(0, sample_rate / 2, n_freqs);

                // calculate mel freq bins
                var m_min = _hz_to_mel(f_min, mel_scale: mel_scale);
                var m_max = _hz_to_mel(f_max, mel_scale: mel_scale);

                var m_pts = torch.linspace(m_min, m_max, n_mels + 2);
                var f_pts = _mel_to_hz(m_pts, mel_scale: mel_scale);

                // create filterbank
                var fb = _create_triangular_filterbank(all_freqs, f_pts);

                if (norm == MelNorm.slaney) {
                    // Slaney-style mel is scaled to be approx constant energy per channel
                    var enorm = 2.0 / (f_pts[TensorIndex.Slice(2, n_mels + 2)] - f_pts[TensorIndex.Slice(null, n_mels)]);
                    fb *= enorm.unsqueeze(0);
                }

                if ((fb.max(dim: 0).values == 0.0).any().item<bool>()) {
                    Debug.Print(
                        "At least one mel filterbank has all zero values. " +
                        $"The value for `n_mels` ({n_mels}) may be set too high. " +
                        $"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
                    );
                }

                return fb;
            }

            /// <summary>
            /// Creates a linear triangular filterbank.
            /// </summary>
            /// <param name="n_freqs">Number of frequencies to highlight/apply</param>
            /// <param name="f_min">Minimum frequency (Hz)</param>
            /// <param name="f_max">Maximum frequency (Hz)</param>
            /// <param name="n_filter">Number of (linear) triangular filter</param>
            /// <param name="sample_rate">Sample rate of the audio waveform</param>
            /// <returns>Triangular filter banks</returns>
            public static Tensor linear_fbanks(int n_freqs, double f_min, double f_max, int n_filter, int sample_rate)
            {
                // freq bins
                var all_freqs = torch.linspace(0, sample_rate / 2, n_freqs);

                // filter mid-points
                var f_pts = torch.linspace(f_min, f_max, n_filter + 2);

                // create filterbank
                var fb = _create_triangular_filterbank(all_freqs, f_pts);

                return fb;
            }

            /// <summary>
            /// Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``)
            /// </summary>
            /// <param name="n_mfcc">Number of mfc coefficients to retain</param>
            /// <param name="n_mels">Number of mel filterbanks</param>
            /// <param name="norm">Norm to use</param>
            /// <returns>The transformation matrix</returns>
            public static Tensor create_dct(int n_mfcc, int n_mels, DCTNorm norm)
            {
                // http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
                var n = torch.arange((float)n_mels);
                var k = torch.arange((float)n_mfcc).unsqueeze(1);
                var dct = torch.cos(Math.PI / n_mels * (n + 0.5) * k);

                if (norm == DCTNorm.none) {
                    dct *= 2.0;
                } else {
                    dct[0] *= 1.0 / Math.Sqrt(2.0);
                    dct *= Math.Sqrt(2.0 / n_mels);
                }
                return dct.t();
            }

            /// <summary>
            /// Encode signal based on mu-law companding.
            /// </summary>
            /// <param name="x">Input tensor</param>
            /// <param name="quantization_channels">Number of channels</param>
            /// <returns>Input after mu-law encoding</returns>
            /// <exception cref="ArgumentException"></exception>
            public static Tensor mu_law_encoding(Tensor x, int quantization_channels)
            {
                if (!x.is_floating_point()) {
                    throw new ArgumentException("The input Tensor must be of floating type. ");
                }
                var mu = torch.tensor(quantization_channels - 1.0, dtype: x.dtype);
                var x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu);
                x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64);
                return x_mu;
            }

            /// <summary>
            /// Decode mu-law encoded signal.
            /// </summary>
            /// <param name="x_mu">Input tensor</param>
            /// <param name="quantization_channels">Number of channels</param>
            /// <returns>Input after mu-law decoding</returns>
            public static Tensor mu_law_decoding(Tensor x_mu, int quantization_channels)
            {
                if (!x_mu.is_floating_point()) {
                    x_mu = x_mu.to(torch.@float);
                }
                var mu = torch.tensor(quantization_channels - 1.0, dtype: x_mu.dtype);
                var x = x_mu / mu * 2 - 1.0;
                x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu;
                return x;
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