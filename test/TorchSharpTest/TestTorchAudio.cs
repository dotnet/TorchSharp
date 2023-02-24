using System;
using Xunit;

namespace TorchSharp
{
    public class TestTorchAudio
    {
        private torch.Tensor make_waveform()
        {
            var fc = 220;
            var fm = 440;
            var beta = 1;
            var t = torch.linspace(0, 5, 16000 * 5);
            var waveform = 1.0 * torch.sin(2 * Math.PI * t * fc + beta * torch.sin(2 * Math.PI * fm * t));
            return waveform.unsqueeze(0);
        }

        [Fact]
        public void FunctionalSpectrogram()
        {
            var waveform = make_waveform();

            var spectrogram = torchaudio.functional.spectrogram(
                waveform: waveform,
                pad: 3,
                window: torch.hann_window(400),
                n_fft: 512,
                hop_length: 160,
                win_length: 400,
                power: 2.0f,
                normalized: true);

            Assert.Equal(new long[] { 1, 257, 501 }, spectrogram.shape);
            var mean_square = torch.mean(torch.square(spectrogram)).item<float>();
            Assert.InRange(mean_square - 50.7892f, -1e-2f, 1e-2f);
        }

        [Fact]
        public void FunctionalInverseSpectrogram()
        {
            var waveform = make_waveform();

            var spectrogram = torchaudio.functional.spectrogram(
                waveform: waveform,
                pad: 3,
                window: torch.hann_window(400),
                n_fft: 512,
                hop_length: 160,
                win_length: 400,
                power: null,
                normalized: false);

            var inversed_waveform = torchaudio.functional.inverse_spectrogram(
                spectrogram: spectrogram,
                length: null,
                pad: 3,
                window: torch.hann_window(400),
                n_fft: 512,
                hop_length: 160,
                win_length: 400,
                normalized: false);

            var mse = torch.mean(torch.square(waveform - inversed_waveform)).item<float>();
            Assert.InRange(mse, 0, 0.1f);
        }

        [Fact]
        public void TransformsSpectrogram()
        {
            var transform = torchaudio.transforms.Spectrogram();
            var waveform = make_waveform();
            var spectrogram = transform.call(waveform);
            var expected = torchaudio.functional.spectrogram(
                waveform: waveform,
                pad: 0,
                window: torch.hann_window(400),
                n_fft: 400,
                hop_length: 200,
                win_length: 400,
                power: 2.0,
                normalized: false);
            var mse = torch.mean(torch.square(spectrogram - expected)).item<float>();
            Assert.InRange(mse, 0f, 1e-10f);
        }

        [Fact]
        public void TransformsInverseSpectrogram()
        {
            var transform = torchaudio.transforms.InverseSpectrogram();
            var spectrogram = torchaudio.functional.spectrogram(
                waveform: make_waveform(),
                pad: 0,
                window: torch.hann_window(400),
                n_fft: 400,
                hop_length: 200,
                win_length: 400,
                power: null,
                normalized: false);
            var waveform = transform.call(spectrogram);
            var expected = torchaudio.functional.inverse_spectrogram(
                spectrogram: spectrogram,
                length: null,
                pad: 0,
                window: torch.hann_window(400),
                n_fft: 400,
                hop_length: 200,
                win_length: 400,
                normalized: false);
            var mse = torch.mean(torch.square(waveform - expected)).item<float>();
            Assert.InRange(mse, 0f, 1e-10f);
        }

        [Fact]
        public void TransformsMelSpectrogram()
        {
            var mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                n_fft: 1024, n_mels: 80, hop_length: 400);
            var inverse_mel_scale = torchaudio.transforms.InverseMelScale(
                mel_spectrogram.n_fft / 2 + 1,
                n_mels: mel_spectrogram.n_mels, max_iter: 10);
            var waveform = make_waveform();
            var mel_spec = mel_spectrogram.call(waveform);
            long expected_length = waveform.shape[1] / mel_spectrogram.hop_length + 1;
            Assert.Equal(new long[] { 1, mel_spectrogram.n_mels, expected_length }, mel_spec.shape);
            var spec = inverse_mel_scale.call(mel_spec);
            Assert.Equal(new long[] { 1, mel_spectrogram.n_fft / 2 + 1, expected_length }, spec.shape);
        }

        [Fact]
        public void TestAmplitudeToDB()
        {
            var x = torch.linspace(0, 3.0, 101)[torch.TensorIndex.None, torch.TensorIndex.Colon];
            var y = torchaudio.functional.amplitude_to_DB(x, 20, 1e-10, 0.0, 80);
            var z = 20.0 * torch.log10(torch.clamp(x, min: 1e-10));
            z = torch.clamp(z, min: torch.max(z) - 80);
            var mse = torch.mean(torch.square(y - z)).item<float>();
            Assert.InRange(mse, 0f, 1e-10f);
        }

        [Fact]
        public void TestDBToAmplitude()
        {
            var x = torch.linspace(-20.0, 0.0, 101)[torch.TensorIndex.None, torch.TensorIndex.Colon];
            var y = torchaudio.functional.DB_to_amplitude(x, 1.0, 0.5);
            var z = torch.pow(torch.pow(10.0, 0.1 * x), 0.5);
            var mse = torch.mean(torch.square(y - z)).item<float>();
            Assert.InRange(mse, 0f, 1e-10f);
        }

        [Fact]
        public void TestGriffinLim()
        {
            var waveform = make_waveform();
            var window = torch.hann_window(400);
            var specgram = torchaudio.functional.spectrogram(
                waveform: waveform,
                pad: 200,
                window: window,
                n_fft: 512,
                hop_length: 160,
                win_length: 400,
                power: 2.0,
                normalized: false);
            var recovered_waveform = torchaudio.functional.griffinlim(
                specgram: specgram,
                window: window,
                n_fft: 512,
                hop_length: 160,
                win_length: 400,
                power: 2.0,
                n_iter: 32,
                momentum: 0.99,
                length: null,
                rand_init: true);
            Assert.Equal(new long[] { 1, 80320 }, recovered_waveform.shape);
        }

        [Fact]
        public void TestTransformsGriffinLim()
        {
            var transform = torchaudio.transforms.Spectrogram(
                pad: 200,
                n_fft: 512,
                hop_length: 160,
                win_length: 400,
                window_fn: win_length => torch.hann_window(400),
                power: 2.0,
                normalized: false);
            var inverse_transform = torchaudio.transforms.GriffinLim(
                n_fft: 512,
                hop_length: 160,
                win_length: 400,
                window_fn: win_length => torch.hann_window(400),
                power: 2.0,
                n_iter: 32,
                momentum: 0.99,
                length: null,
                rand_init: true);
            var waveform = make_waveform();
            var specgram = transform.call(waveform);
            var recovered_waveform = inverse_transform.call(specgram);

            Assert.Equal(new long[] { 1, 80320 }, recovered_waveform.shape);
        }

        [Fact]
        public void TestMelscaleFbanks()
        {
            int n_freqs = 257;
            double f_min = 50;
            double f_max = 7600;
            int n_mels = 64;
            int sample_rate = 16000;
            var fb = torchaudio.functional.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate);
            Assert.Equal(new long[] { n_freqs, n_mels }, fb.shape);
            // Sum of all banks should be 1.0
            Assert.True((fb.sum(dim: 1)[torch.TensorIndex.Slice(3, -23)] == 1.0).all().item<bool>());
        }

        [Fact]
        public void TestLinearFbanks()
        {
            int n_freqs = 257;
            double f_min = 50;
            double f_max = 7600;
            int n_filter = 64;
            int sample_rate = 16000;
            var fb = torchaudio.functional.linear_fbanks(n_freqs, f_min, f_max, n_filter, sample_rate);
            Assert.Equal(new long[] { n_freqs, n_filter }, fb.shape);
            // Sum of all banks should be 1.0
            Assert.True((fb.sum(dim: 1)[torch.TensorIndex.Slice(6, -17)] == 1.0).all().item<bool>());
        }

        [Fact]
        public void TestCreateDCT()
        {
            float eps = 1e-7f;
            var n_mfcc = 40;
            var n_mels = 128;
            var dct_mat = torchaudio.functional.create_dct(n_mfcc, n_mels, torchaudio.DCTNorm.ortho);
            Assert.Equal(new long[] { n_mels, n_mfcc }, dct_mat.shape);
            Assert.InRange(dct_mat[0, 10].item<float>() - 0.12405993789434433, -eps, eps);
            Assert.InRange(dct_mat[10, 0].item<float>() - 0.0883883461356163, -eps, eps);
            Assert.InRange(dct_mat[15, 20].item<float>() - 0.030372507870197296, -eps, eps);
            Assert.InRange(dct_mat[127, 39].item<float>() + 0.1109548956155777, -eps, eps);
            // dct_mat is normalized along axis=0
            Assert.InRange(torch.square(torch.square(dct_mat).sum(dim: 0) - 1.0).sum().item<float>(), -eps, eps);
        }

        [Fact]
        public void TestMuLawEncodeDecode()
        {
            int quantization_channels = 256;
            var waveform = make_waveform();
            var encoded = torchaudio.functional.mu_law_encoding(waveform, quantization_channels);
            Assert.Equal(torch.int64, encoded.dtype);
            Assert.True(torch.min(encoded).item<long>() >= 0);
            Assert.True(torch.max(encoded).item<long>() < quantization_channels);
            var decoded = torchaudio.functional.mu_law_decoding(encoded, quantization_channels);
            var mse = torch.mean(torch.square(waveform - decoded)).item<float>();
            Assert.InRange(mse, 0f, 0.01);
        }

        [Fact]
        public void TestFunctionalResampleIdent()
        {
            var waveform = make_waveform();
            var resampled_waveform = torchaudio.functional.resample(waveform, 16000, 16000);
            Assert.Equal(waveform, resampled_waveform);
        }

        [Fact]
        public void FunctionalResampleUpSample()
        {
            var waveform = make_waveform();
            var resampled_waveform = torchaudio.functional.resample(waveform, 16000, 24000);
            var x = waveform.reshape(1, -1, 2).mean(dimensions: new long[] { -1 });
            var y = resampled_waveform.reshape(1, -1, 3).mean(dimensions: new long[] { -1 });
            var mse = torch.mean(torch.square(x - y)).item<float>();
            Assert.True(mse < 1e-2);
        }

        [Fact]
        public void FunctionalResampleDownSample()
        {
            var waveform = make_waveform();
            var resampled_waveform = torchaudio.functional.resample(waveform, 16000, 8000);
            var x = waveform.reshape(1, -1, 2).mean(dimensions: new long[] { -1 });
            var y = resampled_waveform;
            var mse = torch.mean(torch.square(x - y)).item<float>();
            Assert.True(mse < 1e-2);
        }

        [Fact]
        public void TransformsResampleDownSample()
        {
            var waveform = make_waveform();
            var transform = torchaudio.transforms.Resample(16000, 8000, device: waveform.device, dtype: waveform.dtype);
            var resampled_waveform = transform.call(waveform);
            var x = waveform.reshape(1, -1, 2).mean(dimensions: new long[] { -1 });
            var y = resampled_waveform;
            var mse = torch.mean(torch.square(x - y)).item<float>();
            Assert.True(mse < 1e-2);
        }
    }
}
