using System;
using Xunit;
using TorchSharp;

namespace TorchAudio
{
    public class TestTorchAudioLoadSave
    {
        const int ExpectedSampleRate = 16000;

        private static torch.Tensor make_waveform()
        {
            var fc = 220;
            var fm = 440;
            var beta = 1;
            var t = torch.linspace(0, 5, ExpectedSampleRate * 5);
            var waveform = 1.0 * torch.sin(2 * Math.PI * t * fc + beta * torch.sin(2 * Math.PI * fm * t));
            return waveform;
        }

        private class MockAudioBackend : torchaudio.backend.AudioBackend
        {
            public override (torch.Tensor, int) load(string filepath, long frame_offset = 0, long num_frames = -1, bool normalize = true, bool channels_first = true, torchaudio.AudioFormat? format = null)
            {
                return (make_waveform(), ExpectedSampleRate);
            }

            public override void save(string filepath, torch.Tensor src, int sample_rate, bool channels_first = true, float? compression = null, torchaudio.AudioFormat? format = null, torchaudio.AudioEncoding? encoding = null, int? bits_per_sample = null)
            {
                Assert.Equal(ExpectedSampleRate, sample_rate);
                float mse = torch.mean(torch.square(make_waveform() - src)).item<float>();
                Assert.True(mse < 1e-10);
            }

            public override torchaudio.AudioMetaData info(string filepath, torchaudio.AudioFormat? format = null)
            {
                return new torchaudio.AudioMetaData {
                };
            }
        }

        [Fact]
        public void LoadSaveNoBackend()
        {
            Assert.Throws<InvalidOperationException>(() => {
                var (waveform, sample_rate) = torchaudio.load("input.wav");
            });
            Assert.Throws<InvalidOperationException>(() => {
                var waveform = make_waveform();
                var sample_rate = ExpectedSampleRate;
                torchaudio.save("output.wav", waveform, sample_rate);
            });
        }

        [Fact]
        public void LoadSaveStub()
        {
            torchaudio.backend.utils.set_audio_backend(new MockAudioBackend());
            var (waveform, sample_rate) = torchaudio.load("input.wav");
            torchaudio.save("output.wav", waveform, sample_rate);
        }
    }
}
