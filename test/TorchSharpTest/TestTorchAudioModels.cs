using System;
using Xunit;

using TorchSharp;

namespace TorchSharp
{
    public class TestTorchAudioModels
    {
        [Fact]
        public void Tacotron2Model()
        {
            var tacotron2 = torchaudio.models.Tacotron2(n_symbol: 96);
            Assert.Equal(80, tacotron2.n_mels);
        }
    }
}
