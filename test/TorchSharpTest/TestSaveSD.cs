using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Xunit;


namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestSaveSD
    {
        private class LSTMModel : Module
        {
            public static int NUM_WORDS = 100;
            public static int EMBEDDING_VEC_LEN = 100;
            public static int HIDDEN_SIZE = 128;

            private Module embedding;
            private LSTM lstm;
            private Module dropout;
            private Module dense;
            private Module sigmoid;
            private Device _device;

            public LSTMModel(string name, Device device = null) : base(name)
            {
                _device = device;
                embedding = Embedding(NUM_WORDS, EMBEDDING_VEC_LEN);
                lstm = LSTM(EMBEDDING_VEC_LEN, HIDDEN_SIZE, batchFirst: true);
                dropout = Dropout(0.5);
                dense = Linear(HIDDEN_SIZE, 1);
                sigmoid = Sigmoid();

                RegisterComponents();
                if (device != null && device.type == DeviceType.CUDA)
                    this.to(device);
            }

            public override Tensor forward(Tensor input)
            {
                var x_embed = embedding.forward(input);
                var h0 = zeros(1, input.shape[0], HIDDEN_SIZE, device: _device);
                var c0 = zeros(1, input.shape[0], HIDDEN_SIZE, device: _device);
                var (x_rnn, _, _) = lstm.forward(x_embed, (h0, c0));
                var x_rnn_last_seq = x_rnn[.., -1, ..];
                x_rnn_last_seq = dropout.forward(x_rnn_last_seq);
                var logits = dense.forward(x_rnn_last_seq);
                return sigmoid.forward(logits);
            }
        }
        
        [Fact]
        public void TestSaveSDData()
        {
            var lstm = new LSTMModel("lstm", torch.CPU);
            lstm.save("./lstm.dat");
        }
    }
}
