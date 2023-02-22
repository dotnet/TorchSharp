using System.Collections.Generic;
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
        private class LSTMModel : nn.Module<Tensor, Tensor>
        {
            public static int NUM_WORDS = 100;
            public static int EMBEDDING_VEC_LEN = 100;
            public static int HIDDEN_SIZE = 128;

            private Module<Tensor, Tensor> embedding;
            private LSTM lstm;
            private Module<Tensor, Tensor> dropout;
            private Module<Tensor, Tensor> dense;
            private Module<Tensor, Tensor> sigmoid;
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
                var x_embed = embedding.call(input);
                var h0 = zeros(1, input.shape[0], HIDDEN_SIZE, device: _device);
                var c0 = zeros(1, input.shape[0], HIDDEN_SIZE, device: _device);
                var (x_rnn, _, _) = lstm.call(x_embed, (h0, c0));
                var x_rnn_last_seq = x_rnn[(null,null), -1, (null,null)];
                x_rnn_last_seq = dropout.call(x_rnn_last_seq);
                var logits = dense.call(x_rnn_last_seq);
                return sigmoid.call(logits);
            }
        }
        
        [Fact]
        public void TestSaveSDData_LSTM()
        {
            var lstm = new LSTMModel("lstm", torch.CPU);
            lstm.save("./lstm.dat");
        }
        
        class LeNet1Model : Module<Tensor, Tensor>
        {
            // The names of properties should be the same in C# and Python
            // in this case, we both name the Sequential as layers
            private readonly Module<Tensor, Tensor> layers;
            private Device _device;

            public LeNet1Model(string name, Device device = null) : base(name)
            {
                _device = device;

                // the names of each layer should also be the same in C# and Python
                var modules = new List<(string, Module<Tensor, Tensor>)>();
                modules.Add(("conv-1", Conv2d(1, 4, 5, padding: 2)));
                modules.Add(("bnrm2d-1", BatchNorm2d(4)));
                modules.Add(("relu-1", ReLU()));
                modules.Add(("maxpool-1", MaxPool2d(2, stride: 2)));
                modules.Add(("conv-2", Conv2d(4, 12, 5)));
                modules.Add(("bnrm2d-2", BatchNorm2d(12)));
                modules.Add(("relu-2", ReLU()));
                modules.Add(("maxpool-2", MaxPool2d(2, stride: 2)));
                modules.Add(("flatten", Flatten()));
                modules.Add(("linear", Linear(300, 10)));
                layers = Sequential(modules);

                RegisterComponents();
                if (device != null && device.type == TorchSharp.DeviceType.CUDA)
                    this.to(device);
            }

            public override Tensor forward(Tensor input)
            {
                return layers.call(input);
            }
        }
        
        
        [Fact]
        public void TestSaveSDData_LeNet1()
        {
            var lenet1 = new LeNet1Model("lenet1", torch.CPU);
            lenet1.save("./lenet1.dat");
        }
    }
}
