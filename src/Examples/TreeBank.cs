using System;
using System.IO;
using System.IO.Compression;
using ICSharpCode.SharpZipLib.Core;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using System.Collections.Generic;
using System.Diagnostics;
using TorchSharp.Tensor;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using static TorchSharp.NN.Functions;

namespace TorchSharp.Examples
{
    public class TreeBank
    {
        static void Main(string[] args)

        {
            Torch.SetSeed(1);

            var cwd = Environment.CurrentDirectory;

            //var device = Device.CPU; //Torch.IsCudaAvailable() ? Device.CUDA : Device.CPU;
            var device = Torch.IsCudaAvailable() ? Device.CUDA : Device.CPU;
            Console.WriteLine($"Running on {device.Type.ToString()}");

            // Test PosEnc

            var pe = new PositionalEncoding("Test", 32, 0.1, 64);
            var res = pe.forward(Float32Tensor.rand(new long[] { 4, 16, 32 }));
            var data = res.Data<float>();

            var tm = new TransformerModel("test");
            var mask = tm.GenerateSquareSubsequentMask(10);

        }

        class TransformerModel : CustomModule
        {
            private TransformerEncoder transformer_encoder;
            private PositionalEncoding pos_encoder;
            private Embedding encoder;
            private Linear decoder;

            public TransformerModel(string name) : base(name)
            {

            }

            public TorchTensor GenerateSquareSubsequentMask(long size)
            {
                var mask = (Float32Tensor.ones(new long[] { size, size }) == 1).triu().transpose(0, 1);
                return mask.to_type(ScalarType.Float32).masked_fill(mask == 0, float.NegativeInfinity).masked_fill(mask == 1, 0.0f);
            }

            private void InitWeights()
            {
                var initrange = 0.1;

                NN.Init.encoder.Weight.uniform(-initrange, initrange);
                decoder.Bias.zeros_out()
            }

            public override TorchTensor forward(TorchTensor t)
            {
                throw new NotImplementedException();
            }
        }

        class PositionalEncoding : CustomModule
        {
            private Dropout dropout;
            private TorchTensor pe;

            public PositionalEncoding(string name, long dmodel, double dropout, int maxLen = 5000) : base(name)
            {
                this.dropout = Dropout(dropout);
                var pe = Float32Tensor.zeros(new long[] { maxLen, dmodel });
                var position = Float32Tensor.arange(0, maxLen, 1).unsqueeze(1);
                var divTerm = (Float32Tensor.arange(0, dmodel, 2) * (-Math.Log(10000.0) / dmodel)).exp();
                pe[TorchTensorIndex.Ellipsis, TorchTensorIndex.Slice(0, null, 2)] = (position * divTerm).sin();
                pe[TorchTensorIndex.Ellipsis, TorchTensorIndex.Slice(1, null, 2)] = (position * divTerm).cos();
                this.pe = pe.unsqueeze(0).transpose(0, 1);
            }

            public override TorchTensor forward(TorchTensor t)
            {
                var x = t + pe[TorchTensorIndex.Slice(null, t.shape[0]), TorchTensorIndex.Slice()];
                return dropout.forward(x);
            }
        }
    }
}
