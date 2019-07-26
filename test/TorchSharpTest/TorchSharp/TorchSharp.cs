using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.JIT;
using TorchSharp.Tensor;
using Xunit;

namespace TorchSharp.Test
{
    public class TorchSharp
    {
        [Fact]
        public void CreateFloatTensorOnes()
        {
            TorchTensor ones = FloatTensor.Ones(new long[] { 2, 2 });
        }

        [Fact]
        public void CreateFloatTensorCheckMemory()
        {
            TorchTensor? ones = null;

            for (int i = 0; i < 10; i++)
            {
                using (var tmp = FloatTensor.Ones(new long[] { 1000, 1000, 1000 }))
                {
                    ones = tmp;
                    Assert.NotNull(ones);
                }
            }
        }

        [Fact]
        public void CreateFloatTensorOnesCheckData()
        {
            var ones = FloatTensor.Ones(new long[] { 2, 2 });
            var data = ones.Data<float>();

            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(1.0, data[i]);
            }
        }

        [Fact]
        public void CreateFloatTensorZerosCheckData()
        {
            var zeros = FloatTensor.Zeros(new long[] { 2, 2 });
            var data = zeros.Data<float>();

            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(0, data[i]);
            }
        }

        [Fact]
        public void CreateIntTensorOnesCheckData()
        {
            var ones = IntTensor.Ones(new long[] { 2, 2 });
            var data = ones.Data<int>();

            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(1, data[i]);
            }
        }

        [Fact]
        public void CreateFloatTensorCheckDevice()
        {
            var ones = FloatTensor.Ones(new long[] { 2, 2 });
            var device = ones.Device;

            Assert.Equal("cpu", ones.Device);
        }

        [Fact]
        public void CreateFloatTensorFromData()
        {
            var data = new float[1000];
            data[100] = 1;

            using (var tensor = FloatTensor.From(data, new long[] { 100, 10 }))
            {
                Assert.Equal(1, tensor.Data<float>()[100]);
            }
        }

        [Fact]
        public void CreateFloatTensorFromDataCheckDispose()
        {
            var data = new float[1000];
            data[100] = 1;

            using (var tensor = FloatTensor.From(data, new long[] { 100, 10 }))
            {
                Assert.Equal(1, tensor.Data<float>()[100]);
            }

            Assert.Equal(1, data[100]);
        }

        [Fact]
        public void CreateFloatTensorFromData2()
        {
            CreateFloatTensorFromData2Generic<float>();
        }

        private static void CreateFloatTensorFromData2Generic<T>()
        {
            var data = new T[1000];

            using (var tensor = data.ToTorchTensor(new long[] { 10, 100 }))
            {
                Assert.Equal(default(T), tensor.Data<T>()[100]);
            }
        }

        [Fact]
        public void CreateFloatTensorFromDataCheckStrides()
        {
            var data = new double[] { 0.2663158, 0.1144736, 0.1147367, 0.1249998, 0.1957895, 0.1231576, 0.1944732, 0.111842, 0.1065789, 0.667881, 0.5682123, 0.5824502, 0.4824504, 0.4844371, 0.6463582, 0.5334439, 0.5079474, 0.2281452 };
            var dataTensor = data.ToTorchTensor(new long[] { 2, 9 });

            for (int r = 0; r < 2; r++)
            {
                for (int i = 0; i < 9; i++)
                {
                    var fromData = data[(r * 9) + i];
                    var fromTensor = dataTensor[r, i].DataItem<double>();
                    Assert.True(Math.Abs(fromData - fromTensor) < 0.0001);
                }
            }

            var firstHalf = dataTensor[0];

            for (int i = 0; i < 9; i++)
            {
                var fromData = data[i];
                var fromChunk = firstHalf[i].DataItem<double>();
                Assert.True(Math.Abs(fromData - fromChunk) < 0.0001);
            }
        }

        [Fact]
        public void CreateFloatTensorFromScalar()
        {
            float scalar = 333.0f;

            using (var tensor = FloatTensor.From(scalar))
            {
                Assert.Equal(333, tensor.DataItem<float>());
            }
        }

        [Fact]
        public void CreateFloatTensorFromScalar2()
        {
            float scalar = 333.0f;

            using (var tensor = scalar.ToTorchTensor())
            {
                Assert.Equal(333, tensor.DataItem<float>());
            }
        }

        [Fact]
        public void TestScalarToTensor()
        {
            Assert.Throws<ArgumentException>(() => 1.ToTorchTensor(requiresGrad: true));
        }

        [Fact]
        public void TestScalarToTensor2()
        {
            using (var tensor = 1.ToTorchTensor())
            {
                Assert.Equal(1, tensor.DataItem<int>());
            }
        }

        [Fact]
        public void InitUniform()
        {
            using (TorchTensor tensor = FloatTensor.Zeros(new long[] { 2, 2 }))
            {
                NN.Init.Uniform(tensor);
            }
        }

        [Fact]
        public void TestSparse()
        {
            using (var i = LongTensor.From(new long[] { 0, 1, 1, 2, 0, 2 }, new long[] { 2, 3 }))
            using (var v = FloatTensor.From(new float[] { 3, 4, 5 }, new long[] { 3 }))
            {
                var sparse = FloatTensor.Sparse(i, v, new long[] { 2, 3 });

                Assert.True(sparse.IsSparse);
                Assert.False(i.IsSparse);
                Assert.False(v.IsSparse);
                Assert.Equal(sparse.Indices.Data<long>().ToArray(), new long[] { 0, 1, 1, 2, 0, 2 });
                Assert.Equal(sparse.Values.Data<float>().ToArray(), new float[] { 3, 4, 5 });
            }
        }

        [Fact]
        public void CopyCpuToCuda()
        {
            TorchTensor cpu = FloatTensor.Ones(new long[] { 2, 2 });
            Assert.Equal("cpu", cpu.Device);

            if (Torch.IsCudaAvailable())
            {
                var cuda = cpu.Cuda();
                Assert.Equal("cuda", cuda.Device);

                // Copy back to CPU to inspect the elements
                cpu = cuda.Cpu();
                var data = cpu.Data<float>();
                for (int i = 0; i < 4; i++)
                {
                    Assert.Equal(1, data[i]);
                }
            }
            else
            {
                Assert.Throws<InvalidOperationException>(() => cpu.Cuda());
            }

        }

        [Fact]
        public void CopyCudaToCpu()
        {
            if (Torch.IsCudaAvailable())
            {
                var cuda = FloatTensor.Ones(new long[] { 2, 2 }, "cuda");
                Assert.Equal("cuda", cuda.Device);

                var cpu = cuda.Cpu();
                Assert.Equal("cpu", cpu.Device);

                var data = cpu.Data<float>();
                for (int i = 0; i < 4; i++)
                {
                    Assert.Equal(1, data[i]);
                }
            }
            else
            {
                Assert.Throws<InvalidOperationException>(() => { FloatTensor.Ones(new long[] { 2, 2 }, "cuda"); });
            }
        }

        [Fact(Skip = "Need model.pt")]
        public void ScoreModel()
        {
            var ones = FloatTensor.Ones(new long[] { 1, 3, 224, 224 });

            var module = JIT.Module.Load(@"..\..\..\Resources\model.pt");
            Assert.NotNull(module);

            var result = module.Forward(ones);
        }

        [Fact(Skip = "Need model.pt")]
        public void LoadModelCheckInput()
        {
            var module = JIT.Module.Load(@"E:\Source\Repos\libtorch\model.pt");
            Assert.NotNull(module);

            var num = module.GetNumberOfInputs();

            for (int i = 0; i < num; i++)
            {
                var type = module.GetInputType(i);

                Assert.NotNull(type as DynamicType);
            }
        }

        [Fact(Skip = "Need model.pt")]
        public void LoadModelCheckOutput()
        {
            var module = JIT.Module.Load(@"E:\Source\Repos\libtorch\model.pt");
            Assert.NotNull(module);

            var num = module.GetNumberOfOutputs();

            for (int i = 0; i < num; i++)
            {
                var type = module.GetOutputType(i);

                Assert.NotNull(type as DynamicType);
            }
        }

        [Fact(Skip = "Need model.pt")]
        public void ScoreModelCheckOutput()
        {
            var module = JIT.Module.Load(@"E:\Source\Repos\libtorch\model.pt");
            Assert.NotNull(module);

            var num = module.GetNumberOfOutputs();

            for (int i = 0; i < num; i++)
            {
                var type = module.GetOutputType(i);

                Assert.NotNull(type as DynamicType);
            }
        }

        [Fact]
        public void TestSquareEuclideanDistance()
        {
            var input = new double[] { 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1 }.ToTorchTensor(new long[] { 9 }).ToType(ATenScalarMapping.Float);
            var zeros = FloatTensor.Zeros(new long[] { 1, 9 });
            var ones = FloatTensor.Ones(new long[] { 1, 9 });
            var centroids = new TorchTensor[] { zeros, ones }.Cat(0);

            var distanceFromZero = input.Reshape(new long[] { -1, 1, 9 }).Sub(zeros).Pow(2.ToScalar()).Sum(new long[] { 2 });
            var distanceFromOne = input.Reshape(new long[] { -1, 1, 9 }).Sub(ones).Pow(2.ToScalar()).Sum(new long[] { 2 });
            var distanceFromCentroids = input.Reshape(new long[] { -1, 1, 9 }).Sub(centroids).Pow(2.ToScalar()).Sum(new long[] { 2 });

            Assert.True(true);
        }

        [Fact]
        public void CreateLinear()
        {
            var lin = NN.Module.Linear(1000, 100);
            Assert.NotNull(lin);
            var modules = lin.GetName();
        }

        [Fact]
        public void TestGetBiasInLinear()
        {
            var lin = NN.Module.Linear(1000, 100);
            Assert.False(lin.WithBias);
            Assert.True(lin.Bias == null);
        }

        [Fact]
        public void TestSetGetBiasInLinear()
        {
            var lin = NN.Module.Linear(1000, 100, true);

            var bias = FloatTensor.Ones(new long[] { 1000 });

            lin.Bias = bias;

            Assert.Equal(lin.Bias?.NumberOfElements, bias.NumberOfElements);
        }

        [Fact]
        public void TestWeightAndBiasShapeInLinear()
        {
            var lin = NN.Module.Linear(1000, 100, true);

            Assert.Equal(2, lin.Weight.Shape.Length);
            Assert.Equal(100, lin.Weight.Shape[0]);
            Assert.Equal(1000, lin.Weight.Shape[1]);
            Assert.True(1 == lin.Bias?.Shape.Length);
            Assert.Equal(100, lin.Bias?.Shape[0]);
        }

        [Fact]
        public void TestWeightAndBiasParametersInLinear()
        {
            var lin = NN.Module.Linear(1000, 100, true);
            var names = lin.NamedParameters().Select(p => p.name);
            Assert.True(names.Contains("weight") == true);
            Assert.True(names.Contains("bias") == true);
        }

        [Fact]
        public void TestWeightParameterInLinear()
        {
            var lin = NN.Module.Linear(1000, 100, false);
            var names = lin.NamedParameters().Select(p => p.name);
            Assert.True(names.Contains("weight") == true);
            Assert.False(names.Contains("bias") == true);
        }

        [Fact]
        public void TestWeightAndBiasShapeInLinear3()
        {
            var lin = NN.Module.Linear(1000, 100, true);
            var weight = lin.GetParameter("weight");
            var bias = lin.GetParameter("bias");
            Assert.Equal(2, weight.Shape.Length);
            Assert.Equal(100, weight.Shape[0]);
            Assert.Equal(1000, weight.Shape[1]);
            Assert.True(1 == bias.Shape.Length);
            Assert.Equal(100, bias.Shape[0]);
        }

        [Fact]
        public void TestLinearWithBias()
        {
            var lin = NN.Module.Linear(1000, 100, true);
            var bias = lin.Bias;
            var weight = lin.Weight.T();
            var input = FloatTensor.RandomN(new long[] { 1, 1000 });
            var forward = lin.Forward(input);
            var matmul = input.MatMul(weight).Add(bias.Value);

            Assert.Equal(forward.Shape.Length, matmul.Shape.Length);
            Assert.Equal(forward.Shape[0], matmul.Shape[0]);
            Assert.Equal(forward.Shape[1], matmul.Shape[1]);

            for (int i = 0; i < 100; i++)
            {
                Assert.InRange(forward.Data<float>()[i], matmul.Data<float>()[i] - 10e5f, matmul.Data<float>()[i] + 10e5f);
            }
        }

        [Fact]
        public void TestLinearNoBias()
        {
            var lin = NN.Module.Linear(1000, 100, false);
            var weight = lin.Weight.Transpose(0, 1);
            var input = FloatTensor.RandomN(new long[] { 1, 1000 });
            var forward = lin.Forward(input);
            var matmul = input.MatMul(weight);

            Assert.Equal(forward.Shape.Length, matmul.Shape.Length);
            Assert.Equal(forward.Shape[0], matmul.Shape[0]);
            Assert.Equal(forward.Shape[1], matmul.Shape[1]);

            for (int i = 0; i < 100; i++)
            {
                Assert.Equal(forward.Data<float>()[i], matmul.Data<float>()[i]);
            }
        }

        [Fact]
        public void TestLinearEditBias()
        {
            var lin = NN.Module.Linear(1000, 100, true);
            var bias = FloatTensor.RandomN(new long[] { 100 });
            lin.Bias = bias;

            for (int i = 0; i < 100; i++)
            {
                Assert.Equal(lin.Bias.Value.Data<float>()[i], bias.Data<float>()[i]);
            }
        }

        [Fact]
        public void TestLinearEditWeightsAndBias()
        {
            var lin = NN.Module.Linear(0, 0, true);
            var bias = FloatTensor.RandomN(new long[] { 100 });
            var weights = FloatTensor.RandomN(new long[] { 100, 1000 });
            lin.Bias = bias;
            lin.Weight = weights;

            Assert.Equal(lin.Weight.Shape.Length, weights.Shape.Length);
            Assert.Equal(lin.Weight.Shape[0], weights.Shape[0]);
            Assert.Equal(lin.Weight.Shape[1], weights.Shape[1]);

            for (int i = 0; i < 100; i++)
            {
                Assert.Equal(lin.Bias.Value.Data<float>()[i], bias.Data<float>()[i]);
            }
        }

        [Fact]
        public void TestLinearEditWeightsAndBiasGetParameters()
        {
            var lin = NN.Module.Linear(0, 0, true);
            var bias = FloatTensor.RandomN(new long[] { 100 });
            var weights = FloatTensor.RandomN(new long[] { 100, 1000 });
            lin.Bias = bias;
            lin.Weight = weights;

            var parameters = lin.Parameters().ToArray();

            Assert.Equal(lin.Weight.Shape.Length, parameters[0].Shape.Length);
            Assert.Equal(lin.Weight.Shape[0], parameters[0].Shape[0]);
            Assert.Equal(lin.Weight.Shape[1], parameters[0].Shape[1]);
        }

        [Fact]
        public void CreateRelu()
        {
            var rel = NN.Module.Relu();
            Assert.NotNull(rel);
            var modules = rel.GetName();
        }

        [Fact]
        public void EvalSequence()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0", requiresGrad: true);
            var eval = seq.Forward(x);
        }

        [Fact]
        public void CreateSequence()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);
            var modules = seq.GetModules();
            Assert.Equal(3, modules.Count());
        }

        [Fact]
        public void EvalLossSequence()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            var eval = seq.Forward(x);
            var loss = NN.LossFunction.MSE(NN.Reduction.Sum);
            var output = loss(eval, y);

            var result = output.DataItem<float>();
        }

        [Fact]
        public void TestPoissonNLLLoss()
        {
            using (TorchTensor input = FloatTensor.From(new float[] { 0.5f, 1.5f, 2.5f }))
            using (TorchTensor target = FloatTensor.From(new float[] { 1f, 2f, 3f }))
            {
                var componentWiseLoss = ((TorchTensor)input.Exp()) - target * input;
                Assert.True(componentWiseLoss.Equal(NN.LossFunction.PoissonNLL(reduction: NN.Reduction.None)(input, target)));
                Assert.True(componentWiseLoss.Sum().Equal(NN.LossFunction.PoissonNLL(reduction: NN.Reduction.Sum)(input, target)));
                Assert.True(componentWiseLoss.Mean().Equal(NN.LossFunction.PoissonNLL(reduction: NN.Reduction.Mean)(input, target)));
            }
        }

        [Fact]
        public void TestPoissonNLLLoss2()
        {
            using (TorchTensor input = FloatTensor.Random(new long[] { 5, 2 }))
            using (TorchTensor target = FloatTensor.Random(new long[] { 5, 2 }))
            {
                var outTensor = NN.LossFunction.PoissonNLL(true, true)(input, target);
            }
        }

#if DEBUG
        [Fact(Skip = "Not working on Mac and Ubuntu")]
        public void TestErrorHandling()
        {
            using (TorchTensor input = FloatTensor.From(new float[] { 0.5f, 1.5f }))
            using (TorchTensor target = FloatTensor.From(new float[] { 1f, 2f, 3f }))
            {
                Assert.Throws<SEHException>(() => NN.LossFunction.PoissonNLL()(input, target));
            }
        }
#endif

        [Fact]
        public void TestZeroGrad()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);
            seq.ZeroGrad();
        }

        [Fact]
        public void TestBackward()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            var eval = seq.Forward(x);
            var loss = NN.LossFunction.MSE(NN.Reduction.None);
            var output = loss(eval, y);

            seq.ZeroGrad();

            output.Backward();
        }

        [Fact]
        public void TestGettingParameters()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            var eval = seq.Forward(x);
            var loss = NN.LossFunction.MSE(NN.Reduction.None);
            var output = loss(eval, y);

            seq.ZeroGrad();

            output.Backward();

            foreach (var parm in seq.Parameters())
            {
            }
        }

        [Fact]
        public void TestGrad()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            var eval = seq.Forward(x);
            var loss = NN.LossFunction.MSE(NN.Reduction.None);
            var output = loss(eval, y);

            seq.ZeroGrad();

            output.Backward();

            foreach (var parm in seq.Parameters())
            {
                var grad = parm.Grad();
            }
        }

        [Fact] 
        public void TestGrad2()
        {
            var y = FloatTensor.RandomN(new long[] { 32, 1 }, device: "cpu:0");
            var input = new double[] { -2.75, 0.77, -0.61, 0.14, 1.39, 0.38, -0.53, -0.5, -2.13, -0.39, 0.46, -0.61, -0.37, -0.12, 0.55, -1, 0.84, -0.02, 1.3, -0.24, -0.5, -2.12, -0.85, -0.91, 1.81, 0.02, -0.78, -1.41, -1.09, -0.65, 0.9, -0.37, -0.22, 0.28, 1.05, -0.24, 0.3, -0.99, 0.19, 0.32, -0.95, -1.19, -0.63, 0.75, 0.16, 0.15, 0.3, -0.69, 0.2, -0.4, -0.67, 0.18, -1.43, -0.61, -0.78, -0.11, -1.07, -1.71, -0.45, -0.6, 0.05, -1.59, 1.24, 0.62, 0.01, 1.35, -0.9, -1.25, 1.62, -1.45, 0.92, 1.51, -0.19, -1.33, -0.01, -0.13, 0.1, -1.34, 1.23, 0.57, -0.24, 0.5, 0.71, -0.15, -1.37, -1.03, 1.8, 1.4, -0.63, 0.8, -0.97, -0.64, 0.51, 0.52, 0.95, 0.86, 0.43, 0.73, -1.38, -0.56, 0.44, 1.2, -1.45, -0.07, 1.88, 1.57, 0.38, -2.2, -0.56, -1.52, -0.17, 1.38, -1.02, -1.61, -0.13, -0.44, -0.37, 0.23, 1.75, 0.83, -0.02, -1.91, -0.23, -0.47, -1.41, -1.01, -0.91, -0.56, -1.72, 1.47, 0.31, 0.24, 0.48, 2.06, 0.07, -0.96, 1.03, -0.4, -0.64, -0.85, 0.42, -0.33, 0.85, -0.11, -1.24, -0.71, -1.04, -0.37, -0.37, 0.84, -0.9, -1.63, -2.91, -0.71, 0.09, 1.64, -1.1, -1.05, 0.51, 0.57, 0.19, 0.36, 1.36, 1.45, 0.35, -1.66, -0.65, 0.47, 1.95, -0.32, 0.19, -2.06, 0.5, 1.03, 0.94, -0.65, -2.94, 0.41, 1.13, 0.95, -0.02, 1.12, 0.19, 0.66, -0.77, -0.39, 0.59, -1.58, -0.67, 0.88, 0.26, -0.63, 0.49, 1.38, 1.48, -0.55, 0.4, 0.65, 0.19, 0.25, 0.03, -0.31, 0.75, 2.16, -1.36, 0.05, 0.22, 0.65, 1.28, 0.42, 1.35, -0.08, 1.1, 0.25, 0.44, 1.06, -1.78, 0.47, 1.38, 0.43, -1.56, 0.14, -0.22, 1.48, 0.04, 0.33, 0.1, 0.2, -0.99, 1.04, 0.61, -0.4, 0.96, 0.4, 0.5, 0.1, 0.02, 0.01, 0.22, 1.45, -0.77, 0.69, 0.95, 0.96, -0.09, -0.26, 0.22, -1.61, 1.86, -0.06, -0.34, -0.35, 0.55, -1.08, 1.29, 0.92, 0.16, 0.55, -0.01, 0.2, -0.61, -0.28, -2.17, -0.46, 1.63, 1.61, 0.64, 0.32, -0.75, 0.33, 0.3, -1.15, 0.42, -0.06, -1.14, 1.62, -0.9, -0.39, 0.4, 1.52, -0.43, 1.22, -0.32, -0.02, 1, -0.92, 0.11, 0.8, -0.99, -0.26, -2.85, -1.13, 0.49, -0.63, -0.54, -0.86, -0.97, -0.9, 0.23, 1.26, -1.78, -0.84, -0.48, 0.35, -1.13, -2.23, 0.1, 0.95, 1.27, 0.08, -2.21, 0.67, -0.2, 0.6, -1.14, 0.65, -0.73, -0.01, 0.9, -1.33, -1.16, 0.29, 1.16, 1.19, 0.84, 0.66, -1.55, -0.58, 1.85, -1.16, -0.95, 0.98, -0.1, -1.47, 0.78, -0.75, -1.32, 0.61, -0.5, -1, -0.42, 0.96, -1.39, 0.08, -1.82, 0.51, -0.71, -0.02, 2.32, -0.71, 0.08, -1.07 }.ToTorchTensor(new long[] { 32, 11}).ToType(ATenScalarMapping.Float);
            var inputs = new TorchTensor[] { input };
            var scaler = new double[] { 0.2544529, 0.3184713, 0.2597403, 0.3246753, 0.3144654, 0.3322259, 0.3436426, 0.3215434, 0.308642, 0.3154574, 0.3448276 }.ToTorchTensor(new long[] { 1, 11 }).ToType(ATenScalarMapping.Float).RequiresGrad(true);
            var linear = new NN.Linear(11, 1, true);
            linear.Bias = new double[] { 373.8864 }.ToTorchTensor(new long[] { 1, 1 }).ToType(ATenScalarMapping.Float).RequiresGrad(true);
            linear.Weight = new double[] { 300.2818, -0.5905267, 286.2787, 0.1970505, 0.9004903, 0.1373157, 55.85495, 11.43741, 1.525748, 0.4299785, 239.9356 }.ToTorchTensor(new long[] { 1, 11 }).ToType(ATenScalarMapping.Float).RequiresGrad(true);

            var afterCat = inputs.Cat(1);
            var afterScaler = afterCat * scaler;
            var prediction = linear.Forward(afterScaler);

            var loss = NN.LossFunction.MSE();
            var output = loss(prediction, y);

            linear.ZeroGrad();

            output.Backward();

            var scalerGrad = scaler.Grad();
            var weightGrad = linear.Weight.Grad();
            var biasGrad = linear.Bias.Value.Grad();
            Assert.True(scalerGrad.Shape.Length == 2);
            Assert.True(weightGrad.Shape.Length == 2);
            Assert.True(biasGrad.Shape.Length == 2);
        }

        [Fact]
        public void TestSetGrad()
        {
            var x = FloatTensor.Random(new long[] { 10, 10 });
            Assert.False(x.IsGradRequired);

            x.RequiresGrad(true);
            Assert.True(x.IsGradRequired);
            x.RequiresGrad(false);
            Assert.False(x.IsGradRequired);
        }

        private class CondModel : NN.Module
        {
            private NN.Module fb = Linear(1000, 100);
            private NN.Module fbT1 = Linear(100, 10);
            private NN.Module fbF1 = Linear(100, 50);
            private NN.Module fbF2 = Linear(50, 10);
            private bool _isTrue = false;

            public CondModel(bool isTrue)
            {
                _isTrue = isTrue;
                RegisterModule(fb);
                RegisterModule(fbT1);
                RegisterModule(fbF1);
                RegisterModule(fbF2);
            }

            public override TorchTensor Forward(TorchTensor input)
            {
                using (var x = fb.Forward(input))
                    if (_isTrue)
                    {
                        return fbT1.Forward(x);
                    }
                    else
                    {
                        return fbF2.Forward(fbF1.Forward(x));
                    }
            }
        }

        [Fact]
        public void TestGradConditional()
        {
            var modT = new CondModel(true);
            var modF = new CondModel(false);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            modT.Train();

            var eval = modT.Forward(x);
            var loss = NN.LossFunction.MSE(NN.Reduction.None);
            var output = loss(eval, y);

            modT.ZeroGrad();

            output.Backward();
            var gradCounts = 0;

            foreach (var parm in modT.Parameters())
            {
                var grad = parm.Grad();
                gradCounts += grad.Handle == IntPtr.Zero ? 0 : 1;
            }

            Assert.Equal(2, gradCounts);

            modF.Train();

            eval = modF.Forward(x);
            output = loss(eval, y);

            modF.ZeroGrad();

            output.Backward();
            gradCounts = 0;

            foreach (var parm in modF.Parameters())
            {
                var grad = parm.Grad();
                gradCounts += grad.Handle == IntPtr.Zero ? 0 : 1;
            }

            Assert.Equal(3, gradCounts);
        }

        [Fact(Skip = "Not working on MacOS")]
        public void TestAutoGradMode()
        {
            var x = FloatTensor.RandomN(new long[] { 2, 3 }, device: "cpu:0", requiresGrad: true);
            using (var mode = new AutoGradMode(false))
            {
                Assert.False(AutoGradMode.IsAutogradEnabled());
                var sum = x.Sum();
                sum.Backward();
                var grad = x.Grad();
                Assert.True(grad.Handle == IntPtr.Zero);
            }
            using (var mode = new AutoGradMode(true))
            {
                Assert.True(AutoGradMode.IsAutogradEnabled());
                var sum = x.Sum();
                sum.Backward();
                var grad = x.Grad();
                Assert.False(grad.Handle == IntPtr.Zero);
                var data = grad.Data<float>();
                for (int i = 0; i < 2 * 3; i++)
                {
                    Assert.Equal(1.0, data[i]);
                }
            }
        }

        [Fact]
        public void TestSubInPlace()
        {
            var x = IntTensor.Ones(new long[] { 100, 100 });
            var y = IntTensor.Ones(new long[] { 100, 100 });

            x.SubInPlace(y);

            var xdata = x.Data<int>();

            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    Assert.Equal(0, xdata[i + j]);
                }
            }
        }

        [Fact]
        public void TestArithmeticOperators()
        {
            // scalar-tensor operators
            TestOneTensor<float, float>(a => a + 0.5f, a => a + 0.5f);
            TestOneTensor<float, float>(a => 0.5f + a, a => 0.5f + a);
            TestOneTensor<float, float>(a => a - 0.5f, a => a - 0.5f);
            TestOneTensor<float, float>(a => 0.5f - a, a => 0.5f - a);
            TestOneTensor<float, float>(a => a * 0.5f, a => a * 0.5f);
            TestOneTensor<float, float>(a => 0.5f * a, a => 0.5f * a);
            TestOneTensor<float, float>(a => a / 0.5f, a => a / 0.5f);
            TestOneTensor<float, float>(a => 0.5f / a, a => 0.5f / a);

            TestOneTensor<float, float>(a => a.Add(0.5f), a => a + 0.5f);
            TestOneTensor<float, float>(a => a.Sub(0.5f), a => a - 0.5f);
            TestOneTensor<float, float>(a => a.Mul(0.5f), a => a * 0.5f);
            TestOneTensor<float, float>(a => a.Div(0.5f), a => a / 0.5f);

            TestOneTensorInPlace<float>(a => a.AddInPlace(0.5f), a => a + 0.5f);
            TestOneTensorInPlace<float>(a => a.SubInPlace(0.5f), a => a - 0.5f);
            TestOneTensorInPlace<float>(a => a.MulInPlace(0.5f), a => a * 0.5f);
            TestOneTensorInPlace<float>(a => a.DivInPlace(0.5f), a => a / 0.5f);

            // tensor-tensor operators
            TestTwoTensor<float, float>((a, b) => a + b, (a, b) => a + b);
            TestTwoTensor<float, float>((a, b) => a - b, (a, b) => a - b);
            TestTwoTensor<float, float>((a, b) => a * b, (a, b) => a * b);
            TestTwoTensor<float, float>((a, b) => a / b, (a, b) => a / b);

            TestTwoTensor<float, float>((a, b) => a.Add(b), (a, b) => a + b);
            TestTwoTensor<float, float>((a, b) => a.Sub(b), (a, b) => a - b);
            TestTwoTensor<float, float>((a, b) => a.Mul(b), (a, b) => a * b);
            TestTwoTensor<float, float>((a, b) => a.Div(b), (a, b) => a / b);

            TestTwoTensorInPlace<float>((a, b) => a.AddInPlace(b), (a, b) => a + b);
            TestTwoTensorInPlace<float>((a, b) => a.SubInPlace(b), (a, b) => a - b);
            TestTwoTensorInPlace<float>((a, b) => a.MulInPlace(b), (a, b) => a * b);
            TestTwoTensorInPlace<float>((a, b) => a.DivInPlace(b), (a, b) => a / b);
        }

        [Fact]
        public void TestComparisonOperators()
        {
            // scalar-tensor operators
            TestOneTensor<float, bool>(a => a == 5.0f, a => a == 5.0f);
            TestOneTensor<float, bool>(a => a != 5.0f, a => a != 5.0f);
            TestOneTensorInPlace<float>(a => a.EqInPlace(5.0f), a => a == 5.0f ? 1.0f : 0.0f);
            TestOneTensorInPlace<float>(a => a.NeInPlace(5.0f), a => a != 5.0f ? 1.0f : 0.0f);

            TestOneTensor<float, bool>(a => a < 5.0f, a => a < 5.0f);
            TestOneTensor<float, bool>(a => 5.0f < a, a => 5.0f < a);
            TestOneTensor<float, bool>(a => a <= 5.0f, a => a <= 5.0f);
            TestOneTensor<float, bool>(a => 5.0f <= a, a => 5.0f <= a);
            TestOneTensor<float, bool>(a => a > 5.0f, a => a > 5.0f);
            TestOneTensor<float, bool>(a => 5.0f > a, a => 5.0f > a);
            TestOneTensor<float, bool>(a => a >= 5.0f, a => a >= 5.0f);
            TestOneTensor<float, bool>(a => 5.0f >= a, a => 5.0f >= a);

            TestOneTensorInPlace<float>(a => a.LtInPlace(5.0f), a => a < 5.0f ? 1.0f : 0.0f);
            TestOneTensorInPlace<float>(a => a.LeInPlace(5.0f), a => a <= 5.0f ? 1.0f : 0.0f);
            TestOneTensorInPlace<float>(a => a.GtInPlace(5.0f), a => a > 5.0f ? 1.0f : 0.0f);
            TestOneTensorInPlace<float>(a => a.GeInPlace(5.0f), a => a >= 5.0f ? 1.0f : 0.0f);

            TestOneTensor<float, float>(a => 5.0f % a, a => 5.0f % a);
            TestOneTensor<float, float>(a => a % 5.0f, a => a % 5.0f);
            TestOneTensorInPlace<float>(a => a.RemainderInPlace(5.0f), a => a % 5.0f);

            // tensor-tensor operators
            TestTwoTensor<float, bool>((a, b) => a == b, (a, b) => a == b);
            TestTwoTensor<float, bool>((a, b) => a != b, (a, b) => a != b);
            TestTwoTensorInPlace<float>((a, b) => a.EqInPlace(b), (a, b) => a == b ? 1.0f : 0.0f);
            TestTwoTensorInPlace<float>((a, b) => a.NeInPlace(b), (a, b) => a != b ? 1.0f : 0.0f);

            TestTwoTensor<float, bool>((a, b) => a < b, (a, b) => a < b);
            TestTwoTensor<float, bool>((a, b) => a <= b, (a, b) => a <= b);
            TestTwoTensor<float, bool>((a, b) => a > b, (a, b) => a > b);
            TestTwoTensor<float, bool>((a, b) => a >= b, (a, b) => a >= b);

            TestTwoTensorInPlace<float>((a, b) => a.LtInPlace(b), (a, b) => a < b ? 1.0f : 0.0f);
            TestTwoTensorInPlace<float>((a, b) => a.LeInPlace(b), (a, b) => a <= b ? 1.0f : 0.0f);
            TestTwoTensorInPlace<float>((a, b) => a.GtInPlace(b), (a, b) => a > b ? 1.0f : 0.0f);
            TestTwoTensorInPlace<float>((a, b) => a.GeInPlace(b), (a, b) => a >= b ? 1.0f : 0.0f);

            TestTwoTensor<float, float>((a, b) => a % b, (a, b) => a % b);
            TestTwoTensorInPlace<float>((a, b) => a.RemainderInPlace(b), (a, b) => a % b);
        }

        private void TestOneTensor<Tin, Tout>(Func<TorchTensor, TorchTensor> tensorFunc, 
            Func<Tin, Tout> scalarFunc)
        {
            var c1 = FloatTensor.Arange(0, 10, 1);
            var c2 = FloatTensor.Ones(new long[] { 10, 10 });

            var x = c1 * c2;
            var y = tensorFunc(x);

            var xData = x.Data<Tin>();
            var yData = y.Data<Tout>();

            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.Equal(yData[i + j], scalarFunc(xData[i + j]));
                }
            }
        }

        private void TestOneTensorInPlace<Tin>(Func<TorchTensor, TorchTensor> tensorFunc,
            Func<Tin, Tin> scalarFunc)
        {
            var c1 = FloatTensor.Arange(0, 10, 1);
            var c2 = FloatTensor.Ones(new long[] { 10, 10 });

            var x = c1 * c2;
            var xClone = x.Clone();
            var y = tensorFunc(x);

            var xData = x.Data<Tin>();
            var xCloneData = xClone.Data<Tin>();
            var yData = y.Data<Tin>();

            Assert.True(xData == yData);
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.Equal(yData[i + j], scalarFunc(xCloneData[i + j]));
                }
            }
        }

        private void TestTwoTensor<Tin, Tout>(Func<TorchTensor, TorchTensor, TorchTensor> tensorFunc,
            Func<Tin, Tin, Tout> scalarFunc)
        {
            var c1 = FloatTensor.Arange(0, 10, 1);
            var c2 = FloatTensor.Arange(10, 0, -1);
            var c3 = FloatTensor.Ones(new long[] { 10, 10 });

            var x = c1 * c3;
            var y = c2 * c3;

            var z = tensorFunc(x, y);

            var xData = x.Data<Tin>();
            var yData = y.Data<Tin>();
            var zData = z.Data<Tout>();

            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.Equal(zData[i + j], scalarFunc(xData[i + j], yData[i + j]));
                }
            }
        }

        private void TestTwoTensorInPlace<Tin>(Func<TorchTensor, TorchTensor, TorchTensor> tensorFunc,
            Func<Tin, Tin, Tin> scalarFunc)
        {
            var c1 = FloatTensor.Arange(0, 10, 1);
            var c2 = FloatTensor.Arange(10, 0, -1);
            var c3 = FloatTensor.Ones(new long[] { 10, 10 });

            var x = c1 * c3;
            var xClone = x.Clone();
            var y = c2 * c3;

            var z = tensorFunc(x, y);

            var xData = x.Data<Tin>();
            var xCloneData = xClone.Data<Tin>();
            var yData = y.Data<Tin>();
            var zData = z.Data<Tin>();

            Assert.True(xData == zData);
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.Equal(zData[i + j], scalarFunc(xCloneData[i + j], yData[i + j]));
                }
            }
        }

        [Fact]
        public void TestMul()
        {
            var x = FloatTensor.Ones(new long[] { 100, 100 });

            var y = x.Mul(0.5f.ToScalar());

            var ydata = y.Data<float>();
            var xdata = x.Data<float>();

            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    Assert.Equal(ydata[i + j], xdata[i + j] * 0.5f);
                }
            }
        }

        [Fact]
        public void TestCustomModule()
        {
            var module = new TestModule("test", FloatTensor.RandomN(new long[] { 2, 2 }), true);
            var name = module.GetName();
            Assert.NotNull(name);
            Assert.True(module.HasParameter("test"));
        }

        [Fact]
        public void TestCustomModuleWithInPlaceModification()
        {
            var param = FloatTensor.RandomN(new long[] { 1000, 100 });
            var module = new TestModule("test", param, true);

            Assert.Equal(1000, module.GetParameter("test").Shape[0]);
            Assert.Equal(100, module.GetParameter("test").Shape[1]);

            using (var grad = new AutoGradMode(false))
            {
                param.TransposeInPlace(0, 1);
            }
            Assert.Equal(100, module.GetParameter("test").Shape[0]);
            Assert.Equal(1000, module.GetParameter("test").Shape[1]);
            Assert.Equal(100, param.Shape[0]);
            Assert.Equal(1000, param.Shape[1]);
        }

        private class TestModule : NN.Module
        {
            public TestModule(string name, TorchTensor tensor, bool withGrad)
                : base(new NN.Parameter(name, tensor, withGrad))
            {
            }

            public override TorchTensor Forward(TorchTensor input)
            {
                throw new NotImplementedException();
            }
        }


        /// <summary>
        /// Fully connected Relu net with one hidden layer trained using gradient descent.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html"/>.
        /// </summary>
        [Fact]
        public void TestTraining()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            float learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var loss = NN.LossFunction.MSE(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++)
            {
                var eval = seq.Forward(x);
                var output = loss(eval, y);
                var lossVal = output.DataItem<float>();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                seq.ZeroGrad();

                output.Backward();

                using (var noGrad = new AutoGradMode(false))
                {
                    foreach (var param in seq.Parameters())
                    {
                        var grad = param.Grad();
                        var update = grad.Mul(learning_rate.ToScalar());
                        param.SubInPlace(update);
                    }
                }
            }
        }

        [Fact]
        public void TestAdam()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            double learning_rate = 0.00001;

            var optimizer = NN.Optimizer.Adam(seq.Parameters(), learning_rate);

            Assert.NotNull(optimizer);
        }

        /// <summary>
        /// Fully connected Relu net with one hidden layer trained using Adam optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TestTrainingAdam()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            double learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var optimizer = NN.Optimizer.Adam(seq.Parameters(), learning_rate);
            var loss = NN.LossFunction.MSE(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++)
            {
                var eval = seq.Forward(x);
                var output = loss(eval, y);
                var lossVal = output.DataItem<float>();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.ZeroGrad();

                output.Backward();

                optimizer.Step();
            }
        }

        [Fact(Skip = "MNIST data too big to keep in repo")]
        public void TestMNISTLoader()
        {
            using (var train = Data.Loader.MNIST("../../../../test/data/MNIST", 32))
            {
                Assert.NotNull(train);

                var size = train.Size();
                int i = 0;

                foreach (var (data, target) in train)
                {
                    i++;

                    Assert.Equal(data.Shape, new long[] { 32, 1, 28, 28 });
                    Assert.Equal(target.Shape, new long[] { 32 });

                    data.Dispose();
                    target.Dispose();
                }

                Assert.Equal(size, i * 32);
            }
        }

        [Fact(Skip = "CIFAR10 data too big to keep in repo")]
        public void TestCIFAR10Loader()
        {
            using (var train = Data.Loader.CIFAR10("../../../../src/Examples/Data", 16))
            {
                Assert.NotNull(train);

                var size = train.Size();
                int i = 0;

                foreach (var (data, target) in train)
                {
                    i++;

                    Assert.Equal(data.Shape, new long[] { 16, 3, 32, 32 });
                    Assert.Equal(target.Shape, new long[] { 16 });
                    Assert.True(target.Data<int>().ToArray().Where(x => x >= 0 && x < 10).Count() == 16);

                    data.Dispose();
                    target.Dispose();
                }

                Assert.Equal(size, i * 16);
            }
        }

        [Fact(Skip = "MNIST data too big to keep in repo")]
        public void TestMNISTLoaderWithEpochs()
        {
            using (var train = Data.Loader.MNIST("../../../../test/data/MNIST", 32))
            {
                var size = train.Size();
                var epochs = 10;

                int i = 0;

                for (int e = 0; e < epochs; e++)
                {
                    foreach (var (data, target) in train)
                    {
                        i++;

                        Assert.Equal(data.Shape, new long[] { 32, 1, 28, 28 });
                        Assert.Equal(target.Shape, new long[] { 32 });

                        data.Dispose();
                        target.Dispose();
                    }
                }

                Assert.Equal(size * epochs, i * 32);
            }
        }
    }
}