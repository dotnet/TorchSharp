// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using static TorchSharp.NN.Losses;
using static TorchSharp.NN.Functions;
using TorchSharp.Tensor;
using Xunit;

namespace TorchSharp.Test
{
    public class TorchSharp
    {
        [Fact]
        public void CreateFloatTensorOnes()
        {
            var shape = new long[] { 2, 2 };
            TorchTensor ones = FloatTensor.Ones(shape);
            Assert.Equal(shape, ones.Shape);
        }

        [Fact]
        public void CreateFloatTensorCheckMemory()
        {
            TorchTensor? ones = null;

            for (int i = 0; i < 10; i++)
            {
                using (var tmp = FloatTensor.Ones(new long[] { 100, 100, 100 }))
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
                Assert.Equal(ATenScalarMapping.Int, tensor.Type);
                Assert.Equal(1, tensor.DataItem<int>());
            }
            using (var tensor = ((byte)1).ToTorchTensor())
            {
                Assert.Equal(ATenScalarMapping.Byte, tensor.Type);
                Assert.Equal(1, tensor.DataItem<byte>());
            }
            using (var tensor = ((sbyte)-1).ToTorchTensor())
            {
                Assert.Equal(ATenScalarMapping.SByte, tensor.Type);
                Assert.Equal(-1, tensor.DataItem<sbyte>());
            }
            using (var tensor = ((short)-1).ToTorchTensor())
            {
                Assert.Equal(ATenScalarMapping.Short, tensor.Type);
                Assert.Equal(-1, tensor.DataItem<short>());
            }
            using (var tensor = ((long)-1).ToTorchTensor())
            {
                Assert.Equal(ATenScalarMapping.Long, tensor.Type);
                Assert.Equal(-1L, tensor.DataItem<short>());
            }
            using (var tensor = ((float)-1).ToTorchTensor())
            {
                Assert.Equal(ATenScalarMapping.Float, tensor.Type);
                Assert.Equal(-1.0f, tensor.DataItem<float>());
            }
            using (var tensor = ((double)-1).ToTorchTensor())
            {
                Assert.Equal(ATenScalarMapping.Double, tensor.Type);
                Assert.Equal(-1.0, tensor.DataItem<double>());
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

        //[Fact(Skip = "Need model.pt")]
        //public void ScoreModel()
        //{
        //    var ones = FloatTensor.Ones(new long[] { 1, 3, 224, 224 });

        //    var module = JIT.Module.Load(@"..\..\..\Resources\model.pt");
        //    Assert.NotNull(module);

        //    var result = module.Forward(ones);
        //}

        //[Fact(Skip = "Need model.pt")]
        //public void LoadModelCheckInput()
        //{
        //    var module = JIT.Module.Load(@"E:\Source\Repos\libtorch\model.pt");
        //    Assert.NotNull(module);

        //    var num = module.GetNumberOfInputs();

        //    for (int i = 0; i < num; i++)
        //    {
        //        var type = module.GetInputType(i);

        //        Assert.NotNull(type as DynamicType);
        //    }
        //}

        //[Fact(Skip = "Need model.pt")]
        //public void LoadModelCheckOutput()
        //{
        //    var module = JIT.Module.Load(@"E:\Source\Repos\libtorch\model.pt");
        //    Assert.NotNull(module);

        //    var num = module.GetNumberOfOutputs();

        //    for (int i = 0; i < num; i++)
        //    {
        //        var type = module.GetOutputType(i);

        //        Assert.NotNull(type as DynamicType);
        //    }
        //}

        //[Fact(Skip = "Need model.pt")]
        //public void ScoreModelCheckOutput()
        //{
        //    var module = JIT.Module.Load(@"E:\Source\Repos\libtorch\model.pt");
        //    Assert.NotNull(module);

        //    var num = module.GetNumberOfOutputs();

        //    for (int i = 0; i < num; i++)
        //    {
        //        var type = module.GetOutputType(i);

        //        Assert.NotNull(type as DynamicType);
        //    }
        //}

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
            var lin = Linear(1000, 100);
            Assert.NotNull(lin);
            Assert.True(lin.Bias.HasValue);
            //var name = lin.GetName();

            var ps = lin.GetParameters();
            Assert.Equal(2, ps.Length);
        }

        [Fact]
        public void TestGetBiasInLinear()
        {
            var lin = Linear(1000, 100, false);
            var ps = lin.GetParameters();
            var nps = ps.Length;
            Assert.Equal(1, nps);
            Assert.True(lin.Bias == null);

            var lin2 = Linear(1000, 100, true);
            Assert.True(lin2.Bias != null);
        }

        [Fact]
        public void TestSetGetBiasInLinear()
        {
            var lin = Linear(1000, 100, true);
            var bias = FloatTensor.Ones (new long[] { 1000 });
            lin.Bias = bias;
            Assert.True(lin.Bias.HasValue);

            Assert.Equal(lin.Bias?.NumberOfElements, bias.NumberOfElements);
        }

        [Fact]
        public void TestWeightAndBiasShapeInLinear()
        {
            var lin = Linear(1000, 100, true);

            Assert.Equal(2, lin.Weight.Shape.Length);
            Assert.Equal(100, lin.Weight.Shape[0]);
            Assert.Equal(1000, lin.Weight.Shape[1]);
            Assert.True(1 == lin.Bias?.Shape.Length);
            Assert.Equal(100, lin.Bias?.Shape[0]);
        }

        [Fact]
        public void TestWeightAndBiasParametersInLinear()
        {
            var lin = Linear(1000, 100, true);
            var names = lin.NamedParameters().Select(p => p.name);
            Assert.True(names.Contains("weight") == true);
            Assert.True(names.Contains("bias") == true);
        }

        [Fact]
        public void TestWeightParameterInLinear()
        {
            var lin = Linear(1000, 100, false);
            var names = lin.NamedParameters().Select(p => p.name);
            Assert.True(names.Contains("weight") == true);
            Assert.False(names.Contains("bias") == true);
        }

        [Fact]
        public void TestWeightAndBiasShapeInLinear3()
        {
            var lin = Linear(1000, 100, true);
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
            var lin = Linear(1000, 100, true);
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
            var lin = Linear(1000, 100, false);
            Assert.False(lin.Bias.HasValue);

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
            var lin = Linear(1000, 100, true);
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
            var lin = Linear(1000, 1000, true);
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
            var lin = Linear(1000, 1000, true);
            var bias = FloatTensor.RandomN(new long[] { 100 });
            var weights = FloatTensor.RandomN(new long[] { 1000, 1000 });
            lin.Bias = bias;
            lin.Weight = weights;

            var parameters = lin.GetParameters().ToArray();

            Assert.Equal(lin.Weight.Shape.Length, parameters[0].Shape.Length);
            Assert.Equal(lin.Weight.Shape[0], parameters[0].Shape[0]);
            Assert.Equal(lin.Weight.Shape[1], parameters[0].Shape[1]);
        }

        [Fact]
        public void CreateRelu()
        {
            var rel = Relu();
            Assert.NotNull(rel);
            var modules = rel.GetName();
        }

        [Fact]
        public void EvalSequence()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", Relu()),
                ("lin2", lin2));

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0", requiresGrad: true);
            var eval = seq.Forward(x);
        }

        [Fact]
        public void CreateSequence()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", Relu()),
                ("lin2", lin2));
            var parameters = seq.GetParameters();
            var parametersCount = parameters.Count ();
            Assert.Equal (4, parametersCount);

            var namedParams = seq.GetParameters ();
            var namedParamsCount = namedParams.Count ();
            Assert.Equal(4, namedParamsCount);
        }

        [Fact]
        public void EvalLossSequence()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", Relu()),
                ("lin2", lin2));

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            var eval = seq.Forward(x);
            var loss = MSE(NN.Reduction.Sum);
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
                Assert.True(componentWiseLoss.Equal(PoissonNLL(reduction: NN.Reduction.None)(input, target)));
                Assert.True(componentWiseLoss.Sum().Equal(PoissonNLL(reduction: NN.Reduction.Sum)(input, target)));
                Assert.True(componentWiseLoss.Mean().Equal(PoissonNLL(reduction: NN.Reduction.Mean)(input, target)));
            }
        }

        [Fact]
        public void TestPoissonNLLLoss2()
        {
            using (TorchTensor input = FloatTensor.Random(new long[] { 5, 2 }))
            using (TorchTensor target = FloatTensor.Random(new long[] { 5, 2 }))
            {
                var outTensor = PoissonNLL(true, true)(input, target);
            }
        }

#if DEBUG
        [Fact(Skip = "Not working on Mac and Ubuntu (note: may now be working, we need to recheck)")]
        public void TestErrorHandling()
        {
            using (TorchTensor input = FloatTensor.From(new float[] { 0.5f, 1.5f }))
            using (TorchTensor target = FloatTensor.From(new float[] { 1f, 2f, 3f }))
            {
                Assert.Throws<ExternalException>(() => PoissonNLL()(input, target));
            }
        }
#endif

        [Fact]
        public void TestZeroGrad()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", Relu()),
                ("lin2", lin2));
            seq.ZeroGrad();
        }

        [Fact]
        public void TestBackward()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", Relu()),
                ("lin2", lin2));

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0", requiresGrad: true);
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0", requiresGrad: true);

            var eval = seq.Forward(x);
            var loss = MSE(NN.Reduction.Sum);
            var output = loss(eval, y);

            seq.ZeroGrad();

            output.Backward();
        }

        [Fact]
        public void TestGettingParameters()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", Relu()),
                ("lin2", lin2));

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0", requiresGrad: true);
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0", requiresGrad: true);

            var eval = seq.Forward(x);
            var loss = MSE(NN.Reduction.Sum);
            var output = loss(eval, y);

            seq.ZeroGrad();

            output.Backward();

            foreach (var parm in seq.GetParameters())
            {
            }
        }

        [Fact]
        public void TestGrad()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", Relu()),
                ("lin2", lin2));

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0", requiresGrad: true);
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0", requiresGrad: true);

            var eval = seq.Forward(x);
            var loss = MSE(NN.Reduction.Sum);
            var output = loss(eval, y);

            seq.ZeroGrad();

            output.Backward();

            foreach (var parm in seq.GetParameters())
            {
                var grad = parm.Grad();
            }
        }

        [Fact]
        public void TestGrad2()
        {
            var y = FloatTensor.RandomN(new long[] { 32, 1 }, device: "cpu:0");
            var input = new double[] { -2.75, 0.77, -0.61, 0.14, 1.39, 0.38, -0.53, -0.5, -2.13, -0.39, 0.46, -0.61, -0.37, -0.12, 0.55, -1, 0.84, -0.02, 1.3, -0.24, -0.5, -2.12, -0.85, -0.91, 1.81, 0.02, -0.78, -1.41, -1.09, -0.65, 0.9, -0.37, -0.22, 0.28, 1.05, -0.24, 0.3, -0.99, 0.19, 0.32, -0.95, -1.19, -0.63, 0.75, 0.16, 0.15, 0.3, -0.69, 0.2, -0.4, -0.67, 0.18, -1.43, -0.61, -0.78, -0.11, -1.07, -1.71, -0.45, -0.6, 0.05, -1.59, 1.24, 0.62, 0.01, 1.35, -0.9, -1.25, 1.62, -1.45, 0.92, 1.51, -0.19, -1.33, -0.01, -0.13, 0.1, -1.34, 1.23, 0.57, -0.24, 0.5, 0.71, -0.15, -1.37, -1.03, 1.8, 1.4, -0.63, 0.8, -0.97, -0.64, 0.51, 0.52, 0.95, 0.86, 0.43, 0.73, -1.38, -0.56, 0.44, 1.2, -1.45, -0.07, 1.88, 1.57, 0.38, -2.2, -0.56, -1.52, -0.17, 1.38, -1.02, -1.61, -0.13, -0.44, -0.37, 0.23, 1.75, 0.83, -0.02, -1.91, -0.23, -0.47, -1.41, -1.01, -0.91, -0.56, -1.72, 1.47, 0.31, 0.24, 0.48, 2.06, 0.07, -0.96, 1.03, -0.4, -0.64, -0.85, 0.42, -0.33, 0.85, -0.11, -1.24, -0.71, -1.04, -0.37, -0.37, 0.84, -0.9, -1.63, -2.91, -0.71, 0.09, 1.64, -1.1, -1.05, 0.51, 0.57, 0.19, 0.36, 1.36, 1.45, 0.35, -1.66, -0.65, 0.47, 1.95, -0.32, 0.19, -2.06, 0.5, 1.03, 0.94, -0.65, -2.94, 0.41, 1.13, 0.95, -0.02, 1.12, 0.19, 0.66, -0.77, -0.39, 0.59, -1.58, -0.67, 0.88, 0.26, -0.63, 0.49, 1.38, 1.48, -0.55, 0.4, 0.65, 0.19, 0.25, 0.03, -0.31, 0.75, 2.16, -1.36, 0.05, 0.22, 0.65, 1.28, 0.42, 1.35, -0.08, 1.1, 0.25, 0.44, 1.06, -1.78, 0.47, 1.38, 0.43, -1.56, 0.14, -0.22, 1.48, 0.04, 0.33, 0.1, 0.2, -0.99, 1.04, 0.61, -0.4, 0.96, 0.4, 0.5, 0.1, 0.02, 0.01, 0.22, 1.45, -0.77, 0.69, 0.95, 0.96, -0.09, -0.26, 0.22, -1.61, 1.86, -0.06, -0.34, -0.35, 0.55, -1.08, 1.29, 0.92, 0.16, 0.55, -0.01, 0.2, -0.61, -0.28, -2.17, -0.46, 1.63, 1.61, 0.64, 0.32, -0.75, 0.33, 0.3, -1.15, 0.42, -0.06, -1.14, 1.62, -0.9, -0.39, 0.4, 1.52, -0.43, 1.22, -0.32, -0.02, 1, -0.92, 0.11, 0.8, -0.99, -0.26, -2.85, -1.13, 0.49, -0.63, -0.54, -0.86, -0.97, -0.9, 0.23, 1.26, -1.78, -0.84, -0.48, 0.35, -1.13, -2.23, 0.1, 0.95, 1.27, 0.08, -2.21, 0.67, -0.2, 0.6, -1.14, 0.65, -0.73, -0.01, 0.9, -1.33, -1.16, 0.29, 1.16, 1.19, 0.84, 0.66, -1.55, -0.58, 1.85, -1.16, -0.95, 0.98, -0.1, -1.47, 0.78, -0.75, -1.32, 0.61, -0.5, -1, -0.42, 0.96, -1.39, 0.08, -1.82, 0.51, -0.71, -0.02, 2.32, -0.71, 0.08, -1.07 }.ToTorchTensor(new long[] { 32, 11 }).ToType(ATenScalarMapping.Float);
            var inputs = new TorchTensor[] { input };
            var scaler = new double[] { 0.2544529, 0.3184713, 0.2597403, 0.3246753, 0.3144654, 0.3322259, 0.3436426, 0.3215434, 0.308642, 0.3154574, 0.3448276 }.ToTorchTensor(new long[] { 1, 11 }).ToType(ATenScalarMapping.Float).RequiresGrad(true);
            var linear = Linear(11, 1, true);
            linear.Bias = new double[] { 373.8864 }.ToTorchTensor(new long[] { 1, 1 }).ToType(ATenScalarMapping.Float).RequiresGrad(true);
            linear.Weight = new double[] { 300.2818, -0.5905267, 286.2787, 0.1970505, 0.9004903, 0.1373157, 55.85495, 11.43741, 1.525748, 0.4299785, 239.9356 }.ToTorchTensor(new long[] { 1, 11 }).ToType(ATenScalarMapping.Float).RequiresGrad(true);

            var afterCat = inputs.Cat(1);
            var afterScaler = afterCat * scaler;
            var prediction = linear.Forward(afterScaler);

            var loss = MSE();
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

        private class CondModel : CustomModule
        {
            private Linear fb = Linear(1000, 100, false);
            private Linear fbT1 = Linear(100, 10, false);
            private Linear fbF1 = Linear(100, 50, false);
            private Linear fbF2 = Linear(50, 10, false);
            private bool _isTrue = false;

            public CondModel(string name, bool isTrue) : base(name)
            {
                _isTrue = isTrue;
                RegisterModule("fb", fb);
                RegisterModule("fbT1", fbT1);
                RegisterModule ("fbF1", fbF1);
                RegisterModule ("fbF2", fbF2);
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
            var modT = new CondModel("modT", true);
            var modF = new CondModel("modF", false);

            var psT = modT.GetParameters();
            Assert.Equal(4, psT.Length);

            var psF = modF.GetParameters();
            Assert.Equal(4, psF.Length);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0", requiresGrad: true);
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0", requiresGrad: true);

            modT.Train();

            var eval = modT.Forward(x);
            var loss = MSE(NN.Reduction.Sum);
            var output = loss(eval, y);

            modT.ZeroGrad();

            output.Backward();
            var gradCounts = 0;

            foreach (var parm in modT.GetParameters())
            {
                var grad = parm.Grad();
                gradCounts += grad.Handle == IntPtr.Zero ? 0 : 1;
            }

            Assert.Equal(2, gradCounts);

            //{ "grad can be implicitly created only for scalar outputs (_make_grads at ..\\..\\torch\\csrc\\autograd\\autograd.cpp:47)\n(no backtrace available)"}
            modF.Train();

            eval = modF.Forward(x);
            output = loss(eval, y);

            modF.ZeroGrad();

            output.Backward();
            gradCounts = 0;

            foreach (var parm in modF.GetParameters())
            {
                var grad = parm.Grad();
                gradCounts += grad.Handle == IntPtr.Zero ? 0 : 1;
            }

            Assert.Equal(3, gradCounts);
        }

        [Fact(Skip = "Not working on MacOS (note: may now be working, we need to recheck)")]
        public void TestAutoGradMode()
        {
            var x = FloatTensor.RandomN(new long[] { 2, 3 }, device: "cpu:0", requiresGrad: true);
            using (var mode = new AutoGradMode(false))
            {
                Assert.False(AutoGradMode.IsAutogradEnabled());
                var sum = x.Sum();
                Assert.Throws<ExternalException>(() => sum.Backward());
                //var grad = x.Grad();
                //Assert.True(grad.Handle == IntPtr.Zero);
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
        public void TestSaveLoadLinear()
        {
            if (File.Exists (".model.ts")) File.Delete (".model.ts");
            var linear = Linear(100, 10, true);
            linear.Save(".model.ts");
            var loadedLinear = NN.Linear.Load(".model.ts");
            File.Delete(".model.ts");
            Assert.NotNull(loadedLinear);
        }

        [Fact]
        public void TestSaveLoadConv2D()
        {
            if (File.Exists (".model.ts")) File.Delete (".model.ts");
            var conv = Conv2D(100, 10, 5);
            conv.Save(".model.ts");
            var loaded = NN.Conv2D.Load(".model.ts");
            File.Delete(".model.ts");
            Assert.NotNull(loaded);
        }

        [Fact]
        public void TestSaveLoadSequence()
        {
            if (File.Exists (".model-list.txt")) File.Delete (".model-list.txt");
            var lin1 = Linear(100, 10, true);
            var lin2 = Linear(10, 5, true);
            var seq = Sequential(("lin1", lin1), ("lin2", lin2));
            seq.Save(".model-list.txt");
            var loaded = NN.Sequential.Load(".model-list.txt");
            File.Delete("model-list.txt");
            Assert.NotNull(loaded);
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
            Assert.Equal("test", name);
            Assert.True(module.HasParameter("test"));

            var ps = module.GetParameters();
            Assert.Equal(1, ps.Length);
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

        private class TestModule : CustomModule
        {
            public TestModule(string name, TorchTensor tensor, bool withGrad)
                : base(name, new Parameter(name, tensor, withGrad))
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
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", Relu()), ("lin2", lin2));

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            float learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var loss = MSE(NN.Reduction.Sum);

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
                    foreach (var param in seq.GetParameters())
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
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
                        var seq = Sequential(("lin1", lin1), ("relu1", Relu()), ("lin2", lin2));

            double learning_rate = 0.00001;

            var optimizer = NN.Optimizer.Adam(seq.GetParameters(), learning_rate);

            Assert.NotNull(optimizer);
        }

        /// <summary>
        /// Fully connected Relu net with one hidden layer trained using Adam optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TestTrainingAdam()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", Relu()), ("lin2", lin2));

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            double learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var optimizer = NN.Optimizer.Adam(seq.GetParameters(), learning_rate);
            var loss = MSE(NN.Reduction.Sum);

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

        [Fact]
        public void AvgPool2DObjectInitialized()
        {
            TorchTensor ones = FloatTensor.Ones(new long[] { 2, 2, 2 });
            var obj = Functions.AvgPool2D(ones, new long[] { 2, 2 }, new long[] { 2, 2 });
            Assert.Equal(typeof(TorchTensor), obj.GetType());
        }

        [Fact]
        public void MaxPool2DObjectInitialized()
        {
            TorchTensor ones = FloatTensor.Ones(new long[] { 2, 2, 2 });
            var obj = Functions.MaxPool2D(ones, new long[] { 2, 2 }, new long[] { 2, 2 });
            Assert.Equal(typeof(TorchTensor), obj.GetType());
        }

        [Fact]
        public void SinTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Sin).ToArray();
            var res = FloatTensor.From(data).Sin();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void CosTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Cos).ToArray();
            var res = FloatTensor.From(data).Cos();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void TanTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Tan).ToArray();
            var res = FloatTensor.From(data).Tan();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void SinhTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Sinh).ToArray();
            var res = FloatTensor.From(data).Sinh();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void CoshTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Cosh).ToArray();
            var res = FloatTensor.From(data).Cosh();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void TanhTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Tanh).ToArray();
            var res = FloatTensor.From(data).Tanh();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void AsinTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Asin).ToArray();
            var res = FloatTensor.From(data).Asin();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void AcosTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Acos).ToArray();
            var res = FloatTensor.From(data).Acos();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void AtanTest()
        {
            var data = new float[] { 1.0f, 0.2f, -0.1f };
            var expected = data.Select(MathF.Atan).ToArray();
            var res = FloatTensor.From(data).Atan();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void LogTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(x => MathF.Log(x)).ToArray();
            var res = FloatTensor.From(data).Log();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void Log10Test()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = data.Select(MathF.Log10).ToArray();
            var res = FloatTensor.From(data).Log10();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void FloorTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(MathF.Floor).ToArray();
            var res = FloatTensor.From(data).Floor();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void CeilTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(MathF.Ceiling).ToArray();
            var res = FloatTensor.From(data).Ceil();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void RoundTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };
            var expected = data.Select(x => MathF.Round(x)).ToArray();
            var res = FloatTensor.From(data).Round();
            Assert.True(res.AllClose(FloatTensor.From(expected)));
        }

        [Fact]
        public void ExpandTest()
        {
            TorchTensor ones = FloatTensor.Ones(new long[] { 2 });
            TorchTensor onesExpanded = ones.Expand(new long[] { 3, 2 });

            Assert.Equal(onesExpanded.Shape, new long[] { 3, 2 });
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Assert.Equal(1.0, onesExpanded[i, j].DataItem<float>());
                }
            }
        }

        [Fact]
        public void TopKTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res1 = FloatTensor.From(data).TopK(1);
            var res1_0 = res1.values[0].DataItem<float>();
            var index1_0 = res1.indexes[0].DataItem<long>();
            Assert.Equal(3.1f, res1_0);
            Assert.Equal(2L, index1_0);

            var res2 = FloatTensor.From(data).TopK(2, sorted: true);
            var res2_0 = res2.values[0].DataItem<float>();
            var index2_0 = res2.indexes[0].DataItem<long>();
            var res2_1 = res2.values[1].DataItem<float>();
            var index2_1 = res2.indexes[1].DataItem<long>();
            Assert.Equal(3.1f, res2_0);
            Assert.Equal(2L, index2_0);
            Assert.Equal(2.0f, res2_1);
            Assert.Equal(1L, index2_1);
        }

        [Fact]
        public void SumTest()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f };

            var res1 = FloatTensor.From(data).Sum();
            var res1_0 = res1.DataItem<float>();
            Assert.Equal(6.0f, res1_0);

            var res2 = FloatTensor.From(data).Sum(type: ATenScalarMapping.Double);
            var res2_0 = res2.DataItem<double>();
            Assert.Equal(6.0, res2_0);

            // summing integers gives long unless type is explicitly specified
            var dataInt32 = new int[] { 1, 2, 3 };
            var res3 = IntTensor.From(dataInt32).Sum();
            Assert.Equal(ATenScalarMapping.Long, res3.Type);
            var res3_0 = res3.DataItem<long>();
            Assert.Equal(6L, res3_0);

            // summing integers gives long unless type is explicitly specified
            var res4 = IntTensor.From(dataInt32).Sum(type: ATenScalarMapping.Int);
            Assert.Equal(ATenScalarMapping.Int, res4.Type);
            var res4_0 = res4.DataItem<int>();
            Assert.Equal(6L, res4_0);

        }

        [Fact]
        public void UnbindTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = FloatTensor.From(data).Unbind();
            Assert.Equal(3, res.Length);
            Assert.Equal(new long[] { }, res[0].Shape);
            Assert.Equal(new long[] { }, res[1].Shape);
            Assert.Equal(new long[] { }, res[2].Shape);
            Assert.Equal(1.1f, res[0].DataItem<float>());
            Assert.Equal(2.0f, res[1].DataItem<float>());
            Assert.Equal(3.1f, res[2].DataItem<float>());
        }

        [Fact]
        public void SplitWithSizesTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = FloatTensor.From(data).SplitWithSizes(new long[] { 2, 1 });
            Assert.Equal(2, res.Length);
            Assert.Equal(new long[] { 2 }, res[0].Shape);
            Assert.Equal(new long[] { 1 }, res[1].Shape);
            Assert.Equal(1.1f, res[0][0].DataItem<float>());
            Assert.Equal(2.0f, res[0][1].DataItem<float>());
            Assert.Equal(3.1f, res[1][0].DataItem<float>());
        }

        [Fact]
        public void RandomTest()
        {
            var res = FloatTensor.Random(new long[] { 2 });
            Assert.Equal(new long[] { 2 }, res.Shape);

            var res1 = ShortTensor.RandomIntegers(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res1.Shape);

            var res2 = IntTensor.RandomIntegers(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res2.Shape);

            var res3 = LongTensor.RandomIntegers(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res3.Shape);

            var res4 = ByteTensor.RandomIntegers(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res4.Shape);

            var res5 = SByteTensor.RandomIntegers(10, new long[] { 200 });
            Assert.Equal(new long[] { 200 }, res5.Shape);
        }

        [Fact]
        public void SqueezeTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = FloatTensor.From(data).Expand(new long[] { 1, 1, 3 }).Squeeze(0).Squeeze(0);
            Assert.Equal(new long[] { 3 }, res.Shape);
            Assert.Equal(1.1f, res[0].DataItem<float>());
            Assert.Equal(2.0f, res[1].DataItem<float>());
            Assert.Equal(3.1f, res[2].DataItem<float>());
        }

        [Fact]
        public void NarrowTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f };

            var res = FloatTensor.From(data).Narrow(0, 1, 2);
            Assert.Equal(new long[] { 2 }, res.Shape);
            Assert.Equal(2.0f, res[0].DataItem<float>());
            Assert.Equal(3.1f, res[1].DataItem<float>());
        }

        [Fact]
        public void SliceTest()
        {
            var data = new float[] { 1.1f, 2.0f, 3.1f, 4.0f };

            var res = FloatTensor.From(data).Slice(0, 1, 1, 1);
            Assert.Equal(new long[] { 0 }, res.Shape);

            var res2 = FloatTensor.From(data).Slice(0, 1, 2, 1);
            Assert.Equal(new long[] { 1 }, res2.Shape);
            Assert.Equal(2.0f, res2[0].DataItem<float>());

            var res3 = FloatTensor.From(data).Slice(0, 1, 4, 2);
            Assert.Equal(new long[] { 2 }, res3.Shape);
            Assert.Equal(2.0f, res3[0].DataItem<float>());
            Assert.Equal(4.0f, res3[1].DataItem<float>());
        }
        [Fact]
        public void Conv1DTest()
        {
            var t1 =
                new float[3, 4, 5]
                   {{{0.3460f, 0.4414f, 0.2384f, 0.7905f, 0.2267f},
                                     {0.5161f, 0.9032f, 0.6741f, 0.6492f, 0.8576f},
                                     {0.3373f, 0.0863f, 0.8137f, 0.2649f, 0.7125f},
                                     {0.7144f, 0.1020f, 0.0437f, 0.5316f, 0.7366f}},

                                    {{0.9871f, 0.7569f, 0.4329f, 0.1443f, 0.1515f},
                                     {0.5950f, 0.7549f, 0.8619f, 0.0196f, 0.8741f},
                                     {0.4595f, 0.7844f, 0.3580f, 0.6469f, 0.7782f},
                                     {0.0130f, 0.8869f, 0.8532f, 0.2119f, 0.8120f}},

                                    {{0.5163f, 0.5590f, 0.5155f, 0.1905f, 0.4255f},
                                     {0.0823f, 0.7887f, 0.8918f, 0.9243f, 0.1068f},
                                     {0.0337f, 0.2771f, 0.9744f, 0.0459f, 0.4082f},
                                     {0.9154f, 0.2569f, 0.9235f, 0.9234f, 0.3148f}}};
            var t2 =
                new float[2, 4, 3]
                   {{{0.4941f, 0.8710f, 0.0606f},
                     {0.2831f, 0.7930f, 0.5602f},
                     {0.0024f, 0.1236f, 0.4394f},
                     {0.9086f, 0.1277f, 0.2450f}},

                    {{0.5196f, 0.1349f, 0.0282f},
                     {0.1749f, 0.6234f, 0.5502f},
                     {0.7678f, 0.0733f, 0.3396f},
                     {0.6023f, 0.6546f, 0.3439f}}};

            var t1raw = new float[3 * 4 * 5];
            var t2raw = new float[2 * 4 * 3];
            { for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 5; k++) { t1raw[i * 4 * 5 + j * 5 + k] = t1[i, j, k]; } }
            { for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 3; k++) { t2raw[i * 4 * 3 + j * 3 + k] = t2[i, j, k]; } }
            var t1t = FloatTensor.From(t1raw, new long[] { 3, 4, 5 });
            var t2t = FloatTensor.From(t2raw, new long[] { 2, 4, 3 });
            var t3t = t1t.Conv1D(t2t);

            // Check the answer
            var t3Correct =
                new float[3, 2, 3]
                    {{{2.8516f, 2.0732f, 2.6420f},
                      {2.3239f, 1.7078f, 2.7450f}},

                    {{3.0127f, 2.9651f, 2.5219f},
                     {3.0899f, 3.1496f, 2.4110f}},

                    {{3.4749f, 2.9038f, 2.7131f},
                     {2.7692f, 2.9444f, 3.2554f}}};
            {
                var data = t3t.Data<float>();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 2; j++)
                        for (int k = 0; k < 3; k++)
                        {
                            var itemCorrect = t3Correct[i, j, k];
                            var item = data[i * 2 * 3 + j * 3 + k];
                            Assert.True(Math.Abs(itemCorrect - item) < 0.01f);
                        }
            }
            
            var t3p2d3 = t1t.Conv1D(t2t, padding: 2, dilation: 3);

            // Check the answer
            var t3p2d3Correct =
                new float[3, 2, 3]
                    {{{ 2.1121f, 0.8484f, 2.2709f},
                      {1.6692f, 0.5406f, 1.8381f}},

                     {{2.5078f, 1.2137f, 0.9173f},
                      {2.2395f, 1.1805f, 1.1954f}},

                     {{1.5215f, 1.3946f, 2.1327f},
                      {1.0732f, 1.3014f, 2.0696f}}};
            {
                var data = t3p2d3.Data<float>();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 2; j++)
                        for (int k = 0; k < 3; k++)
                        {
                            var itemCorrect = t3p2d3Correct[i, j, k];
                            var item = data[i * 2 * 3 + j * 3 + k];
                            Assert.True(Math.Abs(itemCorrect - item) < 0.01f);
                        }
            }
        }
    }
}
