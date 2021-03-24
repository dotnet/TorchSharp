// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using static TorchSharp.NN.Functions;
using TorchSharp.Tensor;
using Xunit;

#nullable enable

namespace TorchSharp
{
    public class TestNN
    {

        [Fact]
        public void CreateLinear()
        {
            var lin = Linear(1000, 100);
            Assert.NotNull(lin);
            Assert.True(!(lin.Bias is null));
            //var name = lin.GetName();

            var ps = lin.parameters();
            Assert.Equal(2, ps.Length);
        }

        [Fact]
        public void TestGetBiasInLinear()
        {
            var lin = Linear(1000, 100, false);
            var ps = lin.parameters();
            var nps = ps.Length;
            Assert.Equal(1, nps);
            Assert.True(lin.Bias is null);

            var lin2 = Linear(1000, 100, true);
            Assert.True(!(lin2.Bias is null));
        }

        [Fact]
        public void TestSetGetBiasInLinear()
        {
            var lin = Linear(1000, 100, true);
            var bias = Float32Tensor.ones(new long[] { 1000 });
            lin.Bias = bias;
            Assert.True(!(lin.Bias is null));

            Assert.Equal(lin.Bias?.NumberOfElements, bias.NumberOfElements);
        }

        [Fact]
        public void TestWeightAndBiasShapeInLinear()
        {
            var lin = Linear(1000, 100, true);

            Assert.Equal(2, lin.Weight.shape.Length);
            Assert.Equal(100, lin.Weight.shape[0]);
            Assert.Equal(1000, lin.Weight.shape[1]);
            Assert.True(1 == lin.Bias?.shape.Length);
            Assert.Equal(100, lin.Bias?.shape[0]);
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
            Assert.Equal(2, weight.shape.Length);
            Assert.Equal(100, weight.shape[0]);
            Assert.Equal(1000, weight.shape[1]);
            Assert.True(1 == bias.shape.Length);
            Assert.Equal(100, bias.shape[0]);
        }

        [Fact]
        public void TestLinearWithBias()
        {
            var lin = Linear(1000, 100, true);
            var bias = lin.Bias!;
            var weight = lin.Weight.t();
            var input = Float32Tensor.randn(new long[] { 1, 1000 });
            var forward = lin.forward(input);
            var matmul = input.matmul(weight).add(bias);

            Assert.Equal(forward.shape.Length, matmul.shape.Length);
            Assert.Equal(forward.shape[0], matmul.shape[0]);
            Assert.Equal(forward.shape[1], matmul.shape[1]);

            for (int i = 0; i < 100; i++) {
                Assert.InRange(forward.Data<float>()[i], matmul.Data<float>()[i] - 10e5f, matmul.Data<float>()[i] + 10e5f);
            }
        }

        [Fact]
        public void TestLinearNoBias()
        {
            var lin = Linear(1000, 100, false);
            Assert.False(!(lin.Bias is null));

            var weight = lin.Weight.transpose(0, 1);
            var input = Float32Tensor.randn(new long[] { 1, 1000 });
            var forward = lin.forward(input);
            var matmul = input.matmul(weight);

            Assert.Equal(forward.shape.Length, matmul.shape.Length);
            Assert.Equal(forward.shape[0], matmul.shape[0]);
            Assert.Equal(forward.shape[1], matmul.shape[1]);

            for (int i = 0; i < 100; i++) {
                Assert.Equal(forward.Data<float>()[i], matmul.Data<float>()[i]);
            }
        }

        [Fact]
        public void TestIdentity()
        {
            var lin = Identity();

            var input = Float32Tensor.randn(new long[] { 1, 1000 });
            var output = lin.forward(input);

            for (int i = 0; i < 1000; i++) {
                Assert.Equal(input.Data<float>()[i], output.Data<float>()[i]);
            }
        }

        [Fact]
        public void TestLinearEditBias()
        {
            var lin = Linear(1000, 100, true);
            var bias = Float32Tensor.randn(new long[] { 100 });
            lin.Bias = bias;

            for (int i = 0; i < 100; i++) {
                Assert.Equal(lin.Bias.Data<float>()[i], bias.Data<float>()[i]);
            }
        }

        [Fact]
        public void TestLinearEditWeightsAndBias()
        {
            var lin = Linear(1000, 1000, true);
            var bias = Float32Tensor.randn(new long[] { 100 });
            var weights = Float32Tensor.randn(new long[] { 100, 1000 });

            lin.Bias = bias;
            lin.Weight = weights;

            Assert.Equal(lin.Weight.shape.Length, weights.shape.Length);
            Assert.Equal(lin.Weight.shape[0], weights.shape[0]);
            Assert.Equal(lin.Weight.shape[1], weights.shape[1]);

            for (int i = 0; i < 100; i++) {
                Assert.Equal(lin.Bias.Data<float>()[i], bias.Data<float>()[i]);
            }
        }

        [Fact]
        public void TestLinearEditWeightsAndBiasGetParameters()
        {
            var lin = Linear(1000, 1000, true);
            var bias = Float32Tensor.randn(new long[] { 100 });
            var weights = Float32Tensor.randn(new long[] { 1000, 1000 });
            lin.Bias = bias;
            lin.Weight = weights;

            var parameters = lin.parameters().ToArray();

            Assert.Equal(lin.Weight.shape.Length, parameters[0].shape.Length);
            Assert.Equal(lin.Weight.shape[0], parameters[0].shape[0]);
            Assert.Equal(lin.Weight.shape[1], parameters[0].shape[1]);
        }

        [Fact]
        public void CreateRelu()
        {
            var rel = ReLU();
            Assert.NotNull(rel);
            var modules = rel.GetName();
        }

        [Fact]
        public void EvaluateRelu()
        {
            var rel = ReLU();
            var input = Float32Tensor.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= 0.0));
        }

        [Fact]
        public void EvaluateRelu6()
        {
            var rel = ReLU6();
            var input = Float32Tensor.randn(new long[] { 64, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= 0.0 && val <= 6.0));
        }

        [Fact]
        public void EvaluateLeakyRelu()
        {
            var rel = LeakyReLU();
            var input = Float32Tensor.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
        }

        [Fact]
        public void EvaluateRRelu()
        {
            var rel = RReLU();
            var input = Float32Tensor.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
        }

        [Fact]
        public void EvaluateCELU()
        {
            var rel = CELU();
            var input = Float32Tensor.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -1.0));
        }

        [Fact]
        public void EvaluateELU()
        {
            var rel = ELU();
            var input = Float32Tensor.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -1.0));
        }

        [Fact]
        public void EvaluateSELU()
        {
            var rel = SELU();
            var input = Float32Tensor.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -1.76));
        }

        [Fact]
        public void EvaluateGELU()
        {
            var rel = GELU();
            var input = Float32Tensor.randn(new long[] { 64, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -0.2));
        }

        [Fact]
        public void EvaluateSigmoid()
        {
            var rel = Sigmoid();
            var input = Float32Tensor.randn(new long[] { 64, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= 0.0 && val <= 1.0));
        }

        [Fact]
        public void EvaluateSiLU()
        {
            var rel = SiLU();
            var input = Float32Tensor.randn(new long[] { 64, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -1.0));
        }

        [Fact]
        public void EvaluateSoftmax2d()
        {
            var rel = Softmax2d();
            var input = Float32Tensor.randn(new long[] { 64, 3, 8, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= 0.0 && val <= 1.0));
        }

        [Fact]
        public void EvaluateTanh()
        {
            var rel = Tanh();
            var input = Float32Tensor.randn(new long[] { 64, 3, 8, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -1.0 && val <= 1.0));
        }

        [Fact]
        public void EvaluateSoftmax()
        {
            var rel = Softmax(1);
            var input = Float32Tensor.randn(new long[] { 64, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= 0.0 && val <= 1.0));
        }

        [Fact]
        public void EvaluateSoftmin()
        {
            var rel = Softmax(1);
            var input = Float32Tensor.randn(new long[] { 64, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.Data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= 0.0 && val <= 1.0));
        }

        [Fact]
        public void EvalSequence()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", ReLU()),
                ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 }, requiresGrad: true);
            var eval = seq.forward(x);
        }

        [Fact]
        public void CreateSequence()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", ReLU()),
                ("lin2", lin2));
            var parameters = seq.parameters();
            var parametersCount = parameters.Count();
            Assert.Equal(4, parametersCount);

            var namedParams = seq.parameters();
            var namedParamsCount = namedParams.Count();
            Assert.Equal(4, namedParamsCount);
        }

        [Fact]
        public void EvalLossSequence()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", ReLU()),
                ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            var eval = seq.forward(x);
            var loss = mse_loss(NN.Reduction.Sum);
            var output = loss(eval, y);

            var result = output.ToSingle();
        }

        [Fact]
        public void TestPoissonNLLLoss()
        {
            using (TorchTensor input = Float32Tensor.from(new float[] { 0.5f, 1.5f, 2.5f }))
            using (TorchTensor target = Float32Tensor.from(new float[] { 1f, 2f, 3f })) {
                var componentWiseLoss = ((TorchTensor)input.exp()) - target * input;
                Assert.True(componentWiseLoss.Equals(poisson_loss(reduction: NN.Reduction.None)(input, target)));
                Assert.True(componentWiseLoss.sum().Equals(poisson_loss(reduction: NN.Reduction.Sum)(input, target)));
                Assert.True(componentWiseLoss.mean().Equals(poisson_loss(reduction: NN.Reduction.Mean)(input, target)));
            }
        }

        [Fact]
        public void TestPoissonNLLLoss2()
        {
            using (TorchTensor input = Float32Tensor.rand(new long[] { 5, 2 }))
            using (TorchTensor target = Float32Tensor.rand(new long[] { 5, 2 })) {
                var outTensor = poisson_loss(true, true)(input, target);
                var values = outTensor.Data<float>().ToArray();
                Assert.Empty(outTensor.shape);
                Assert.Single(values);
            }
        }

        [Fact]
        public void TestCrossEntropyLoss()
        {
            using (TorchTensor input = Float32Tensor.rand(new long[] { 5, 12 }))
            using (TorchTensor target = Int64Tensor.randint(12, new long[] { 5 })) {
                var outTensor = cross_entropy_loss()(input, target);
                var values = outTensor.Data<float>().ToArray();
                Assert.Empty(outTensor.shape);
                Assert.Single(values);
            }
        }

        [Fact]
        public void TestL1Loss()
        {
            using (TorchTensor input = Float32Tensor.rand(new long[] { 5, 2 }))
            using (TorchTensor target = Float32Tensor.rand(new long[] { 5, 2 })) {
                var outTensor = l1_loss()(input, target);
                var values = outTensor.Data<float>().ToArray();
                Assert.Empty(outTensor.shape);
                Assert.Single(values);
            }
        }

        [Fact]
        public void TestBinaryCrossEntropyLoss()
        {
            var m = Sigmoid();
            using (TorchTensor input = Float32Tensor.randn(new long[] { 3 }))
            using (TorchTensor target = Float32Tensor.randn(new long[] { 3 })) {
                var outTensor = binary_cross_entropy_loss()(m.forward(input), target);
                var values = outTensor.Data<float>().ToArray();
                Assert.Empty(outTensor.shape);
                Assert.Single(values);
            }
        }

        [Fact]
        public void TestBinaryCrossEntropyLossWithLogits()
        {
            using (TorchTensor input = Float32Tensor.randn(new long[] { 3 }))
            using (TorchTensor target = Float32Tensor.randn(new long[] { 3 })) {
                var outTensor = binary_cross_entropy_with_logits_loss()(input, target);
                var values = outTensor.Data<float>().ToArray();
                Assert.Empty(outTensor.shape);
                Assert.Single(values);
            }
        }

#if DEBUG
        [Fact(Skip = "Not working on Mac and Ubuntu (note: may now be working, we need to recheck)")]
        public void TestErrorHandling()
        {
            using (TorchTensor input = Float32Tensor.from(new float[] { 0.5f, 1.5f }))
            using (TorchTensor target = Float32Tensor.from(new float[] { 1f, 2f, 3f })) {
                Assert.Throws<ExternalException>(() => poisson_loss()(input, target));
            }
        }
#endif

        [Fact]
        public void TestBackward()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", ReLU()),
                ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 }, requiresGrad: true);
            var y = Float32Tensor.randn(new long[] { 64, 10 }, requiresGrad: true);

            var eval = seq.forward(x);
            var loss = mse_loss(NN.Reduction.Sum);
            var output = loss(eval, y);

            seq.ZeroGrad();

            output.backward();
        }

        [Fact]
        public void TestGettingParameters()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", ReLU()),
                ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 }, requiresGrad: true);
            var y = Float32Tensor.randn(new long[] { 64, 10 }, requiresGrad: true);

            var eval = seq.forward(x);
            var loss = mse_loss(NN.Reduction.Sum);
            var output = loss(eval, y);

            seq.ZeroGrad();

            output.backward();

            foreach (var parm in seq.parameters()) {
            }
        }

        [Fact]
        public void TestGrad()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", ReLU()),
                ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 }, requiresGrad: true);
            var y = Float32Tensor.randn(new long[] { 64, 10 }, requiresGrad: true);

            var eval = seq.forward(x);
            var loss = mse_loss(NN.Reduction.Sum);
            var output = loss(eval, y);

            seq.ZeroGrad();

            output.backward();

            foreach (var parm in seq.parameters()) {
                var grad = parm.grad();
            }
        }

        [Fact]
        public void TestGrad2()
        {
            var y = Float32Tensor.randn(new long[] { 32, 1 });
            var input = new double[] { -2.75, 0.77, -0.61, 0.14, 1.39, 0.38, -0.53, -0.5, -2.13, -0.39, 0.46, -0.61, -0.37, -0.12, 0.55, -1, 0.84, -0.02, 1.3, -0.24, -0.5, -2.12, -0.85, -0.91, 1.81, 0.02, -0.78, -1.41, -1.09, -0.65, 0.9, -0.37, -0.22, 0.28, 1.05, -0.24, 0.3, -0.99, 0.19, 0.32, -0.95, -1.19, -0.63, 0.75, 0.16, 0.15, 0.3, -0.69, 0.2, -0.4, -0.67, 0.18, -1.43, -0.61, -0.78, -0.11, -1.07, -1.71, -0.45, -0.6, 0.05, -1.59, 1.24, 0.62, 0.01, 1.35, -0.9, -1.25, 1.62, -1.45, 0.92, 1.51, -0.19, -1.33, -0.01, -0.13, 0.1, -1.34, 1.23, 0.57, -0.24, 0.5, 0.71, -0.15, -1.37, -1.03, 1.8, 1.4, -0.63, 0.8, -0.97, -0.64, 0.51, 0.52, 0.95, 0.86, 0.43, 0.73, -1.38, -0.56, 0.44, 1.2, -1.45, -0.07, 1.88, 1.57, 0.38, -2.2, -0.56, -1.52, -0.17, 1.38, -1.02, -1.61, -0.13, -0.44, -0.37, 0.23, 1.75, 0.83, -0.02, -1.91, -0.23, -0.47, -1.41, -1.01, -0.91, -0.56, -1.72, 1.47, 0.31, 0.24, 0.48, 2.06, 0.07, -0.96, 1.03, -0.4, -0.64, -0.85, 0.42, -0.33, 0.85, -0.11, -1.24, -0.71, -1.04, -0.37, -0.37, 0.84, -0.9, -1.63, -2.91, -0.71, 0.09, 1.64, -1.1, -1.05, 0.51, 0.57, 0.19, 0.36, 1.36, 1.45, 0.35, -1.66, -0.65, 0.47, 1.95, -0.32, 0.19, -2.06, 0.5, 1.03, 0.94, -0.65, -2.94, 0.41, 1.13, 0.95, -0.02, 1.12, 0.19, 0.66, -0.77, -0.39, 0.59, -1.58, -0.67, 0.88, 0.26, -0.63, 0.49, 1.38, 1.48, -0.55, 0.4, 0.65, 0.19, 0.25, 0.03, -0.31, 0.75, 2.16, -1.36, 0.05, 0.22, 0.65, 1.28, 0.42, 1.35, -0.08, 1.1, 0.25, 0.44, 1.06, -1.78, 0.47, 1.38, 0.43, -1.56, 0.14, -0.22, 1.48, 0.04, 0.33, 0.1, 0.2, -0.99, 1.04, 0.61, -0.4, 0.96, 0.4, 0.5, 0.1, 0.02, 0.01, 0.22, 1.45, -0.77, 0.69, 0.95, 0.96, -0.09, -0.26, 0.22, -1.61, 1.86, -0.06, -0.34, -0.35, 0.55, -1.08, 1.29, 0.92, 0.16, 0.55, -0.01, 0.2, -0.61, -0.28, -2.17, -0.46, 1.63, 1.61, 0.64, 0.32, -0.75, 0.33, 0.3, -1.15, 0.42, -0.06, -1.14, 1.62, -0.9, -0.39, 0.4, 1.52, -0.43, 1.22, -0.32, -0.02, 1, -0.92, 0.11, 0.8, -0.99, -0.26, -2.85, -1.13, 0.49, -0.63, -0.54, -0.86, -0.97, -0.9, 0.23, 1.26, -1.78, -0.84, -0.48, 0.35, -1.13, -2.23, 0.1, 0.95, 1.27, 0.08, -2.21, 0.67, -0.2, 0.6, -1.14, 0.65, -0.73, -0.01, 0.9, -1.33, -1.16, 0.29, 1.16, 1.19, 0.84, 0.66, -1.55, -0.58, 1.85, -1.16, -0.95, 0.98, -0.1, -1.47, 0.78, -0.75, -1.32, 0.61, -0.5, -1, -0.42, 0.96, -1.39, 0.08, -1.82, 0.51, -0.71, -0.02, 2.32, -0.71, 0.08, -1.07 }.ToTorchTensor(new long[] { 32, 11 }).to_type(ScalarType.Float32);
            var inputs = new TorchTensor[] { input };
            var scaler = new double[] { 0.2544529, 0.3184713, 0.2597403, 0.3246753, 0.3144654, 0.3322259, 0.3436426, 0.3215434, 0.308642, 0.3154574, 0.3448276 }.ToTorchTensor(new long[] { 1, 11 }).to_type(ScalarType.Float32).with_requires_grad();
            var linear = Linear(11, 1, true);
            linear.Bias = new double[] { 373.8864 }.ToTorchTensor(new long[] { 1, 1 }).to_type(ScalarType.Float32).with_requires_grad();
            linear.Weight = new double[] { 300.2818, -0.5905267, 286.2787, 0.1970505, 0.9004903, 0.1373157, 55.85495, 11.43741, 1.525748, 0.4299785, 239.9356 }.ToTorchTensor(new long[] { 1, 11 }).to_type(ScalarType.Float32).with_requires_grad();

            var afterCat = inputs.cat(1);
            var afterScaler = afterCat * scaler;
            var prediction = linear.forward(afterScaler);

            var loss = mse_loss();
            var output = loss(prediction, y);

            linear.ZeroGrad();

            output.backward();

            var scalerGrad = scaler.grad();
            var weightGrad = linear.Weight.grad();
            var biasGrad = linear.Bias.grad();
            Assert.True(scalerGrad.shape.Length == 2);
            Assert.True(weightGrad.shape.Length == 2);
            Assert.True(biasGrad.shape.Length == 2);
        }

        [Fact]
        public void TestSetGrad()
        {
            var x = Float32Tensor.rand(new long[] { 10, 10 });
            Assert.False(x.requires_grad);

            x.requires_grad = true;
            Assert.True(x.requires_grad);
            x.requires_grad = false;
            Assert.False(x.requires_grad);
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
                RegisterModule("fbF1", fbF1);
                RegisterModule("fbF2", fbF2);
            }

            public override TorchTensor forward(TorchTensor input)
            {
                using (var x = fb.forward(input))
                    if (_isTrue) {
                        return fbT1.forward(x);
                    } else {
                        return fbF2.forward(fbF1.forward(x));
                    }
            }
        }

        [Fact]
        public void TestGradConditional()
        {
            var modT = new CondModel("modT", true);
            var modF = new CondModel("modF", false);

            var psT = modT.parameters();
            Assert.Equal(4, psT.Length);

            var psF = modF.parameters();
            Assert.Equal(4, psF.Length);

            var x = Float32Tensor.randn(new long[] { 64, 1000 }, requiresGrad: true);
            var y = Float32Tensor.randn(new long[] { 64, 10 }, requiresGrad: true);

            modT.Train();

            var eval = modT.forward(x);
            var loss = mse_loss(NN.Reduction.Sum);
            var output = loss(eval, y);

            modT.ZeroGrad();

            output.backward();
            var gradCounts = 0;

            foreach (var parm in modT.parameters()) {
                var grad = parm.grad();
                gradCounts += grad.Handle == IntPtr.Zero ? 0 : 1;
            }

            Assert.Equal(2, gradCounts);

            //{ "grad can be implicitly created only for scalar outputs (_make_grads at ..\\..\\torch\\csrc\\autograd\\autograd.cpp:47)\n(no backtrace available)"}
            modF.Train();

            eval = modF.forward(x);
            output = loss(eval, y);

            modF.ZeroGrad();

            output.backward();
            gradCounts = 0;

            foreach (var parm in modF.parameters()) {
                var grad = parm.grad();
                gradCounts += grad.Handle == IntPtr.Zero ? 0 : 1;
            }

            Assert.Equal(3, gradCounts);
        }

        [Fact(Skip = "Not working on MacOS (note: may now be working, we need to recheck)")]
        public void TestAutoGradMode()
        {
            var x = Float32Tensor.randn(new long[] { 2, 3 }, requiresGrad: true);
            using (var mode = new AutoGradMode(false)) {
                Assert.False(AutoGradMode.IsAutogradEnabled());
                var sum = x.sum();
                Assert.Throws<ExternalException>(() => sum.backward());
                //var grad = x.Grad();
                //Assert.True(grad.Handle == IntPtr.Zero);
            }
            using (var mode = new AutoGradMode(true)) {
                Assert.True(AutoGradMode.IsAutogradEnabled());
                var sum = x.sum();
                sum.backward();
                var grad = x.grad();
                Assert.False(grad.Handle == IntPtr.Zero);
                var data = grad.Data<float>();
                for (int i = 0; i < 2 * 3; i++) {
                    Assert.Equal(1.0, data[i]);
                }
            }
        }

        [Fact]
        public void TestSaveLoadLinear()
        {
            if (File.Exists(".model.ts")) File.Delete(".model.ts");
            var linear = Linear(100, 10, true);
            linear.Save(".model.ts");
            var loadedLinear = NN.Linear.Load(".model.ts");
            File.Delete(".model.ts");
            Assert.NotNull(loadedLinear);
        }

        [Fact]
        public void TestConv1D()
        {
            var shape = new long[] { 16, 3, 28 };
            TorchTensor t = Float32Tensor.rand(shape);
            var conv = Conv1D(3, 64, 3);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(26, output.shape[2]);
        }

        [Fact]
        public void TestConv1DGetWeight()
        {
            var conv = Conv1D(3, 64, 3);
            var weight = conv.Weight;
            var bias = conv.Bias;
            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3 }, weight.shape);
        }

        [Fact]
        public void TestConv1DEditWeightAndBias()
        {
            var conv = Conv1D(3, 64, 3);

            conv.Bias = Float32Tensor.randn(new long[] { 64 });
            var weights = Float32Tensor.randn(new long[] { 64, 3, 3 });

            var weight = conv.Weight;
            var bias = conv.Bias;

            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3 }, weight.shape);

            for (int i = 0; i < 64; i++) {
                Assert.Equal(conv.Bias.Data<float>()[i], bias.Data<float>()[i]);
            }
        }

        [Fact]
        public void TestConv1DStride()
        {
            var shape = new long[] { 16, 3, 28 };
            TorchTensor t = Float32Tensor.rand(shape);
            var conv = Conv1D(3, 64, 3, stride: 2);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(13, output.shape[2]);
        }

        [Fact]
        public void TestConv1DPadding()
        {
            var shape = new long[] { 16, 3, 28 };
            TorchTensor t = Float32Tensor.rand(shape);

            using (var conv = Conv1D(3, 64, 3, padding: 1))
            using (var output = conv.forward(t)){
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
            }
            using (var conv = Conv1D(3, 64, 3, padding: 1, paddingMode: PaddingModes.Reflect))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
            }
        }

        [Fact]
        public void TestConv2D()
        {
            var shape = new long[] { 16, 3, 28, 28 };
            TorchTensor t = Float32Tensor.rand(shape);
            var conv = Conv2D(3, 64, 3);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(26, output.shape[2]);
            Assert.Equal(26, output.shape[3]);
        }

        [Fact]
        public void TestConv2DGetWeight()
        {
            var conv = Conv2D(3, 64, 3);
            var weight = conv.Weight;
            var bias = conv.Bias;
            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3, 3 }, weight.shape);
        }

        [Fact]
        public void TestConv2DEditWeightAndBias()
        {
            var conv = Conv2D(3, 64, 3);

            conv.Bias = Float32Tensor.randn(new long[] { 64 });
            var weights = Float32Tensor.randn(new long[] { 64, 3, 3, 3 });

            var weight = conv.Weight;
            var bias = conv.Bias;

            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3, 3 }, weight.shape);

            for (int i = 0; i < 64; i++) {
                Assert.Equal(conv.Bias.Data<float>()[i], bias.Data<float>()[i]);
            }
        }

        [Fact]
        public void TestConv2DStride()
        {
            var shape = new long[] { 16, 3, 28, 28 };
            TorchTensor t = Float32Tensor.rand(shape);
            var conv = Conv2D(3, 64, 3, stride: 2);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(13, output.shape[2]);
            Assert.Equal(13, output.shape[3]);
        }

        [Fact]
        public void TestConv2DPadding()
        {
            var shape = new long[] { 16, 3, 28, 28 };
            TorchTensor t = Float32Tensor.rand(shape);
            using (var conv = Conv2D(3, 64, 3, padding: 1))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
                Assert.Equal(28, output.shape[3]);
            }
            using (var conv = Conv2D(3, 64, 3, padding: 1, paddingMode: PaddingModes.Reflect))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
                Assert.Equal(28, output.shape[3]);
            }
        }

        [Fact]
        public void TestConv3D()
        {
            var shape = new long[] { 16, 3, 28, 28, 28 };
            TorchTensor t = Float32Tensor.rand(shape);
            var conv = Conv3D(3, 64, 3);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(26, output.shape[2]);
            Assert.Equal(26, output.shape[3]);
            Assert.Equal(26, output.shape[4]);
        }

        [Fact]
        public void TestConv3DGetWeight()
        {
            var conv = Conv3D(3, 64, 3);
            var weight = conv.Weight;
            var bias = conv.Bias;
            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3, 3, 3 }, weight.shape);
        }

        [Fact]
        public void TestConv3DEditWeightAndBias()
        {
            var conv = Conv3D(3, 64, 3);

            conv.Bias = Float32Tensor.randn(new long[] { 64 });
            var weights = Float32Tensor.randn(new long[] { 64, 3, 3, 3 });

            var weight = conv.Weight;
            var bias = conv.Bias;

            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3, 3, 3 }, weight.shape);

            for (int i = 0; i < 64; i++) {
                Assert.Equal(conv.Bias.Data<float>()[i], bias.Data<float>()[i]);
            }
        }

        [Fact]
        public void TestConv3DStride()
        {
            var shape = new long[] { 16, 3, 28, 28, 28 };
            TorchTensor t = Float32Tensor.rand(shape);
            var conv = Conv3D(3, 64, 3, stride: 2);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(13, output.shape[2]);
            Assert.Equal(13, output.shape[3]);
            Assert.Equal(13, output.shape[4]);
        }

        [Fact]
        public void TestConv3DPadding()
        {
            var shape = new long[] { 16, 3, 28, 28, 28 };
            TorchTensor t = Float32Tensor.rand(shape);
            using (var conv = Conv3D(3, 64, 3, padding: 1))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
                Assert.Equal(28, output.shape[3]);
                Assert.Equal(28, output.shape[4]);
            }
            using (var conv = Conv3D(3, 64, 3, padding: 1, paddingMode: PaddingModes.Replicate))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
                Assert.Equal(28, output.shape[3]);
                Assert.Equal(28, output.shape[4]);
            }
        }

        [Fact]
        public void TestConvTranspose1D()
        {
            var shape = new long[] { 16, 3, 28 };
            TorchTensor t = Float32Tensor.rand(shape);
            var conv = ConvTranspose1D(3, 64, 3);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(30, output.shape[2]);
        }

        [Fact]
        public void TestConvTranspose2D()
        {
            var shape = new long[] { 16, 3, 28, 28 };
            TorchTensor t = Float32Tensor.rand(shape);
            var conv = ConvTranspose2D(3, 64, 3);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(30, output.shape[2]);
            Assert.Equal(30, output.shape[3]);
        }

        [Fact]
        public void TestConvTranspose3D()
        {
            var shape = new long[] { 16, 3, 28, 28, 28 };
            TorchTensor t = Float32Tensor.rand(shape);
            var conv = ConvTranspose3D(3, 64, 3);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(30, output.shape[2]);
            Assert.Equal(30, output.shape[3]);
            Assert.Equal(30, output.shape[4]);
        }

        [Fact]
        public void TestSaveLoadConv2D()
        {
            if (File.Exists(".model.ts")) File.Delete(".model.ts");
            var conv = Conv2D(100, 10, 5);
            conv.Save(".model.ts");
            var loaded = NN.Conv2D.Load(".model.ts");
            File.Delete(".model.ts");
            Assert.NotNull(loaded);
        }

        [Fact]
        public void TestSaveLoadSequence()
        {
            if (File.Exists(".model-list.txt")) File.Delete(".model-list.txt");
            var lin1 = Linear(100, 10, true);
            var lin2 = Linear(10, 5, true);
            var seq = Sequential(("lin1", lin1), ("lin2", lin2));
            seq.Save(".model-list.txt");
            var loaded = NN.Sequential.Load(".model-list.txt");
            File.Delete("model-list.txt");
            Assert.NotNull(loaded);
        }

        [Fact]
        public void TestCustomModule()
        {
            var module = new TestModule("test", Float32Tensor.randn(new long[] { 2, 2 }), true);
            var name = module.GetName();
            Assert.NotNull(name);
            Assert.Equal("test", name);
            Assert.True(module.HasParameter("test"));

            var ps = module.parameters();
            var n = ps.Length;
            Assert.Equal(1, n);
        }

        [Fact]
        public void TestCustomModuleWithInPlaceModification()
        {
            var param = Float32Tensor.randn(new long[] { 1000, 100 });
            var module = new TestModule("test", param, true);

            Assert.Equal(1000, module.GetParameter("test").shape[0]);
            Assert.Equal(100, module.GetParameter("test").shape[1]);

            using (var grad = new AutoGradMode(false)) {
                param.transpose_(0, 1);
            }
            Assert.Equal(100, module.GetParameter("test").shape[0]);
            Assert.Equal(1000, module.GetParameter("test").shape[1]);
            Assert.Equal(100, param.shape[0]);
            Assert.Equal(1000, param.shape[1]);
        }

        private class TestModule : CustomModule
        {
            public TestModule(string name, TorchTensor tensor, bool withGrad)
                : base(name, new Parameter(name, tensor, withGrad))
            {
            }

            public override TorchTensor forward(TorchTensor input)
            {
                throw new NotImplementedException();
            }
        }


        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using gradient descent.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html"/>.
        /// </summary>
        [Fact]
        public void TestTraining()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            float learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var loss = mse_loss(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                seq.ZeroGrad();

                output.backward();

                using (var noGrad = new AutoGradMode(false)) {
                    foreach (var param in seq.parameters()) {
                        var grad = param.grad();
                        var update = grad.mul(learning_rate.ToScalar());
                        param.sub_(update);
                    }
                }
            }
        }

        [Fact]
        public void TestAdam()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00001;

            var optimizer = NN.Optimizer.Adam(seq.parameters(), learning_rate);

            Assert.NotNull(optimizer);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using Adam optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TestTrainingAdamDefaults()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            float prevLoss = float.MaxValue;
            var optimizer = NN.Optimizer.Adam(seq.parameters());
            var loss = mse_loss(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
        }

        [Fact]
        public void TestTrainingAdamAmsGrad()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            float prevLoss = float.MaxValue;
            var optimizer = NN.Optimizer.Adam(seq.parameters(), amsgrad: true);
            var loss = mse_loss(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using Adagrad optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TestTrainingAdagrad()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var optimizer = NN.Optimizer.Adagrad(seq.parameters(), learning_rate);
            var loss = mse_loss(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using RMSprop optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TestTrainingRMSLR()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var optimizer = NN.Optimizer.RMSProp(seq.parameters(), learning_rate);
            var loss = mse_loss(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
        }

        [Fact]
        public void TestTrainingRMSAlpha()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var optimizer = NN.Optimizer.RMSProp(seq.parameters(), learning_rate, alpha: 0.75);
            var loss = mse_loss(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
        }

        [Fact]
        public void TestTrainingRMSCentered()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var optimizer = NN.Optimizer.RMSProp(seq.parameters(), learning_rate, centered: true);
            var loss = mse_loss(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using SGD optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TestTrainingSGDMomentum()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var optimizer = NN.Optimizer.SGD(seq.parameters(), learning_rate, momentum: 0.5);
            var loss = mse_loss(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
        }

        [Fact(Skip = "Fails with an exception in native code.")]
        public void TestTrainingSGDNesterov()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var optimizer = NN.Optimizer.SGD(seq.parameters(), learning_rate, nesterov: true);
            var loss = mse_loss(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
        }

        [Fact]
        public void TestTrainingSGDDefaults()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;
            var optimizer = NN.Optimizer.SGD(seq.parameters(), learning_rate);
            var loss = mse_loss(NN.Reduction.Sum);

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                Assert.True(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
        }

        [Fact(Skip = "MNIST data too big to keep in repo")]
        public void TestMNISTLoader()
        {
            using (var train = Data.Loader.MNIST("../../../../test/data/MNIST", 32)) {
                Assert.NotNull(train);

                var size = train.Size();
                int i = 0;

                foreach (var (data, target) in train) {
                    i++;

                    Assert.Equal(data.shape, new long[] { 32, 1, 28, 28 });
                    Assert.Equal(target.shape, new long[] { 32 });

                    data.Dispose();
                    target.Dispose();
                }

                Assert.Equal(size, i * 32);
            }
        }

        [Fact(Skip = "CIFAR10 data too big to keep in repo")]
        public void TestCIFAR10Loader()
        {
            using (var train = Data.Loader.CIFAR10("../../../../src/Examples/Data", 16)) {
                Assert.NotNull(train);

                var size = train.Size();
                int i = 0;

                foreach (var (data, target) in train) {
                    i++;

                    Assert.Equal(data.shape, new long[] { 16, 3, 32, 32 });
                    Assert.Equal(target.shape, new long[] { 16 });
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
            using (var train = Data.Loader.MNIST("../../../../test/data/MNIST", 32)) {
                var size = train.Size();
                var epochs = 10;

                int i = 0;

                for (int e = 0; e < epochs; e++) {
                    foreach (var (data, target) in train) {
                        i++;

                        Assert.Equal(data.shape, new long[] { 32, 1, 28, 28 });
                        Assert.Equal(target.shape, new long[] { 32 });

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
            TorchTensor ones = Float32Tensor.ones(new long[] { 2, 2, 2 });
            var obj = AvgPool2D(ones, new long[] { 2, 2 }, new long[] { 2, 2 });
            Assert.Equal(typeof(TorchTensor), obj.GetType());
        }

        [Fact]
        public void AvgPool2DTensor()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 4, 2, 2, 2 });
            var obj = ones.avg_pool2d(new long[] { 2, 2 });
            Assert.Equal(typeof(TorchTensor), obj.GetType());
            Assert.Equal(Float32Tensor.ones(new long[] { 4, 2, 1, 1 }), obj);
        }


        [Fact]
        public void AvgPool2DBackwardTensor()
        {
            var ones = Float32Tensor.ones(new long[] { 4, 2, 2, 2 });
            var kernelSize = new long[] { 2, 2 };
            var avg = Float32Tensor.ones(new long[] { 4, 2, 1, 1 });
            var res = avg.avg_pool2d_backward(ones, kernelSize) * 4.0;

            var ones0000 = ones[0, 0, 0, 0].ToSingle();
            var res0000 = res[0, 0, 0, 0].ToSingle();
            Assert.Equal(ones0000, res0000);
            // This gets back to the original uniform input
            Assert.Equal(res, ones);
        }


        [Fact]
        public void AvgPool3DBackwardTensor()
        {
            var ones = Float32Tensor.ones(new long[] { 4, 2, 2, 2, 2 });
            var kernelSize = new long[] { 2, 2, 2 };
            var avg = Float32Tensor.ones(new long[] { 4, 2, 1, 1, 1 });
            var res = avg.avg_pool3d_backward(ones, kernelSize) * 8.0;

            var ones0000 = ones[0, 0, 0, 0, 0].ToSingle();
            var res0000 = res[0, 0, 0, 0, 0].ToSingle();
            Assert.True(Math.Abs(ones0000 - res0000) < 0.00001);
            // This gets back to the original uniform input
            Assert.True(res.allclose(ones));
        }

        [Fact]
        public void AvgPool3DBackwardTensorExplicitDivisor()
        {
            var ones = Float32Tensor.ones(new long[] { 4, 2, 2, 2, 2 });
            var kernelSize = new long[] { 2, 2, 2 };
            var avg = Float32Tensor.ones(new long[] { 4, 2, 1, 1, 1 });
            var res = avg.avg_pool3d_backward(ones, kernelSize, divisorOverride: 6) * 6.0;

            var ones0000 = ones[0, 0, 0, 0, 0].ToSingle();
            var res0000 = res[0, 0, 0, 0, 0].ToSingle();
            Assert.True(Math.Abs(ones0000 - res0000) < 0.00001);
            // This gets back to the original uniform input
            Assert.True(res.allclose(ones));
        }

        [Fact]
        public void MaxPool2DObjectInitialized()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 2, 2, 2 });
            var obj = MaxPool2D(ones, new long[] { 2, 2 }, new long[] { 2, 2 });
            Assert.Equal(typeof(TorchTensor), obj.GetType());
        }

        [Fact]
        public void TestMaxPool1D_1()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 3, 4 });
            using (var pool = MaxPool1D(2)) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 3, 2 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
            }
        }

        [Fact]
        public void TestMaxPool1D_2()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 3, 4 });
            using (var pool = MaxPool1D(2, 1)) {
                var pooled = pool.forward(ones);

                Assert.Equal(new long[] { 16, 3, 3 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0].ToSingle());
            }
        }

        [Fact]
        public void TestMaxPool2D_1()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 4, 4 });
            using (var pool = MaxPool2D(new long[] { 2, 2 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 2, 2 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1].ToSingle());
            }
        }

        [Fact]
        public void TestMaxPool2D_2()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 4, 4 });
            using (var pool = MaxPool2D(new long[] { 2, 2 }, new long[] { 1, 1 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 3, 3 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1].ToSingle());
                Assert.Equal(1, pooled[0, 1, 2].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 1].ToSingle());
                Assert.Equal(1, pooled[0, 2, 2].ToSingle());
            }
        }

        [Fact]
        public void TestMaxPool3D_1()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 4, 4, 8 });
            using (var pool = MaxPool3D(new long[] { 2, 2, 2 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 2, 2, 4 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1, 0].ToSingle());
            }
        }

        [Fact]
        public void TestMaxPool3D_2()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 4, 4, 8 });
            using (var pool = MaxPool3D(new long[] { 2, 2, 2 }, new long[] { 1, 1, 1 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 3, 3, 7 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 0, 0, 2].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 1].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 2].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 1].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 2].ToSingle());
            }
        }

        [Fact]
        public void TestAvgPool1D_1()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 3, 4 });
            using (var pool = AvgPool1D(2)) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 3, 2 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
            }
        }

        [Fact]
        public void TestAvgPool1D_2()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 3, 4 });
            using (var pool = AvgPool1D(2, 1)) {
                var pooled = pool.forward(ones);

                Assert.Equal(new long[] { 16, 3, 3 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0].ToSingle());
            }
        }

        [Fact]
        public void TestAvgPool2D_1()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 4, 4 });
            using (var pool = AvgPool2D(new long[] { 2, 2 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 2, 2 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1].ToSingle());
            }
        }

        [Fact]
        public void TestAvgPool2D_2()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 4, 4 });
            using (var pool = AvgPool2D(new long[] { 2, 2 }, new long[] { 1, 1 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 3, 3 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1].ToSingle());
                Assert.Equal(1, pooled[0, 1, 2].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 1].ToSingle());
                Assert.Equal(1, pooled[0, 2, 2].ToSingle());
            }

            ones = Float32Tensor.ones(new long[] { 16, 4, 4, 4 });
            using (var pool = AvgPool2D(new long[] { 2, 2 }, new long[] { 1, 1 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 4, 3, 3 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 0, 0, 2].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 1].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 2].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 1].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 2].ToSingle());
            }
        }

        [Fact]
        public void TestAvgPool3D_1()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 4, 4, 8 });
            using (var pool = AvgPool3D(new long[] { 2, 2, 2 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 2, 2, 4 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1, 0].ToSingle());
            }
        }

        [Fact]
        public void TestAvgPool3D_2()
        {
            TorchTensor ones = Float32Tensor.ones(new long[] { 16, 4, 4, 8 });
            using (var pool = AvgPool3D(new long[] { 2, 2, 2 }, new long[] { 1, 1, 1 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 3, 3, 7 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 2, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 2, 0].ToSingle());
            }

            ones = Float32Tensor.ones(new long[] { 16, 3, 4, 4, 8 });
            using (var pool = AvgPool3D(new long[] { 2, 2, 2 }, new long[] { 1, 1, 1 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 3, 3, 3, 7 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 0, 2, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 2, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 2, 0].ToSingle());
            }
        }

        [Fact]
        public void TestBatchNorm1D()
        {
            var ones = Float32Tensor.ones(new long[] { 16, 3, 28 });
            using (var pool = BatchNorm1D(3)) {
                var pooled = pool.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);
            }
        }

        [Fact]
        public void TestBatchNorm2D()
        {
            var ones = Float32Tensor.ones(new long[] { 16, 3, 28, 28 });
            using (var pool = BatchNorm2D(3)) {
                var pooled = pool.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);
            }
        }

        [Fact]
        public void TestBatchNorm3D()
        {
            var ones = Float32Tensor.ones(new long[] { 16, 3, 12, 28, 28 });
            using (var pool = BatchNorm3D(3)) {
                var pooled = pool.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);
            }
        }

        [Fact]
        public void TestEmbeddingDefaults()
        {
            var ones = Int32Tensor.ones(new long[] { 16 });
            using (var emb = Embedding(1000, 12)) {
                var output = emb.forward(ones);
                Assert.Equal(new long[] { 16, 12 }, output.shape);
            }
        }

        [Fact]
        public void TestEmbeddingWithMaxNorm()
        {
            var ones = Int32Tensor.ones(new long[] { 16 });
            using (var emb = Embedding(1000, 128, max_norm: 1.5)) {
                var output = emb.forward(ones);
                Assert.Equal(new long[] { 16, 128 }, output.shape);
            }
        }

        [Fact]
        public void TestEmbeddingSetWeights()
        {
            var ones = Int32Tensor.ones(new long[] { 16 });
            using (var emb = Embedding(1000, 12)) {
                var weights = Float32Tensor.randn(new long[] { 1000, 12 });

                emb.Weight = weights;

                Assert.Equal(emb.Weight.shape.Length, weights.shape.Length);
                Assert.Equal(emb.Weight.shape[0], weights.shape[0]);
                Assert.Equal(emb.Weight.shape[1], weights.shape[1]);
            }
        }

        [Fact]
        public void TestEmbeddingFromPretrained()
        {
            var ones = Int32Tensor.ones(new long[] { 16 });
            var weights = Float32Tensor.randn(new long[] { 1000, 12 });

            using (var emb = NN.Embedding.from_pretrained(weights)) {
                Assert.Equal(emb.Weight.shape.Length, weights.shape.Length);
                Assert.Equal(emb.Weight.shape[0], weights.shape[0]);
                Assert.Equal(emb.Weight.shape[1], weights.shape[1]);
            }
        }

        [Fact]
        public void TestOneHotEncoding1()
        {
            var ones = Int64Tensor.from(new long[] { 1, 2, 0, 0, 3, 4, 2, 2 });
            var env = OneHot(ones,5);
            var values = env.Data<long>().ToArray();
            Assert.Equal(ones.shape[0], env.shape[0]);
            Assert.Equal(5, env.shape[1]);
        }

        [Fact]
        public void TestOneHotEncoding2()
        {
            var ones = Int64Tensor.from(new long[] { 1, 2, 0, 5, 3, 4, 2, 2 });
            var env = OneHot(ones);
            var values = env.Data<long>().ToArray();
            Assert.Equal(ones.shape[0], env.shape[0]);
            Assert.Equal(6, env.shape[1]);
        }

        [Fact]
        public void TestTransformer()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var transformer_model = Transformer(d_model: 64, nhead: 2, num_encoder_layers: 2, dim_feedforward: 128)) {
                var src = Float32Tensor.rand(new long[] { 10, 16, 64 });
                var tgt = Float32Tensor.rand(new long[] { 20, 16, 64 });
                var output = transformer_model.forward(src, tgt);
                Assert.Equal(tgt.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerWithMasks()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var transformer_model = Transformer(d_model: 64, nhead: 2, num_encoder_layers: 2, dim_feedforward: 128)) {
                var src = Float32Tensor.rand(new long[] { 10, 16, 64 });
                var tgt = Float32Tensor.rand(new long[] { 20, 16, 64 });
                var src_mask = Float32Tensor.rand(new long[] { 10, 10 });
                var tgt_mask = Float32Tensor.rand(new long[] { 20, 20 });
                var output = transformer_model.forward(src, tgt, src_mask: src_mask, tgt_mask: tgt_mask);
                Assert.Equal(tgt.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerEncoderLayer()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var transformer_model = TransformerEncoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128)) {
                var src = Float32Tensor.rand(new long[] { 10, 16, 64 });
                var output = transformer_model.forward(src);
                Assert.Equal(src.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerEncoderLayerWithMasks()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var transformer_model = TransformerEncoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128)) {
                var src = Float32Tensor.rand(new long[] { 10, 16, 64 });
                var src_mask = Float32Tensor.rand(new long[] { 10, 10 });
                var output = transformer_model.forward(src, src_mask: src_mask);
                Assert.Equal(src.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerDecoderLayer()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var transformer_model = TransformerDecoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128)) {
                var tgt = Float32Tensor.rand(new long[] { 20, 16, 64 });
                var memory = Float32Tensor.rand(new long[] { 10, 16, 64 });
                var output = transformer_model.forward(tgt, memory);
                Assert.Equal(tgt.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerDecoderLayerWithMasks()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var transformer_model = TransformerDecoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128)) {
                var tgt = Float32Tensor.rand(new long[] { 20, 16, 64 });
                var memory = Float32Tensor.rand(new long[] { 10, 16, 64 });
                var tgt_mask = Float32Tensor.rand(new long[] { 20, 20 });
                var output = transformer_model.forward(tgt, memory, tgt_mask: tgt_mask);
                Assert.Equal(tgt.shape, output.shape);
            }
        }
    }
}