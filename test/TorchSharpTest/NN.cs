// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;

using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using System.Drawing;

#nullable enable

namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestNN
    {
        #region Linear

        [Fact]
        public void CreateLinear()
        {
            var lin = Linear(1000, 100);
            Assert.NotNull(lin);
            Assert.True(!(lin.bias is null));

            var ps = lin.parameters();
            Assert.Equal(2, ps.Count());
        }

        [Fact]
        public void TestGetBiasInLinear()
        {
            var lin = Linear(1000, 100, false);
            var ps = lin.parameters();
            var nps = ps.Count();
            Assert.Equal(1, nps);
            Assert.True(lin.bias is null);

            var lin2 = Linear(1000, 100, true);
            Assert.True(!(lin2.bias is null));
        }

        [Fact]
        public void TestDeviceAndTypeLinear()
        {
            {
                var lin = Linear(1000, 100, true, dtype: torch.float64);
                var ps = lin.parameters().ToArray();
                var nps = ps.Count();

                Assert.Multiple(
                    () => Assert.Equal(2, nps),
                    () => Assert.False(lin.bias is null),
                    () => Assert.Equal(torch.float64, ps[0].dtype),
                    () => Assert.Equal(torch.float64, ps[1].dtype)
                );
            }
            if (torch.cuda.is_available()) {
                {
                    var lin = Linear(1000, 100, true, device: torch.CUDA, dtype: torch.float64);
                    var ps = lin.parameters().ToArray();
                    var nps = ps.Count();

                    Assert.Multiple(
                        () => Assert.Equal(2, nps),
                        () => Assert.False(lin.bias is null),
                        () => Assert.Equal(torch.float64, ps[0].dtype),
                        () => Assert.Equal(torch.float64, ps[1].dtype),
                        () => Assert.Equal(DeviceType.CUDA, ps[0].device_type),
                        () => Assert.Equal(DeviceType.CUDA, ps[1].device_type)
                    );
                }
                {
                    var lin = Linear(1000, 100, true, device: torch.CUDA);
                    var ps = lin.parameters().ToArray();
                    var nps = ps.Count();

                    Assert.Multiple(
                        () => Assert.Equal(2, nps),
                        () => Assert.False(lin.bias is null),
                        () => Assert.Equal(torch.float32, ps[0].dtype),
                        () => Assert.Equal(torch.float32, ps[1].dtype),
                        () => Assert.Equal(DeviceType.CUDA, ps[0].device_type),
                        () => Assert.Equal(DeviceType.CUDA, ps[1].device_type)
                    );
                }
            }
        }

        [Fact]
        public void TestSetGetBiasInLinear()
        {
            var lin = Linear(1000, 100, true);
            var bias = torch.ones(new long[] { 1000 });
            var bCount = bias.NumberOfElements;
            lin.bias = bias.AsParameter();
            Assert.True(!(lin.bias is null));

            Assert.Equal(lin.bias?.NumberOfElements, bCount);
        }

        [Fact]
        public void TestWeightAndBiasShapeInLinear()
        {
            var lin = Linear(1000, 100, true);

            Assert.Equal(2, lin.weight!.shape.Length);
            Assert.Equal(100, lin.weight!.shape[0]);
            Assert.Equal(1000, lin.weight!.shape[1]);
            Assert.True(1 == lin.bias?.shape.Length);
            Assert.Equal(100, lin.bias?.shape[0]);
        }

        [Fact]
        public void TestWeightAndBiasParametersInLinear()
        {
            var lin = Linear(1000, 100, true);
            var names = lin.named_parameters().Select(p => p.name);
            Assert.True(names.Contains("weight") == true);
            Assert.True(names.Contains("bias") == true);
        }

        [Fact]
        public void TestWeightParameterInLinear()
        {
            var lin = Linear(1000, 100, false);
            var names = lin.named_parameters().Select(p => p.name);
            Assert.True(names.Contains("weight") == true);
            Assert.False(names.Contains("bias") == true);
        }

        [Fact]
        public void TestWeightAndBiasShapeInLinear3()
        {
            var lin = Linear(1000, 100, true);
            var weight = lin.get_parameter("weight");
            var bias = lin.get_parameter("bias");
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
            var bias = lin.bias!;
            var weight = lin.weight!.t();
            var input = torch.randn(new long[] { 1, 1000 });
            var forward = lin.forward(input);
            var matmul = input.matmul(weight).add(bias);

            Assert.Multiple(
                () => Assert.Equal(forward.shape.Length, matmul.shape.Length),
                () => Assert.Equal(forward.shape[0], matmul.shape[0]),
                () => Assert.Equal(forward.shape[1], matmul.shape[1])
            );

            for (int i = 0; i < 100; i++) {
                Assert.InRange(forward.data<float>()[i], matmul.data<float>()[i] - 10e5f, matmul.data<float>()[i] + 10e5f);
            }
        }

        [Fact]
        public void FunctionalLinearWithBias()
        {
            var input = torch.randn(4, 1000);
            var weight = torch.randn(100, 1000);
            var bias = torch.randn(100);
            var forward = torch.nn.functional.linear(input, weight, bias);
            var matmul = input.matmul(weight.t()).add(bias);

            Assert.Multiple(
                () => Assert.Equal(forward.shape.Length, matmul.shape.Length),
                () => Assert.Equal(forward.shape[0], matmul.shape[0]),
                () => Assert.Equal(forward.shape[1], matmul.shape[1])
            );
            for (int i = 0; i < 100; i++) {
                Assert.InRange(forward.data<float>()[i], matmul.data<float>()[i] - 10e5f, matmul.data<float>()[i] + 10e5f);
            }
        }

        [Fact]
        public void FunctionalLinearNoBias()
        {
            var input = torch.randn(4, 1000);
            var weight = torch.randn(100, 1000);
            var forward = torch.nn.functional.linear(input, weight);
            var matmul = input.matmul(weight.t());

            Assert.Multiple(
                () => Assert.Equal(forward.shape.Length, matmul.shape.Length),
                () => Assert.Equal(forward.shape[0], matmul.shape[0]),
                () => Assert.Equal(forward.shape[1], matmul.shape[1])
            );
            for (int i = 0; i < 100; i++) {
                Assert.InRange(forward.data<float>()[i], matmul.data<float>()[i] - 10e5f, matmul.data<float>()[i] + 10e5f);
            }
        }

        [Fact]
        public void TestLinearNoBias()
        {
            var lin = Linear(1000, 100, false);
            Assert.False(!(lin.bias is null));

            var weight = lin.weight!.transpose(0, 1);
            var input = torch.randn(new long[] { 1, 1000 });
            var forward = lin.forward(input);
            var matmul = input.matmul(weight);

            Assert.Equal(forward.shape.Length, matmul.shape.Length);
            Assert.Equal(forward.shape[0], matmul.shape[0]);
            Assert.Equal(forward.shape[1], matmul.shape[1]);

            for (int i = 0; i < 100; i++) {
                Assert.Equal(forward.data<float>()[i], matmul.data<float>()[i]);
            }
        }

        [Fact]
        public void TestBilinearWithBias()
        {
            var lin = Bilinear(20, 30, 40);
            var input1 = torch.randn(new long[] { 128, 20 });
            var input2 = torch.randn(new long[] { 128, 30 });
            var forward = lin.forward(input1, input2);

            Assert.Equal(2, forward.shape.Length);
            Assert.Equal(128, forward.shape[0]);
            Assert.Equal(40, forward.shape[1]);
        }

        [Fact]
        public void FunctionalBilinearWithBias()
        {
            var input1 = torch.randn(new long[] { 128, 20 });
            var input2 = torch.randn(new long[] { 128, 30 });
            var weight = torch.randn(40, 20, 30);
            var bias = torch.randn(40);

            var forward = torch.nn.functional.bilinear(input1, input2, weight, bias);

            Assert.Equal(2, forward.shape.Length);
            Assert.Equal(128, forward.shape[0]);
            Assert.Equal(40, forward.shape[1]);
        }

        [Fact]
        public void FunctionalBilinearNoBias()
        {
            var input1 = torch.randn(new long[] { 128, 20 });
            var input2 = torch.randn(new long[] { 128, 30 });
            var weight = torch.randn(40, 20, 30);

            var forward = torch.nn.functional.bilinear(input1, input2, weight);

            Assert.Equal(2, forward.shape.Length);
            Assert.Equal(128, forward.shape[0]);
            Assert.Equal(40, forward.shape[1]);
        }

        [Fact]
        public void TestIdentity()
        {
            var lin = Identity();

            var input = torch.randn(new long[] { 1, 1000 });
            var output = lin.forward(input);

            for (int i = 0; i < 1000; i++) {
                Assert.Equal(input.data<float>()[i], output.data<float>()[i]);
            }
        }

        [Fact]
        public void TestLinearEditBias()
        {
            var lin = Linear(1000, 100, true);
            var bias = torch.randn(new long[] { 100 });
            lin.bias = bias.clone().AsParameter();

            for (int i = 0; i < 100; i++) {
                Assert.Equal(lin.bias.data<float>()[i], bias.data<float>()[i]);
            }
        }

        [Fact]
        public void TestLinearEditWeightsAndBias()
        {
            var lin = Linear(1000, 1000, true);
            var bias = torch.randn(new long[] { 100 });
            var weights = torch.randn(new long[] { 100, 1000 });

            lin.bias = bias.clone().AsParameter();
            lin.weight = weights.clone().AsParameter();

            var w1 = lin.weight;
            var b1 = lin.bias;

            Assert.Equal(w1.shape.Length, weights.shape.Length);
            Assert.Equal(w1.shape[0], weights.shape[0]);
            Assert.Equal(w1.shape[1], weights.shape[1]);

            for (int i = 0; i < 100; i++) {
                Assert.Equal(b1.data<float>()[i], bias.data<float>()[i]);
            }

            var np = lin.named_parameters().ToArray();
            var w2 = np[0].parameter;
            var b2 = np[1].parameter;

            Assert.Equal(weights.shape.Length, w2.shape.Length);
            Assert.Equal(weights.shape[0], w2.shape[0]);
            Assert.Equal(weights.shape[1], w2.shape[1]);

            for (int i = 0; i < 100; i++) {
                Assert.Equal(b2.data<float>()[i], bias.data<float>()[i]);
            }
        }

        [Fact]
        public void TestLinearEditWeightsAndBiasGetParameters()
        {
            var lin = Linear(1000, 1000, true);
            var bias = torch.randn(new long[] { 100 });
            var weights = torch.randn(new long[] { 1000, 1000 });
            lin.bias = bias.AsParameter();
            lin.weight = weights.AsParameter();

            var parameters = lin.parameters().ToArray();

            Assert.Equal(lin.weight.shape.Length, parameters[0].shape.Length);
            Assert.Equal(lin.weight.shape[0], parameters[0].shape[0]);
            Assert.Equal(lin.weight.shape[1], parameters[0].shape[1]);
        }
        #endregion

        #region Activations
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
            var rel = ReLU().to(torch.DirectML);
            var input = torch.randn(new long[] { 64, 8 }).to(torch.DirectML);
            var output = rel.forward(input);
            var values = output.to(torch.CPU).data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= 0.0));
        }

        [Fact]
        public void EvaluateRelu6()
        {
            var rel = ReLU6();
            var input = torch.randn(new long[] { 64, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= 0.0 && val <= 6.0));
        }

        [Fact]
        public void EvaluateLeakyRelu()
        {
            var rel = LeakyReLU(0.1);
            var input = torch.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);

            var singleton = torch.tensor(15.0f);
            Assert.Equal(15.0f, rel.forward(singleton).item<float>());
            singleton = torch.tensor(-15.0f);
            Assert.Equal(-1.50f, rel.forward(singleton).item<float>());
        }

        [Fact]
        public void EvaluateMish()
        {
            var rel = Mish();
            var input = torch.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
        }

        [Fact]
        public void EvaluateRRelu()
        {
            var rel = RReLU();
            var input = torch.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
        }

        [Fact]
        public void EvaluateCELU()
        {
            var rel = CELU();
            var input = torch.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -1.0));
        }

        [Fact]
        public void EvaluateELU()
        {
            var rel = ELU();
            var input = torch.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -1.0));
        }

        [FactIgnoreOnPlatform(
            "Attempted to read or write protected memory.",
            "Windows",
            Architecture.X64)]
        public void EvaluateGLU()
        {
            var rel = GLU();
            var input = randn(new long[] { 8, 8, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(new long[] { 8, 8, 4 }, output.shape);
        }

        [Fact]
        public void EvaluateSELU()
        {
            var rel = SELU();
            var input = torch.randn(new long[] { 64, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -1.76));
        }

        [Fact]
        public void EvaluateGELU()
        {
            var rel = GELU().to(torch.DirectML);
            var input = torch.randn(new long[] { 64, 8 }).to(torch.DirectML) * 25.0;
            var output = rel.forward(input);
            var values = output.to(torch.CPU).data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -0.2));
        }

        [Fact]
        public void EvaluateHardshrink()
        {
            var rel = Hardshrink();
            var input = torch.randn(new long[] { 8, 8, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(output.shape, new long[] { 8, 8, 8 });
        }

        [Fact]
        public void EvaluateHardsigmoid()
        {
            var rel = Hardsigmoid();
            var input = torch.randn(new long[] { 8, 8, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(output.shape, new long[] { 8, 8, 8 });
        }

        [Fact]
        public void EvaluateHardswish()
        {
            var rel = Hardswish();
            var input = torch.from_array(new float[] { -3.5f, 0.6f, 3.25f });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(new float[] { 0f, 0.36f, 3.25f }, values);
        }

        [Fact]
        public void EvaluateHardtanh()
        {
            var rel = Hardtanh();
            var input = torch.randn(new long[] { 8, 8, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(output.shape, new long[] { 8, 8, 8 });
        }

        [Fact]
        public void EvaluateSigmoid()
        {
            var rel = Sigmoid();
            var input = torch.randn(new long[] { 64, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= 0.0 && val <= 1.0));
        }

        [Fact]
        public void EvaluateSiLU()
        {
            var rel = SiLU();
            var input = torch.randn(new long[] { 64, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -1.0));
        }

        [Fact]
        public void EvaluateSoftmax2d()
        {
            var rel = Softmax2d();
            var input = torch.randn(new long[] { 64, 3, 8, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= 0.0 && val <= 1.0));
        }

        [Fact]
        public void EvaluateTanh()
        {
            var rel = Tanh();
            var input = torch.randn(new long[] { 64, 3, 8, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= -1.0 && val <= 1.0));
        }

        [Fact]
        public void EvaluateSoftmax()
        {
            var input = torch.randn(new long[] { 64, 8 }) * 25.0;
            {
                var rel = Softmax(1);
                var output = rel.forward(input);
                var values = output.data<float>().ToArray();
                Assert.Equal(input.shape, output.shape);
                Assert.All(values, val => Assert.True(val >= 0.0 && val <= 1.0));
            }
            {
                var output = torch.special.softmax(input, 1);
                var values = output.data<float>().ToArray();
                Assert.Equal(input.shape, output.shape);
                Assert.All(values, val => Assert.True(val >= 0.0 && val <= 1.0));
            }
            {
                var output = torch.special.softmax(input, 1, float64);
                Assert.Equal(ScalarType.Float64, output.dtype);
                var values = output.data<double>().ToArray();
                Assert.Equal(input.shape, output.shape);
                Assert.All(values, val => Assert.True(val >= 0.0 && val <= 1.0));
            }
        }

        [Fact]
        public void EvaluateSoftmin()
        {
            var rel = Softmax(1);
            var input = torch.randn(new long[] { 64, 8 }) * 25.0;
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
            Assert.All(values, val => Assert.True(val >= 0.0 && val <= 1.0));
        }

        [Fact]
        public void EvaluateSoftplus()
        {
            var rel = Softplus();
            var input = torch.randn(new long[] { 8, 8, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
        }

        [Fact]
        public void EvaluateSoftshrink()
        {
            var rel = Softshrink();
            var input = torch.randn(new long[] { 8, 8, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
        }

        [Fact]
        public void EvaluateSoftsign()
        {
            var rel = Softsign();
            var input = torch.randn(new long[] { 8, 8, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
        }

        [Fact]
        public void EvaluateTanhshrink()
        {
            var rel = Tanhshrink();
            var input = torch.randn(new long[] { 8, 8, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
        }

        [Fact]
        public void EvaluateThreshold()
        {
            var rel = Threshold(0.1, 0.0);
            var input = torch.randn(new long[] { 8, 8, 8 });
            var output = rel.forward(input);
            var values = output.data<float>().ToArray();
            Assert.Equal(input.shape, output.shape);
        }
        #endregion

        #region Sequence
        [Fact]
        public void EvalSequence()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", ReLU()));

            var seq1 = Sequential(seq, lin2);

            var x = torch.randn(new long[] { 64, 1000 }, requires_grad: true);
            var eval = seq.forward(x);
        }

        [Fact]
        public void EvalEmptySequence()
        {
            var seq = Sequential();

            var x = torch.randn(new long[] { 64, 1000 }, requires_grad: true);
            var eval = seq.forward(x);
            Assert.Equal(x, eval);
        }

        [Fact]
        public void CreateSequence()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", ReLU()));

            var s2 = seq.append("lin2", lin2);
            Assert.Same(seq, s2);

            var parameters = seq.parameters();
            var parametersCount = parameters.Count();
            Assert.Equal(4, parametersCount);

            var namedParams = seq.named_parameters();
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

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            var eval = seq.forward(x);
            var loss = MSELoss(Reduction.Sum);
            var output = loss.forward(eval, y);

            var result = output.ToSingle();

            Assert.Same(lin1, seq[0]);
            Assert.Same(lin2, seq[2]);
        }

        [Fact]
        public void SequentialSlice()
        {
            var seq = Sequential(
                ("lin1", Linear(10,10)),
                ("relu1", ReLU()),
                ("lin2", Linear(10, 10)),
                ("tanh1", Tanh()),
                ("lin2", Linear(10, 10)));

            var slice = seq[(0, 2)];
            Assert.Equal(2, slice.Count);
            Assert.Same(seq[0], slice[0]);

            slice = seq[(1, null)];
            Assert.Equal(4, slice.Count);
            Assert.Same(seq[1], slice[0]);

            slice = seq[(null, 3)];
            Assert.Equal(3, slice.Count);
            Assert.Same(seq[0], slice[0]);
        }

        [Fact]
        public void SequentialSliceNames()
        {
            var seq = Sequential(
                ("lin1", Linear(10, 10)),
                ("relu1", ReLU()),
                ("lin2", Linear(10, 10)),
                ("tanh1", Tanh()),
                ("lin2", Linear(10, 10)));

            var slice = seq[(1, 3)].named_modules().ToArray();
            Assert.Equal("relu1", slice[0].name);
            Assert.Equal("lin2", slice[1].name);
        }

#if !NETFRAMEWORK
        [Fact]
        public void SequentialSliceWithRange()
        {
            var seq = Sequential(
                ("lin1", Linear(10, 10)),
                ("relu1", ReLU()),
                ("lin2", Linear(10, 10)),
                ("tanh1", Tanh()),
                ("lin2", Linear(10, 10)));

            var slice = seq[0..2];
            Assert.Equal(2, slice.Count);
            Assert.Same(seq[0], slice[0]);

            slice = seq[1..];
            Assert.Equal(4, slice.Count);
            Assert.Same(seq[1], slice[0]);

            slice = seq[..3];
            Assert.Equal(3, slice.Count);
            Assert.Same(seq[0], slice[0]);

            slice = seq[..^1];
            Assert.Equal(4, slice.Count);
            Assert.Same(seq[0], slice[0]);
        }

        [Fact]
        public void SequentialSliceNamesRange()
        {
            var seq = Sequential(
                ("lin1", Linear(10, 10)),
                ("relu1", ReLU()),
                ("lin2", Linear(10, 10)),
                ("tanh1", Tanh()),
                ("lin2", Linear(10, 10)));

            var slice = seq[1..3].named_modules().ToArray();
            Assert.Equal("relu1", slice[0].name);
            Assert.Equal("lin2", slice[1].name);
        }
#endif
        [Fact]
        public void EvalSequence2()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                lin1,
                ReLU(),
                lin2);

            var x = torch.randn(new long[] { 64, 1000 }, requires_grad: true);
            var eval = seq.forward(x);
        }

        [Fact]
        public void CreateSequence2()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                lin1,
                ReLU());

            var s2 = seq.append(lin2);
            Assert.Same(seq, s2);

            var parameters = seq.parameters();
            var parametersCount = parameters.Count();
            Assert.Equal(4, parametersCount);

            var namedParams = seq.named_parameters().ToArray();
            var namedParamsCount = namedParams.Count();
            Assert.Equal(4, namedParamsCount);

            Assert.Equal("0.weight", namedParams[0].name);
            Assert.Equal("0.bias", namedParams[1].name);
            Assert.Equal("2.weight", namedParams[2].name);
            Assert.Equal("2.bias", namedParams[3].name);
        }

        [Fact]
        public void EvalLossSequence2()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                lin1,
                ReLU(),
                lin2);

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            var eval = seq.forward(x);
            var loss = MSELoss(Reduction.Sum);
            var output = loss.forward(eval, y);

            var result = output.ToSingle();
        }
        #endregion

        #region Loss Functions
        [Fact]
        public void TestPoissonNLLLoss()
        {
            using (Tensor input = torch.tensor(new float[] { 0.5f, 1.5f, 2.5f }))
            using (Tensor target = torch.tensor(new float[] { 1f, 2f, 3f })) {
                var componentWiseLoss = ((Tensor)input.exp()) - target * input;
                Assert.True(componentWiseLoss.Equals(torch.nn.PoissonNLLLoss(reduction: Reduction.None).forward(input, target)));
                Assert.True(componentWiseLoss.sum().Equals(torch.nn.PoissonNLLLoss(reduction: Reduction.Sum).forward(input, target)));
                Assert.True(componentWiseLoss.mean().Equals(torch.nn.PoissonNLLLoss(reduction: Reduction.Mean).forward(input, target)));
            }
        }
        [Fact]
        public void TestPoissonNLLLossF()
        {
            using (Tensor input = torch.tensor(new float[] { 0.5f, 1.5f, 2.5f }))
            using (Tensor target = torch.tensor(new float[] { 1f, 2f, 3f })) {
                var componentWiseLoss = ((Tensor)input.exp()) - target * input;
                Assert.True(componentWiseLoss.Equals(torch.nn.functional.poisson_nll_loss(input, target, reduction: Reduction.None)));
                Assert.True(componentWiseLoss.sum().Equals(torch.nn.functional.poisson_nll_loss(input, target, reduction: Reduction.Sum)));
                Assert.True(componentWiseLoss.mean().Equals(torch.nn.functional.poisson_nll_loss(input, target, reduction: Reduction.Mean)));
            }
        }

        [Fact]
        public void TestPoissonNLLLoss2()
        {
            using (Tensor input = torch.rand(new long[] { 5, 2 }))
            using (Tensor target = torch.rand(new long[] { 5, 2 })) {
                var outTensor = torch.nn.PoissonNLLLoss(true, true).forward(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestPoissonNLLLossF2()
        {
            using (Tensor input = torch.rand(new long[] { 5, 2 }))
            using (Tensor target = torch.rand(new long[] { 5, 2 })) {
                var outTensor = torch.nn.functional.poisson_nll_loss(input, target, true, true);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestCrossEntropyLoss()
        {
            using (Tensor input = torch.rand(new long[] { 5, 12 }))
            using (Tensor target = torch.randint(12, new long[] { 5 }, torch.int64)) {
                var outTensor = CrossEntropyLoss().forward(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestCrossEntropyLossF()
        {
            using (Tensor input = torch.rand(new long[] { 5, 12 }))
            using (Tensor target = torch.randint(12, new long[] { 5 }, torch.int64)) {
                var outTensor = cross_entropy(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestL1Loss()
        {
            using (Tensor input = torch.rand(new long[] { 5, 2 }))
            using (Tensor target = torch.rand(new long[] { 5, 2 })) {
                var outTensor = L1Loss().forward(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestL1LossF()
        {
            using (Tensor input = torch.rand(new long[] { 5, 2 }))
            using (Tensor target = torch.rand(new long[] { 5, 2 })) {
                var outTensor = l1_loss(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestBinaryCrossEntropyLoss()
        {
            var m = Sigmoid();
            using (Tensor input = torch.randn(new long[] { 3 }))
            using (Tensor target = torch.randn(new long[] { 3 })) {
                var outTensor = BCELoss().forward(m.forward(input), target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestBinaryCrossEntropyLossF()
        {
            var m = Sigmoid();
            using (Tensor input = torch.randn(new long[] { 3 }))
            using (Tensor target = torch.randn(new long[] { 3 })) {
                var outTensor = binary_cross_entropy(m.forward(input), target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestBinaryCrossEntropyLossWithLogits()
        {
            using (Tensor input = torch.randn(new long[] { 3 }))
            using (Tensor target = torch.randn(new long[] { 3 })) {
                var outTensor = BCEWithLogitsLoss().forward(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestBinaryCrossEntropyLossWithLogitsF()
        {
            using (Tensor input = torch.randn(new long[] { 3 }))
            using (Tensor target = torch.randn(new long[] { 3 })) {
                var outTensor = binary_cross_entropy_with_logits(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestKLDivLoss()
        {
            using (Tensor input = torch.randn(new long[] { 3 }))
            using (Tensor target = torch.randn(new long[] { 3 })) {
                var outTensor = KLDivLoss().forward(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestKLDivLossF()
        {
            using (Tensor input = torch.randn(new long[] { 3 }))
            using (Tensor target = torch.randn(new long[] { 3 })) {
                var outTensor = kl_div(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestSmoothL1Loss()
        {
            using (Tensor input = torch.randn(new long[] { 3 }))
            using (Tensor target = torch.randn(new long[] { 3 })) {
                var outTensor = SmoothL1Loss().forward(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestSmoothL1LossF()
        {
            using (Tensor input = torch.randn(new long[] { 3 }))
            using (Tensor target = torch.randn(new long[] { 3 })) {
                var outTensor = smooth_l1_loss(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestSoftMarginLoss()
        {
            using (Tensor input = torch.randn(new long[] { 3 }))
            using (Tensor target = torch.randn(new long[] { 3 })) {
                var outTensor = SoftMarginLoss().forward(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestSoftMarginLossF()
        {
            using (Tensor input = torch.randn(new long[] { 3 }))
            using (Tensor target = torch.randn(new long[] { 3 })) {
                var outTensor = soft_margin_loss(input, target);
                var values = outTensor.data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestGaussianNLLLoss32()
        {
            {
                Tensor variance = torch.rand(new long[] { 15, 1 }, requires_grad: true);
                Tensor input = torch.randn(new long[] { 15, 5, 5 }, requires_grad: true);
                Tensor target = torch.randn(new long[] { 15, 5, 5 });

                if (torch.cuda.is_available()) {
                    input = input.cuda();
                    target = target.cuda();
                    variance = variance.cuda();
                }

                var outTensor = GaussianNLLLoss().forward(input, target, variance);
                outTensor.backward();

                var values = outTensor.cpu().data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
            {
                Tensor variance = torch.rand(new long[] { 15, 1 }, requires_grad: true);
                Tensor input = torch.randn(new long[] { 15, 5, 5 }, requires_grad: true);
                Tensor target = torch.randn(new long[] { 15, 5, 5 });

                if (torch.cuda.is_available()) {
                    input = input.cuda();
                    target = target.cuda();
                    variance = variance.cuda();
                }
                var outTensor = GaussianNLLLoss().forward(input, target, variance);
                outTensor.backward();

                var values = outTensor.cpu().data<float>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(float.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestGaussianNLLLoss64()
        {
            using (Tensor variance = torch.rand(new long[] { 15, 1 }, torch.float64, requires_grad: true))
            using (Tensor input = torch.randn(new long[] { 15, 5, 5 }, torch.float64, requires_grad: true))
            using (Tensor target = torch.randn(new long[] { 15, 5, 5 }, torch.float64)) {

                var outTensor = GaussianNLLLoss().forward(input, target, variance);
                outTensor.backward();

                var values = outTensor.data<double>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(double.IsNaN(values[0]))
                );
            }
            using (Tensor variance = torch.rand(new long[] { 15, 5, 5 }, torch.float64, requires_grad: true))
            using (Tensor input = torch.randn(new long[] { 15, 5, 5 }, torch.float64, requires_grad: true))
            using (Tensor target = torch.randn(new long[] { 15, 5, 5 }, torch.float64)) {

                var outTensor = GaussianNLLLoss().forward(input, target, variance);
                outTensor.backward();

                var values = outTensor.data<double>().ToArray();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.Single(values),
                () => Assert.False(double.IsNaN(values[0]))
                );
            }
        }

        [Fact]
        public void TestCTCLossWithError()
        {
            var device = cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;

            int T = 50, C = 20, N = 16, S = 30, S_min = 10;

            using var input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_().to(device);
            using var target = torch.randint(low: 1, high: C, size: (N, S), dtype: torch.@long).to(device);
            using var input_lengths = torch.full(size: N, value: T, dtype: torch.@long).to(device);
            using var target_lengths = torch.randint(low: S_min, high: S, size: N, dtype: torch.@long).to(device);

            using var ctc_loss = nn.CTCLoss().to(device);
            using var loss = ctc_loss.forward(input, target, input_lengths, target_lengths);
            loss.backward();

            var outTensor = loss.cpu();

            var values = outTensor.data<float>().ToArray();
            Assert.Multiple(
            () => Assert.Empty(outTensor.shape),
            () => Assert.Single(values),
            () => Assert.False(float.IsNaN(values[0]))
            );
        }

        [Fact]
        public void TestGaussianNLLLossWithError()
        {
            using (Tensor variance = torch.rand(new long[] { 15, 1 }, requires_grad: true).neg())
            using (Tensor input = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.randn(new long[] { 15, 5 })) {

                Assert.Throws<ArgumentException>(() => {
                    var outTensor = GaussianNLLLoss().forward(input, target, variance);
                    outTensor.backward();
                });

            }
        }

        [Fact]
        public void TestCosineEmbeddingLoss()
        {
            using (Tensor input1 = torch.rand(new long[] { 15, 5 }, requires_grad: true).neg())
            using (Tensor input2 = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.randn(new long[] { 15 }).sign()) {

                var outTensor = CosineEmbeddingLoss().forward(input1, input2, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestCosineEmbeddingLossF()
        {
            using (Tensor input1 = torch.rand(new long[] { 15, 5 }, requires_grad: true).neg())
            using (Tensor input2 = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.randn(new long[] { 15 }).sign()) {

                var outTensor = cosine_embedding_loss(input1, input2, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestHingeEmbeddingLoss()
        {
            using (Tensor input = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.randn(new long[] { 15, 5 }).sign()) {
                var outTensor = HingeEmbeddingLoss().forward(input, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestHingeEmbeddingLossF()
        {
            using (Tensor input = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.randn(new long[] { 15, 5 }).sign()) {
                var outTensor = hinge_embedding_loss(input, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestHuberLoss()
        {
            using (Tensor input = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.randn(new long[] { 15, 5 }).sign()) {
                var outTensor = HuberLoss().forward(input, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
                outTensor = HuberLoss(1.5).forward(input, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestHuberLossF()
        {
            using (Tensor input = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.randn(new long[] { 15, 5 }).sign()) {
                var outTensor = huber_loss(input, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
                outTensor = huber_loss(input, target, 1.5);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestMarginRankingLoss()
        {
            using (Tensor input1 = torch.randn(new long[] { 15 }, requires_grad: true))
            using (Tensor input2 = torch.randn(new long[] { 15 }, requires_grad: true))
            using (Tensor target = torch.randn(new long[] { 15 }).sign()) {
                var outTensor = MarginRankingLoss().forward(input1, input2, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestMarginRankingLossF()
        {
            using (Tensor input1 = torch.randn(new long[] { 15 }, requires_grad: true))
            using (Tensor input2 = torch.randn(new long[] { 15 }, requires_grad: true))
            using (Tensor target = torch.randn(new long[] { 15 }).sign()) {
                var outTensor = margin_ranking_loss(input1, input2, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestMultilabelMarginLoss()
        {
            using (Tensor input = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.ones(new long[] { 15, 5 }, torch.int64)) {
                var outTensor = MultiLabelMarginLoss().forward(input, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestMultilabelMarginLossF()
        {
            using (Tensor input = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.ones(new long[] { 15, 5 }, torch.int64)) {
                var outTensor = multi_label_margin_loss(input, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestMultilabelSoftMarginLoss()
        {
            using (Tensor input = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.ones(new long[] { 15, 5 })) {
                var outTensor = MultiLabelSoftMarginLoss().forward(input, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestMultilabelSoftMarginLossF()
        {
            using (Tensor input = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.ones(new long[] { 15, 5 })) {
                var outTensor = multilabel_soft_margin_loss(input, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestMultiMarginLoss()
        {
            using (Tensor input = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.ones(new long[] { 15 }, torch.int64)) {
                var outTensor = MultiMarginLoss().forward(input, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestMultiMarginLossF()
        {
            using (Tensor input = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor target = torch.ones(new long[] { 15 }, torch.int64)) {
                var outTensor = multi_margin_loss(input, target);
                outTensor.backward();
                Assert.Multiple(
                () => Assert.Empty(outTensor.shape),
                () => Assert.False(float.IsNaN(outTensor.item<float>()))
                );
            }
        }

        [Fact]
        public void TestTripleMarginLoss()
        {
            using (Tensor anchor = torch.rand(new long[] { 15, 5 }, requires_grad: true).neg())
            using (Tensor positive = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor negative = torch.randn(new long[] { 15, 5 })) {

                var output = TripletMarginLoss();
                var result = output.forward(anchor, positive, negative);
                Assert.Multiple(
                () => Assert.Empty(result.shape),
                () => Assert.False(float.IsNaN(result.item<float>()))
                );
            }
        }

        [Fact]
        public void TestTripleMarginLossF()
        {
            using (Tensor anchor = torch.rand(new long[] { 15, 5 }, requires_grad: true).neg())
            using (Tensor positive = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor negative = torch.randn(new long[] { 15, 5 })) {

                var result = triplet_margin_loss(anchor, positive, negative);
                Assert.Multiple(
                () => Assert.Empty(result.shape),
                () => Assert.False(float.IsNaN(result.item<float>()))
                );
            }
        }

        [Fact]
        public void TestTripleMarginWithDistanceLoss()
        {
            Func<Tensor, Tensor, Tensor> distance =
                (x, y) => {
                    return (x - y).abs();
                };

            using (Tensor anchor = torch.rand(new long[] { 15, 5 }, requires_grad: true).neg())
            using (Tensor positive = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor negative = torch.randn(new long[] { 15, 5 })) {

                var output = TripletMarginWithDistanceLoss(distance);
                var result = output.forward(anchor, positive, negative);
                Assert.Multiple(
                () => Assert.Empty(result.shape),
                () => Assert.False(float.IsNaN(result.item<float>()))
                );
            }
        }

        [Fact]
        public void TestTripleMarginWithDistanceLossF()
        {
            Func<Tensor, Tensor, Tensor> distance =
                (x, y) => {
                    return (x - y).abs();
                };

            using (Tensor anchor = torch.rand(new long[] { 15, 5 }, requires_grad: true).neg())
            using (Tensor positive = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor negative = torch.randn(new long[] { 15, 5 })) {

                var result = triplet_margin_with_distance_loss(anchor, positive, negative, distance);
                Assert.Multiple(
                () => Assert.Empty(result.shape),
                () => Assert.False(float.IsNaN(result.item<float>()))
                );
            }
        }

        [Fact]
        public void TestTripleMarginWithDistanceLossNoDistance()
        {
            using (Tensor anchor = torch.rand(new long[] { 15, 5 }, requires_grad: true).neg())
            using (Tensor positive = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor negative = torch.randn(new long[] { 15, 5 })) {

                var output = TripletMarginWithDistanceLoss();
                var result = output.forward(anchor, positive, negative);
                Assert.Multiple(
                () => Assert.Empty(result.shape),
                () => Assert.False(float.IsNaN(result.item<float>()))
                );
            }
        }

        [Fact]
        public void TestTripleMarginWithDistanceLossNoDistanceF()
        {
            using (Tensor anchor = torch.rand(new long[] { 15, 5 }, requires_grad: true).neg())
            using (Tensor positive = torch.randn(new long[] { 15, 5 }, requires_grad: true))
            using (Tensor negative = torch.randn(new long[] { 15, 5 })) {

                var result = triplet_margin_with_distance_loss(anchor, positive, negative);
                Assert.Multiple(
                () => Assert.Empty(result.shape),
                () => Assert.False(float.IsNaN(result.item<float>()))
                );
            }
        }

        #endregion

        #region Gradients
        [Fact]
        public void TestBackward()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(
                ("lin1", lin1),
                ("relu1", ReLU()),
                ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 }, requires_grad: true);
            var y = torch.randn(new long[] { 64, 10 }, requires_grad: true);

            var eval = seq.forward(x);
            var loss = MSELoss(Reduction.Sum);
            var output = loss.forward(eval, y);

            seq.zero_grad();

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

            var x = torch.randn(new long[] { 64, 1000 }, requires_grad: true);
            var y = torch.randn(new long[] { 64, 10 }, requires_grad: true);

            var eval = seq.forward(x);
            var loss = MSELoss(Reduction.Sum);
            var output = loss.forward(eval, y);

            seq.zero_grad();

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

            var x = torch.randn(new long[] { 64, 1000 }, requires_grad: true);
            var y = torch.randn(new long[] { 64, 10 }, requires_grad: true);

            var eval = seq.forward(x);
            var loss = MSELoss(Reduction.Sum);
            var output = loss.forward(eval, y);

            seq.zero_grad();

            output.backward();

            foreach (var parm in seq.parameters()) {
                var grad = parm.grad();
            }
        }

        [Fact]
        public void TestGrad2()
        {
            var y = torch.randn(new long[] { 32, 1 });
            var input = new double[] { -2.75, 0.77, -0.61, 0.14, 1.39, 0.38, -0.53, -0.5, -2.13, -0.39, 0.46, -0.61, -0.37, -0.12, 0.55, -1, 0.84, -0.02, 1.3, -0.24, -0.5, -2.12, -0.85, -0.91, 1.81, 0.02, -0.78, -1.41, -1.09, -0.65, 0.9, -0.37, -0.22, 0.28, 1.05, -0.24, 0.3, -0.99, 0.19, 0.32, -0.95, -1.19, -0.63, 0.75, 0.16, 0.15, 0.3, -0.69, 0.2, -0.4, -0.67, 0.18, -1.43, -0.61, -0.78, -0.11, -1.07, -1.71, -0.45, -0.6, 0.05, -1.59, 1.24, 0.62, 0.01, 1.35, -0.9, -1.25, 1.62, -1.45, 0.92, 1.51, -0.19, -1.33, -0.01, -0.13, 0.1, -1.34, 1.23, 0.57, -0.24, 0.5, 0.71, -0.15, -1.37, -1.03, 1.8, 1.4, -0.63, 0.8, -0.97, -0.64, 0.51, 0.52, 0.95, 0.86, 0.43, 0.73, -1.38, -0.56, 0.44, 1.2, -1.45, -0.07, 1.88, 1.57, 0.38, -2.2, -0.56, -1.52, -0.17, 1.38, -1.02, -1.61, -0.13, -0.44, -0.37, 0.23, 1.75, 0.83, -0.02, -1.91, -0.23, -0.47, -1.41, -1.01, -0.91, -0.56, -1.72, 1.47, 0.31, 0.24, 0.48, 2.06, 0.07, -0.96, 1.03, -0.4, -0.64, -0.85, 0.42, -0.33, 0.85, -0.11, -1.24, -0.71, -1.04, -0.37, -0.37, 0.84, -0.9, -1.63, -2.91, -0.71, 0.09, 1.64, -1.1, -1.05, 0.51, 0.57, 0.19, 0.36, 1.36, 1.45, 0.35, -1.66, -0.65, 0.47, 1.95, -0.32, 0.19, -2.06, 0.5, 1.03, 0.94, -0.65, -2.94, 0.41, 1.13, 0.95, -0.02, 1.12, 0.19, 0.66, -0.77, -0.39, 0.59, -1.58, -0.67, 0.88, 0.26, -0.63, 0.49, 1.38, 1.48, -0.55, 0.4, 0.65, 0.19, 0.25, 0.03, -0.31, 0.75, 2.16, -1.36, 0.05, 0.22, 0.65, 1.28, 0.42, 1.35, -0.08, 1.1, 0.25, 0.44, 1.06, -1.78, 0.47, 1.38, 0.43, -1.56, 0.14, -0.22, 1.48, 0.04, 0.33, 0.1, 0.2, -0.99, 1.04, 0.61, -0.4, 0.96, 0.4, 0.5, 0.1, 0.02, 0.01, 0.22, 1.45, -0.77, 0.69, 0.95, 0.96, -0.09, -0.26, 0.22, -1.61, 1.86, -0.06, -0.34, -0.35, 0.55, -1.08, 1.29, 0.92, 0.16, 0.55, -0.01, 0.2, -0.61, -0.28, -2.17, -0.46, 1.63, 1.61, 0.64, 0.32, -0.75, 0.33, 0.3, -1.15, 0.42, -0.06, -1.14, 1.62, -0.9, -0.39, 0.4, 1.52, -0.43, 1.22, -0.32, -0.02, 1, -0.92, 0.11, 0.8, -0.99, -0.26, -2.85, -1.13, 0.49, -0.63, -0.54, -0.86, -0.97, -0.9, 0.23, 1.26, -1.78, -0.84, -0.48, 0.35, -1.13, -2.23, 0.1, 0.95, 1.27, 0.08, -2.21, 0.67, -0.2, 0.6, -1.14, 0.65, -0.73, -0.01, 0.9, -1.33, -1.16, 0.29, 1.16, 1.19, 0.84, 0.66, -1.55, -0.58, 1.85, -1.16, -0.95, 0.98, -0.1, -1.47, 0.78, -0.75, -1.32, 0.61, -0.5, -1, -0.42, 0.96, -1.39, 0.08, -1.82, 0.51, -0.71, -0.02, 2.32, -0.71, 0.08, -1.07 }.ToTensor(new long[] { 32, 11 }).to_type(ScalarType.Float32);
            var inputs = new Tensor[] { input };
            var scaler = new double[] { 0.2544529, 0.3184713, 0.2597403, 0.3246753, 0.3144654, 0.3322259, 0.3436426, 0.3215434, 0.308642, 0.3154574, 0.3448276 }.ToTensor(new long[] { 1, 11 }).to_type(ScalarType.Float32).with_requires_grad();
            var linear = Linear(11, 1, true);
            linear.bias = new double[] { 373.8864 }.ToTensor(new long[] { 1, 1 }).to_type(ScalarType.Float32).with_requires_grad().AsParameter();
            linear.weight = new double[] { 300.2818, -0.5905267, 286.2787, 0.1970505, 0.9004903, 0.1373157, 55.85495, 11.43741, 1.525748, 0.4299785, 239.9356 }.ToTensor(new long[] { 1, 11 }).to_type(ScalarType.Float32).with_requires_grad().AsParameter();

            var afterCat = torch.cat(inputs, 1);
            var afterScaler = afterCat * scaler;
            var prediction = linear.forward(afterScaler);

            var loss = MSELoss();
            var output = loss.forward(prediction, y);

            linear.zero_grad();

            output.backward();

            var scalerGrad = scaler.grad();
            var weightGrad = linear.weight.grad();
            var biasGrad = linear.bias.grad();
            Assert.True(scalerGrad is not null && scalerGrad.shape.Length == 2);
            Assert.True(weightGrad is not null && weightGrad.shape.Length == 2);
            Assert.True(biasGrad is not null && biasGrad.shape.Length == 2);
        }

        [Fact]
        public void TestSetGrad()
        {
            var x = torch.rand(new long[] { 10, 10 });
            Assert.False(x.requires_grad);

            x.requires_grad = true;
            Assert.True(x.requires_grad);
            x.requires_grad = false;
            Assert.False(x.requires_grad);
        }

        private class CondModel : Module<Tensor, Tensor>
        {
            private Module<Tensor, Tensor> fb = Linear(1000, 100, false);
            private Module<Tensor, Tensor> fbT1 = Linear(100, 10, false);
            private Module<Tensor, Tensor> fbF1 = Linear(100, 50, false);
            private Module<Tensor, Tensor> fbF2 = Linear(50, 10, false);
            private bool _isTrue = false;

            public CondModel(string name, bool isTrue) : base(name)
            {
                _isTrue = isTrue;
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
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
            Assert.Equal(4, psT.Count());

            var psF = modF.parameters();
            Assert.Equal(4, psF.Count());

            var x = torch.randn(new long[] { 64, 1000 }, requires_grad: true);
            var y = torch.randn(new long[] { 64, 10 }, requires_grad: true);

            modT.train();

            Assert.True(modT.training);

            var eval = modT.forward(x);
            var loss = MSELoss(Reduction.Sum);
            var output = loss.forward(eval, y);

            modT.zero_grad();

            output.backward();
            var gradCounts = 0;

            foreach (var (name, parm) in modT.named_parameters()) {
                var grad = parm.grad();
                gradCounts += grad is not null ? (grad.Handle == IntPtr.Zero ? 0 : 1) : 0;
            }

            Assert.Equal(2, gradCounts);

            //{ "grad can be implicitly created only for scalar outputs (_make_grads at ..\\..\\torch\\csrc\\autograd\\autograd.cpp:47)\n(no backtrace available)"}
            modF.train();

            eval = modF.forward(x);
            output = loss.forward(eval, y);

            modF.zero_grad();

            output.backward();
            gradCounts = 0;

            foreach (var parm in modF.parameters()) {
                var grad = parm.grad();
                gradCounts += grad is not null ? (grad.Handle == IntPtr.Zero ? 0 : 1) : 0;
            }

            Assert.Equal(3, gradCounts);
        }
        #endregion

        #region Convolution
        [Fact]
        public void TestConv1d()
        {
            var shape = new long[] { 16, 3, 28 };
            Tensor t = torch.rand(shape);
            var conv = Conv1d(3, 64, 3);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(26, output.shape[2]);
        }

        [Fact]
        public void TestConv1dGetWeight()
        {
            var conv = Conv1d(3, 64, 3);
            var weight = conv.weight;
            var bias = conv.bias;
            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3 }, weight.shape);
        }

        [Fact]
        public void TestConv1dEditWeightAndBias()
        {
            var conv = Conv1d(3, 64, 3);

            conv.bias = torch.randn(new long[] { 64 }).AsParameter();
            var weights = torch.randn(new long[] { 64, 3, 3 });

            var weight = conv.weight;
            var bias = conv.bias;

            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3 }, weight.shape);

            for (int i = 0; i < 64; i++) {
                Assert.Equal(conv.bias.data<float>()[i], bias.data<float>()[i]);
            }
        }

        [Fact]
        public void TestConv1dStride()
        {
            var shape = new long[] { 16, 3, 28 };
            Tensor t = torch.rand(shape);
            var conv = Conv1d(3, 64, 3, stride: 2);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(13, output.shape[2]);
        }

        [Fact]
        public void TestConv1dPadding()
        {
            var shape = new long[] { 16, 3, 28 };
            Tensor t = torch.rand(shape);

            using (var conv = Conv1d(3, 64, 3, padding: 1))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
            }
            using (var conv = Conv1d(3, 64, 3, padding: 1, paddingMode: PaddingModes.Reflect))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
            }
            using (var conv = Conv1d(3, 64, 3, padding: Padding.Same))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
            }
        }

        [Fact]
        public void TestConv2d()
        {
            var shape = new long[] { 16, 3, 28, 28 };
            Tensor t = torch.rand(shape);
            {
                var conv = Conv2d(3, 64, 3);
                var output = conv.forward(t);
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(26, output.shape[2]);
                Assert.Equal(26, output.shape[3]);
            }
            {
                var conv = Conv2d(3, 64, (3, 3));
                var output = conv.forward(t);
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(26, output.shape[2]);
                Assert.Equal(26, output.shape[3]);
            }
        }

        [Fact]
        public void TestConv2dGetWeight()
        {
            var conv = Conv2d(3, 64, (3, 3));
            var weight = conv.weight;
            var bias = conv.bias;
            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3, 3 }, weight.shape);
        }

        [Fact]
        public void TestConv2dEditWeightAndBias()
        {
            var conv = Conv2d(3, 64, 3);

            conv.bias = torch.randn(new long[] { 64 }).AsParameter();
            var weights = torch.randn(new long[] { 64, 3, 3, 3 });

            var weight = conv.weight;
            var bias = conv.bias;

            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3, 3 }, weight.shape);

            for (int i = 0; i < 64; i++) {
                Assert.Equal(conv.bias.data<float>()[i], bias.data<float>()[i]);
            }
        }

        [Fact]
        public void TestConv2dStride()
        {
            var shape = new long[] { 16, 3, 28, 28 };
            Tensor t = torch.rand(shape);

            {
                var conv = Conv2d(3, 64, 3, stride: 2);
                var output = conv.forward(t);
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(13, output.shape[2]);
                Assert.Equal(13, output.shape[3]);
            }
            {
                var conv = Conv2d(3, 64, (3, 3), stride: (2, 2));
                var output = conv.forward(t);
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(13, output.shape[2]);
                Assert.Equal(13, output.shape[3]);
            }
        }

        [Fact]
        public void TestConv2dPadding()
        {
            var shape = new long[] { 16, 3, 28, 28 };
            Tensor t = torch.rand(shape);
            using (var conv = Conv2d(3, 64, 3, padding: 1))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
                Assert.Equal(28, output.shape[3]);
            }
            using (var conv = Conv2d(3, 64, (3, 3), padding: (1, 1), paddingMode: PaddingModes.Reflect))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
                Assert.Equal(28, output.shape[3]);
            }
            using (var conv = Conv2d(3, 64, 3, padding: Padding.Same))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
                Assert.Equal(28, output.shape[3]);
            }
            using (var conv = Conv2d(3, 64, (3, 3), padding: Padding.Same))
            using (var output = conv.forward(t)) {
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(28, output.shape[2]);
                Assert.Equal(28, output.shape[3]);
            }
        }

        [Fact]
        public void TestConv3d()
        {
            var shape = new long[] { 16, 3, 28, 28, 28 };
            Tensor t = torch.rand(shape);
            {
                var conv = Conv3d(3, 64, 3);
                var output = conv.forward(t);
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(26, output.shape[2]);
                Assert.Equal(26, output.shape[3]);
                Assert.Equal(26, output.shape[4]);
            }
            {
                var conv = Conv3d(3, 64, (3, 3, 3));
                var output = conv.forward(t);
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(26, output.shape[2]);
                Assert.Equal(26, output.shape[3]);
                Assert.Equal(26, output.shape[4]);
            }
        }

        [Fact]
        public void TestConv3dGetWeight()
        {
            var conv = Conv3d(3, 64, 3);
            var weight = conv.weight;
            var bias = conv.bias;
            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3, 3, 3 }, weight.shape);
        }

        [Fact]
        public void TestConv3dEditWeightAndBias()
        {
            var conv = Conv3d(3, 64, 3);

            conv.bias = torch.randn(new long[] { 64 }).AsParameter();
            var weights = torch.randn(new long[] { 64, 3, 3, 3 });

            var weight = conv.weight;
            var bias = conv.bias;

            Assert.NotNull(weight);
            Assert.NotNull(bias);
            Assert.Equal(new long[] { 64, 3, 3, 3, 3 }, weight.shape);

            for (int i = 0; i < 64; i++) {
                Assert.Equal(conv.bias.data<float>()[i], bias.data<float>()[i]);
            }
        }

        [Fact]
        public void TestConv3dStride()
        {
            var shape = new long[] { 16, 3, 28, 28, 28 };
            Tensor t = torch.rand(shape);
            {
                var conv = Conv3d(3, 64, 3, stride: 2);
                var output = conv.forward(t);
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(13, output.shape[2]);
                Assert.Equal(13, output.shape[3]);
                Assert.Equal(13, output.shape[4]);
            }
            {
                var conv = Conv3d(3, 64, (3, 3, 3), stride: (2, 2, 2));
                var output = conv.forward(t);
                Assert.Equal(16, output.shape[0]);
                Assert.Equal(64, output.shape[1]);
                Assert.Equal(13, output.shape[2]);
                Assert.Equal(13, output.shape[3]);
                Assert.Equal(13, output.shape[4]);
            }
        }

        [Fact]
        public void TestConv3dPadding()
        {
            // TODO: (Skip = "Not working on MacOS (note: may now be working, we need to recheck)")
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                var shape = new long[] { 16, 3, 28, 28, 28 };
                Tensor t = torch.rand(shape);
                using (var conv = Conv3d(3, 64, 3, padding: 1))
                using (var output = conv.forward(t)) {
                    Assert.Equal(16, output.shape[0]);
                    Assert.Equal(64, output.shape[1]);
                    Assert.Equal(28, output.shape[2]);
                    Assert.Equal(28, output.shape[3]);
                    Assert.Equal(28, output.shape[4]);
                }
                using (var conv = Conv3d(3, 64, (3, 3, 3), padding: (1, 1, 1), paddingMode: PaddingModes.Replicate))
                using (var output = conv.forward(t)) {
                    Assert.Equal(16, output.shape[0]);
                    Assert.Equal(64, output.shape[1]);
                    Assert.Equal(28, output.shape[2]);
                    Assert.Equal(28, output.shape[3]);
                    Assert.Equal(28, output.shape[4]);
                }
                using (var conv = Conv3d(3, 64, 3, padding: Padding.Same))
                using (var output = conv.forward(t)) {
                    Assert.Equal(16, output.shape[0]);
                    Assert.Equal(64, output.shape[1]);
                    Assert.Equal(28, output.shape[2]);
                    Assert.Equal(28, output.shape[3]);
                    Assert.Equal(28, output.shape[4]);
                }
                using (var conv = Conv3d(3, 64, (3, 3, 3), padding: Padding.Same))
                using (var output = conv.forward(t)) {
                    Assert.Equal(16, output.shape[0]);
                    Assert.Equal(64, output.shape[1]);
                    Assert.Equal(28, output.shape[2]);
                    Assert.Equal(28, output.shape[3]);
                    Assert.Equal(28, output.shape[4]);
                }
            }
        }

        [Fact]
        public void TestConvTranspose1d()
        {
            var shape = new long[] { 16, 3, 28 };
            Tensor t = torch.rand(shape);
            var conv = ConvTranspose1d(3, 64, 3);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(30, output.shape[2]);
        }

        [Fact]
        public void TestConvTranspose2d()
        {
            var shape = new long[] { 16, 3, 28, 28 };
            Tensor t = torch.rand(shape);
            var conv = ConvTranspose2d(3, 64, 3);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(30, output.shape[2]);
            Assert.Equal(30, output.shape[3]);
        }

        [Fact]
        public void TestConvTranspose3d()
        {
            var shape = new long[] { 16, 3, 28, 28, 28 };
            Tensor t = torch.rand(shape);
            var conv = ConvTranspose3d(3, 64, 3);
            var output = conv.forward(t);
            Assert.Equal(16, output.shape[0]);
            Assert.Equal(64, output.shape[1]);
            Assert.Equal(30, output.shape[2]);
            Assert.Equal(30, output.shape[3]);
            Assert.Equal(30, output.shape[4]);
        }
        #endregion

        #region Custom Modules
        [Fact]
        public void TestCustomModule1()
        {
            var module = new TestModule1(torch.randn(new long[] { 2, 2 }), true);
            var name = module.GetName();
            Assert.NotNull(name);
            Assert.Equal("TestModule1", name);

            Assert.True(module.has_parameter("test"));
            Assert.True(module.has_parameter("list.0"));
            Assert.True(module.has_parameter("dict.first"));
            Assert.True(module.has_parameter("dict.second"));

            var ps = module.parameters();
            var n = ps.Count();
            Assert.Equal(4, n);
        }

        [Fact]
        public void TestCustomModuleWithInPlaceModification()
        {
            var param = torch.randn(new long[] { 1000, 100 });
            var module = new TestModule1(param, true);

            Assert.Equal(1000, module.get_parameter("test").shape[0]);
            Assert.Equal(100, module.get_parameter("test").shape[1]);

            param = module.get_parameter("test");

            using (torch.no_grad()) {
                param.transpose_(0, 1);
            }
            Assert.Equal(100, module.get_parameter("test").shape[0]);
            Assert.Equal(1000, module.get_parameter("test").shape[1]);
            Assert.Equal(100, param.shape[0]);
            Assert.Equal(1000, param.shape[1]);
        }

        [Fact]
        public void TestCustomModule2()
        {
            var module = new TestModule2(torch.randn(new long[] { 2, 2 }), true);

            var ps = module.named_parameters().ToArray();
            Assert.Equal(16, ps.Length);

            Assert.True(module.has_parameter("submodule.test"));
            Assert.True(module.has_parameter("submodule.list.0"));
            Assert.True(module.has_parameter("submodule.dict.first"));
            Assert.True(module.has_parameter("submodule.dict.second"));
            Assert.True(module.has_parameter("list.0.test"));
            Assert.True(module.has_parameter("list.0.list.0"));
            Assert.True(module.has_parameter("list.0.dict.first"));
            Assert.True(module.has_parameter("list.0.dict.second"));
            Assert.True(module.has_parameter("dict.first.test"));
            Assert.True(module.has_parameter("dict.first.list.0"));
            Assert.True(module.has_parameter("dict.first.dict.first"));
            Assert.True(module.has_parameter("dict.first.dict.second"));
            Assert.True(module.has_parameter("dict.second.test"));
            Assert.True(module.has_parameter("dict.second.list.0"));
            Assert.True(module.has_parameter("dict.second.dict.first"));
            Assert.True(module.has_parameter("dict.second.dict.second"));
        }

        [Fact]
        public void TestCustomModule3()
        {
            var module = new TestModule1(torch.randn(new long[] { 2, 2 }), true);

            var seq = Sequential(("test", module));

            Assert.True(module.has_parameter("test"));
            Assert.True(module.has_parameter("list.0"));
            Assert.True(module.has_parameter("dict.first"));
            Assert.True(module.has_parameter("dict.second"));

            Assert.True(seq.has_parameter("test.test"));
            Assert.True(seq.has_parameter("test.list.0"));
            Assert.True(seq.has_parameter("test.dict.first"));
            Assert.True(seq.has_parameter("test.dict.second"));
        }

        [Fact]
        public void TestCustomModule4()
        {
            var module = new TestModule1(torch.randn(new long[] { 2, 2 }), true);

            var seq = Sequential(module);

            Assert.True(module.has_parameter("test"));
            Assert.True(module.has_parameter("list.0"));
            Assert.True(module.has_parameter("dict.first"));
            Assert.True(module.has_parameter("dict.second"));

            Assert.True(seq.has_parameter("0.test"));
            Assert.True(seq.has_parameter("0.list.0"));
            Assert.True(seq.has_parameter("0.dict.first"));
            Assert.True(seq.has_parameter("0.dict.second"));
        }

        private class TestModule1 : Module<Tensor, Tensor>
        {
            public TestModule1(Tensor tensor, bool withGrad)
                : base("TestModule1")
            {
                test = Parameter(tensor.clone(), withGrad);
                list.append(Parameter(tensor.clone(), withGrad));
                dict.Add("first", Parameter(tensor.clone(), withGrad));
                dict.Add("second", Parameter(tensor.clone(), withGrad));
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                throw new NotImplementedException();
            }

            private Parameter test;
            private ParameterList list = new ParameterList();
            private ParameterDict dict = new ParameterDict();
        }

        private class TestModule2 : Module<Tensor, Tensor>
        {
            public TestModule2(Tensor tensor, bool withGrad)
                : base("TestModule1")
            {
                submodule = new TestModule1(tensor.clone(), withGrad);

                list.append(new TestModule1(tensor.clone(), withGrad));
                dict.Add("first", new TestModule1(tensor.clone(), withGrad));
                dict.Add("second", new TestModule1(tensor.clone(), withGrad));
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                for (int i = 0; i < list.Count; i++) { input = list[i].forward(input); }
                throw new NotImplementedException();
            }

            public Module<Tensor, Tensor> submodule;
            private ModuleList<Module<torch.Tensor, torch.Tensor>> list = new ModuleList<Module<torch.Tensor, torch.Tensor>>();
            private ModuleDict<Module<torch.Tensor, torch.Tensor>> dict = new ModuleDict<Module<torch.Tensor, torch.Tensor>>();
        }

        private class SequenceModel1 : Sequential
        {
            public SequenceModel1() :
                base(Linear(1000, 100, false),
                     Linear(100, 10, false))
            {
            }
        }

        private class SequenceModel2 : Sequential
        {
            public SequenceModel2() :
                base(("lin1", Linear(1000, 100, false)),
                     ("lin2", Linear(100, 10, false)))
            {
            }
        }

        [Fact]
        public void TestDerivedSequence1Grad()
        {
            using var seq = new SequenceModel1();

            var x = torch.randn(new long[] { 64, 1000 }, requires_grad: true);
            var y = torch.randn(new long[] { 64, 10 }, requires_grad: true);

            var eval = seq.forward(x);
            var loss = MSELoss(Reduction.Sum);
            var output = loss.forward(eval, y);

            seq.zero_grad();

            output.backward();

            foreach (var parm in seq.parameters()) {
                var grad = parm.grad();
            }
        }

        [Fact]
        public void TestDerivedSequence2Grad()
        {
            using var seq = new SequenceModel2();

            var x = torch.randn(new long[] { 64, 1000 }, requires_grad: true);
            var y = torch.randn(new long[] { 64, 10 }, requires_grad: true);

            var eval = seq.forward(x);
            var loss = MSELoss(Reduction.Sum);
            var output = loss.forward(eval, y);

            seq.zero_grad();

            output.backward();

            foreach (var parm in seq.parameters()) {
                var grad = parm.grad();
            }
        }


        [Fact]
        public void TestDatatypeTo()
        {
            var mod = new TestModule3();
            mod.ValidateDtype(torch.float32);
            mod.to(torch.float64);
            mod.ValidateDtype(torch.float64);

            var lin1 = Linear(10, 10);
            var lin2 = Linear(25, 25);
            var seq = Sequential(lin1, lin2);

            Assert.Equal(torch.float32, lin1.weight!.dtype);
            Assert.Equal(torch.float32, lin2.weight!.dtype);
            if (lin1.bias is not null) Assert.Equal(torch.float32, lin1.bias!.dtype);
            if (lin2.bias is not null) Assert.Equal(torch.float32, lin2.bias!.dtype);

            seq.to(torch.float64);

            Assert.Equal(torch.float64, lin1.weight!.dtype);
            Assert.Equal(torch.float64, lin2.weight!.dtype);
            if (lin1.bias is not null) Assert.Equal(torch.float64, lin1.bias!.dtype);
            if (lin2.bias is not null) Assert.Equal(torch.float64, lin2.bias!.dtype);
        }

        [Fact]
        public void TestDeviceTo()
        {
            if (torch.cuda.is_available()) {

                var mod = new TestModule3();
                mod.ValidateDeviceType(DeviceType.CPU);
                mod.cuda();
                mod.ValidateDeviceType(DeviceType.CUDA);

                var lin1 = Linear(10, 10);
                var lin2 = Linear(25, 25);
                var seq = Sequential(lin1, lin2);

                Assert.Equal(DeviceType.CPU, lin1.weight!.device_type);
                Assert.Equal(DeviceType.CPU, lin2.weight!.device_type);
                if (lin1.bias is not null) Assert.Equal(DeviceType.CPU, lin1.bias!.device_type);
                if (lin2.bias is not null) Assert.Equal(DeviceType.CPU, lin2.bias!.device_type);

                seq.cuda();

                Assert.Equal(DeviceType.CUDA, lin1.weight!.device_type);
                Assert.Equal(DeviceType.CUDA, lin2.weight!.device_type);
                if (lin1.bias is not null) Assert.Equal(DeviceType.CUDA, lin1.bias!.device_type);
                if (lin2.bias is not null) Assert.Equal(DeviceType.CUDA, lin2.bias!.device_type);
            }
        }

        private class TestModule3 : Module<Tensor, Tensor>
        {
            public TestModule3() : base(nameof(TestModule3)) { RegisterComponents(); }

            public override Tensor forward(Tensor t)
            {
                return mod2.forward(mod1.forward(t));
            }

            public void ValidateDtype(ScalarType dtype)
            {
                Assert.Equal(dtype, mod1.weight!.dtype);
                Assert.Equal(dtype, mod2.weight!.dtype);
                if (mod1.bias is not null) Assert.Equal(dtype, mod1.bias.dtype);
                if (mod2.bias is not null) Assert.Equal(dtype, mod2.bias.dtype);

                Assert.Equal(dtype, p1.dtype);
                Assert.Equal(dtype, b1.dtype);

                foreach (var p in parameters()) {
                    Assert.Equal(dtype, p.dtype);
                }
            }

            public void ValidateDeviceType(DeviceType device)
            {
                Assert.Equal(device, mod1.weight!.device_type);
                Assert.Equal(device, mod2.weight!.device_type);
                if (mod1.bias is not null) Assert.Equal(device, mod1.bias.device_type);
                if (mod2.bias is not null) Assert.Equal(device, mod2.bias.device_type);

                Assert.Equal(device, p1.device_type);
                Assert.Equal(device, b1.device_type);

                foreach (var p in parameters()) {
                    Assert.Equal(device, p.device_type);
                }
            }

            public Tensor b1 = torch.zeros(5, 5, 5);
            private Modules.Linear mod1 = Linear(10, 10);
            private Modules.Linear mod2 = Linear(10, 10);
            private Parameter p1 = new Parameter(torch.zeros(5, 5, 5), true);
        }
        #endregion

        #region Pooling
        [Fact]
        public void AvgPool2DObjectInitialized()
        {
            Tensor ones = torch.ones(new long[] { 2, 2, 2 });
            var obj = avg_pool2d(ones, new long[] { 2, 2 }, new long[] { 2, 2 });
            Assert.Equal(typeof(Tensor), obj.GetType());
        }

        [Fact]
        public void AvgPool2DTensor()
        {
            {
                Tensor ones = torch.ones(new long[] { 4, 2, 2, 2 });
                var obj = torch.nn.functional.avg_pool2d(ones, new long[] { 2, 2 });
                Assert.Equal(typeof(Tensor), obj.GetType());
                Assert.Equal(torch.ones(new long[] { 4, 2, 1, 1 }), obj);
            }
            {
                Tensor ones = torch.ones(new long[] { 4, 2, 2, 2 });
                var obj = torch.nn.functional.avg_pool2d(ones, (2, 2));
                Assert.Equal(typeof(Tensor), obj.GetType());
                Assert.Equal(torch.ones(new long[] { 4, 2, 1, 1 }), obj);
            }
            {
                Tensor ones = torch.ones(new long[] { 4, 2, 2, 2 });
                var obj = torch.nn.functional.avg_pool2d(ones, 2);
                Assert.Equal(typeof(Tensor), obj.GetType());
                Assert.Equal(torch.ones(new long[] { 4, 2, 1, 1 }), obj);
            }
        }

        [Fact]
        public void AvgPool2DTensorNN()
        {
            Tensor ones = torch.ones(new long[] { 4, 2, 2, 2 });
            {
                var obj = AvgPool2d(new long[] { 2, 2 }).forward(ones);
                Assert.Equal(typeof(Tensor), obj.GetType());
                Assert.Equal(torch.ones(new long[] { 4, 2, 1, 1 }), obj);
            }
            {
                var obj = AvgPool2d(2).forward(ones);
                Assert.Equal(typeof(Tensor), obj.GetType());
                Assert.Equal(torch.ones(new long[] { 4, 2, 1, 1 }), obj);
            }
            {
                var obj = AvgPool2d((2, 2)).forward(ones);
                Assert.Equal(typeof(Tensor), obj.GetType());
                Assert.Equal(torch.ones(new long[] { 4, 2, 1, 1 }), obj);
            }
        }

        [Fact]
        public void AdaptiveAvgPool2DTensorNN()
        {
            {
                Tensor ones = torch.ones(new long[] { 4, 2, 2, 2 });
                var obj = AdaptiveAvgPool2d(new long[] { 2, 2 }).forward(ones);
                Assert.Equal(typeof(Tensor), obj.GetType());
                Assert.Equal(torch.ones(new long[] { 4, 2, 2, 2 }), obj);
            }
            {
                Tensor ones = torch.ones(new long[] { 4, 2, 2, 2 });
                var obj = AdaptiveAvgPool2d(2).forward(ones);
                Assert.Equal(typeof(Tensor), obj.GetType());
                Assert.Equal(torch.ones(new long[] { 4, 2, 2, 2 }), obj);
            }
            {
                Tensor ones = torch.ones(new long[] { 4, 2, 2, 2 });
                var obj = AdaptiveAvgPool2d((2, 2)).forward(ones);
                Assert.Equal(typeof(Tensor), obj.GetType());
                Assert.Equal(torch.ones(new long[] { 4, 2, 2, 2 }), obj);
            }
        }

        [Fact]
        public void AvgPool2DBackwardTensor()
        {
            var ones = torch.ones(new long[] { 4, 2, 2, 2 });
            var kernelSize = new long[] { 2, 2 };
            var avg = torch.ones(new long[] { 4, 2, 1, 1 });
            var res = torch.nn.functional.avg_pool2d_backward(avg, ones, kernelSize) * 4.0;

            var ones0000 = ones[0, 0, 0, 0].ToSingle();
            var res0000 = res[0, 0, 0, 0].ToSingle();
            Assert.Equal(ones0000, res0000);
            // This gets back to the original uniform input
            Assert.Equal(res, ones);
        }


        [Fact]
        public void AvgPool3DBackwardTensor()
        {
            var ones = torch.ones(new long[] { 4, 2, 2, 2, 2 });
            var kernelSize = new long[] { 2, 2, 2 };
            var avg = torch.ones(new long[] { 4, 2, 1, 1, 1 });
            var res = torch.nn.functional.avg_pool3d_backward(avg, ones, kernelSize) * 8.0;

            var ones0000 = ones[0, 0, 0, 0, 0].ToSingle();
            var res0000 = res[0, 0, 0, 0, 0].ToSingle();
            Assert.True(Math.Abs(ones0000 - res0000) < 0.00001);
            // This gets back to the original uniform input
            Assert.True(res.allclose(ones));
        }

        [Fact]
        public void AvgPool3DBackwardTensorExplicitDivisor()
        {
            var ones = torch.ones(new long[] { 4, 2, 2, 2, 2 });
            var kernelSize = new long[] { 2, 2, 2 };
            var avg = torch.ones(new long[] { 4, 2, 1, 1, 1 });
            var res = torch.nn.functional.avg_pool3d_backward(avg, ones, kernelSize, divisorOverride: 6) * 6.0;

            var ones0000 = ones[0, 0, 0, 0, 0].ToSingle();
            var res0000 = res[0, 0, 0, 0, 0].ToSingle();
            Assert.True(Math.Abs(ones0000 - res0000) < 0.00001);
            // This gets back to the original uniform input
            Assert.True(res.allclose(ones));
        }

        [Fact]
        public void MaxPool2DObjectInitialized()
        {
            {
                Tensor ones = torch.ones(new long[] { 2, 2, 2 });
                var obj = max_pool2d(ones, new long[] { 2, 2 }, new long[] { 2, 2 });
                Assert.Equal(typeof(Tensor), obj.GetType());
            }
            {
                Tensor ones = torch.ones(new long[] { 2, 2, 2 });
                var obj = max_pool2d(ones, (2, 2), (2, 2));
                Assert.Equal(typeof(Tensor), obj.GetType());
            }
            {
                Tensor ones = torch.ones(new long[] { 2, 2, 2 });
                var obj = max_pool2d(ones, 2, 2);
                Assert.Equal(typeof(Tensor), obj.GetType());
            }
            {
                Tensor ones = torch.ones(new long[] { 2, 2, 2 });
                var obj = max_pool2d(ones, 2);
                Assert.Equal(typeof(Tensor), obj.GetType());
            }
        }

        [Fact]
        public void TestMaxPool1D_1()
        {
            Tensor ones = torch.ones(new long[] { 16, 3, 4 });
            using (var pool = MaxPool1d(2)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 3, 2 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());

                var (output, indices) = pool.forward_with_indices(ones);
                Assert.Equal(expShape, output.shape);
                Assert.Equal(expShape, indices.shape);
            }
        }

        [Fact]
        public void TestMaxPool1D_2()
        {
            Tensor ones = torch.ones(new long[] { 16, 3, 4 });
            using (var pool = MaxPool1d(2, 1)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 3, 3 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0].ToSingle());
            }
        }

        [Fact]
        public void TestMaxPool1D_3()
        {
            Tensor ones = torch.ones(new long[] { 16, 32, 40 });
            using (var pool = MaxPool1d(3, 1, 1, 2)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 32, 38 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0].ToSingle());
            }
            using (var pool = MaxPool1d(3, 1, 1, 2, true)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 32, 38 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0].ToSingle());
            }
        }

        [Fact]
        public void TestMaxPool2D_1()
        {
            Tensor ones = torch.ones(new long[] { 16, 4, 4 });
            using (var pool = MaxPool2d(2)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 2, 2 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1].ToSingle());

                var witIdx = pool.forward_with_indices(ones);
                Assert.Equal(expShape, witIdx.Values.shape);
                Assert.Equal(expShape, witIdx.Indices.shape);
            }
            using (var pool = MaxPool2d((2, 2))) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 2, 2 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1].ToSingle());

                var (output, indices) = pool.forward_with_indices(ones);
                Assert.Equal(expShape, output.shape);
                Assert.Equal(expShape, indices.shape);
            }
        }

        [Fact]
        public void TestMaxPool2D_2()
        {
            Tensor ones = torch.ones(new long[] { 16, 4, 4 });
            using (var pool = MaxPool2d(2, 1)) {
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
            using (var pool = MaxPool2d((2, 2), (1, 1))) {
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
        public void TestMaxPool2D_3()
        {
            Tensor ones = torch.ones(new long[] { 16, 32, 40 });
            using (var pool = MaxPool2d(3, 1, 1, 2)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 30, 38 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0].ToSingle());
            }
            using (var pool = MaxPool2d((3, 3), (1, 1), (1, 1), (2, 2))) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 30, 38 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0].ToSingle());
            }
            using (var pool = MaxPool2d(3, 1, 1, 2, true)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 30, 38 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0].ToSingle());
            }
            using (var pool = MaxPool2d((3, 3), (1, 1), (1, 1), (2, 2), true)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 30, 38 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 2, 0].ToSingle());
            }
        }

        [Fact]
        public void TestMaxPool3D_1()
        {
            Tensor ones = torch.ones(new long[] { 16, 4, 4, 8 });
            using (var pool = MaxPool3d(2)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 2, 2, 4 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1, 0].ToSingle());

                var witIdx = pool.forward_with_indices(ones);
                Assert.Equal(expShape, witIdx.Values.shape);
                Assert.Equal(expShape, witIdx.Indices.shape);
            }
            using (var pool = MaxPool3d((2, 2, 2))) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 2, 2, 4 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1, 0].ToSingle());

                var (output, indices) = pool.forward_with_indices(ones);
                Assert.Equal(expShape, output.shape);
                Assert.Equal(expShape, indices.shape);
            }
        }

        [Fact]
        public void TestMaxPool3D_2()
        {
            Tensor ones = torch.ones(new long[] { 16, 4, 4, 8 });
            using (var pool = MaxPool3d(2, 1)) {
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
            using (var pool = MaxPool3d((2, 2, 2), (1, 1, 1))) {
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
        public void TestMaxPool3D_3()
        {
            Tensor ones = torch.ones(new long[] { 16, 32, 40, 40 });
            using (var pool = MaxPool3d(3, 1, 1, 2)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 30, 38, 38 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 0].ToSingle());
            }
            using (var pool = MaxPool3d(3, 1, 1, 2, true)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 30, 38, 38 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 0].ToSingle());
            }
            using (var pool = MaxPool3d((3, 3, 3), (1, 1, 1), (1, 1, 1), (2, 2, 2))) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 30, 38, 38 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 0].ToSingle());
            }
            using (var pool = MaxPool3d((3, 3, 3), (1, 1, 1), (1, 1, 1), (2, 2, 2), true)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 30, 38, 38 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 2, 0].ToSingle());
            }
        }

        [Fact]
        public void TestFractionalMaxPool2D_1()
        {
            Tensor ones = torch.ones(new long[] { 16, 24, 24 });
            using (var pool = FractionalMaxPool2d(2, 12)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 12, 12 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1].ToSingle());

                var (output, indices) = pool.forward_with_indices(ones);
                Assert.Equal(expShape, output.shape);
                Assert.Equal(expShape, indices.shape);
            }
            using (var pool = FractionalMaxPool2d((2, 2), (12, 16))) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 12, 16 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1].ToSingle());

                var (output, indices) = pool.forward_with_indices(ones);
                Assert.Equal(expShape, output.shape);
                Assert.Equal(expShape, indices.shape);
            }
        }

        [Fact]
        public void TestFractionalMaxPool2D_2()
        {
            Tensor ones = torch.ones(new long[] { 16, 24, 24 });
            using (var pool = FractionalMaxPool2d(2, output_ratio: 0.5)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 12, 12 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1].ToSingle());

                var (output, indices) = pool.forward_with_indices(ones);
                Assert.Equal(expShape, output.shape);
                Assert.Equal(expShape, indices.shape);
            }
            using (var pool = FractionalMaxPool2d((2, 2), output_ratio: (0.5, 2.0 / 3.0))) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 12, 16 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1].ToSingle());

                var (output, indices) = pool.forward_with_indices(ones);
                Assert.Equal(expShape, output.shape);
                Assert.Equal(expShape, indices.shape);
            }
        }

        [Fact]
        public void TestFractionalMaxPool3D_1()
        {
            Tensor ones = torch.ones(new long[] { 16, 24, 24, 24 });
            using (var pool = FractionalMaxPool3d(2, 12)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 12, 12, 12 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1, 0].ToSingle());

                var (output, indices) = pool.forward_with_indices(ones);
                Assert.Equal(expShape, output.shape);
                Assert.Equal(expShape, indices.shape);
            }
            using (var pool = FractionalMaxPool3d((2, 2, 2), (12, 16, 20))) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 12, 16, 20 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1, 0].ToSingle());

                var (output, indices) = pool.forward_with_indices(ones);
                Assert.Equal(expShape, output.shape);
                Assert.Equal(expShape, indices.shape);
            }
        }

        [Fact]
        public void TestFractionalMaxPool3D_2()
        {
            Tensor ones = torch.ones(new long[] { 16, 3, 24, 24, 24 });
            using (var pool = FractionalMaxPool3d(2, output_ratio: 0.5)) {
                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 3, 12, 12, 12 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1, 0, 0].ToSingle());

                var (output, indices) = pool.forward_with_indices(ones);
                Assert.Equal(expShape, output.shape);
                Assert.Equal(expShape, indices.shape);
            }
            using (var pool = FractionalMaxPool3d((2, 2, 2), output_ratio: (0.5, 2.0 / 3.0, 5.0 / 6.0))) {

                var pooled = pool.forward(ones);
                var expShape = new long[] { 16, 3, 12, 16, 20 };
                Assert.Equal(expShape, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 0, 1, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 1, 0, 0].ToSingle());

                var (output, indices) = pool.forward_with_indices(ones);
                Assert.Equal(expShape, output.shape);
                Assert.Equal(expShape, indices.shape);
            }
        }

        [Fact]
        public void TestMaxUnpool1D_1()
        {
            Tensor input = torch.tensor(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }).reshape(1, 1, 8);

            using var pool = MaxPool1d(2, 2);
            using var unpool = MaxUnpool1d(2, 2);

            var expShape = new long[] { 1, 1, 4 };

            var (output, indices) = pool.forward_with_indices(input);
            Assert.Equal(expShape, output.shape);
            Assert.Equal(expShape, indices.shape);

            var result = unpool.forward(output, indices);
            Tensor expected = torch.tensor(new float[] { 0, 2, 0, 4, 0, 6, 0, 8 }).reshape(1, 1, 8);
            Assert.Equal(input.shape, result.shape);
            Assert.Equal(expected, result);
        }

        [Fact]
        public void TestMaxUnpool2D_1()
        {
            Tensor input = torch.tensor(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }).reshape(1, 1, 4, 4);

            using var pool = MaxPool2d(2, 2);
            using var unpool = MaxUnpool2d(2, 2);

            var expShape = new long[] { 1, 1, 2, 2 };

            var (output, indices) = pool.forward_with_indices(input);
            Assert.Equal(expShape, output.shape);
            Assert.Equal(expShape, indices.shape);

            var result = unpool.forward(output, indices);
            Tensor expected = torch.tensor(new float[] { 0, 0, 0, 0, 0, 6, 0, 8, 0, 0, 0, 0, 0, 14, 0, 16 }).reshape(1, 1, 4, 4);
            Assert.Equal(input.shape, result.shape);
            Assert.Equal(expected, result);
        }

        [Fact]
        public void TestMaxUnpool2D_2()
        {
            Tensor input = torch.tensor(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 }).reshape(1, 1, 4, 5);

            using var pool = MaxPool2d(2, 2);
            using var unpool = MaxUnpool2d(2, 2);

            var expShape = new long[] { 1, 1, 2, 2 };

            var (output, indices) = pool.forward_with_indices(input);
            Assert.Equal(expShape, output.shape);
            Assert.Equal(expShape, indices.shape);

            var result = unpool.forward(output, indices, output_size: input.shape);
            Tensor expected = torch.tensor(new float[] { 0, 0, 0, 0, 0, 0, 7, 0, 9, 0, 0, 0, 0, 0, 0, 0, 17, 0, 19, 0 }).reshape(1, 1, 4, 5);
            Assert.Equal(input.shape, result.shape);
            Assert.Equal(expected, result);
        }

        [Fact]
        public void TestMaxUnpool3D_1()
        {
            Tensor input = torch.tensor(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }).reshape(1, 1, 2, 2, 4);

            using var pool = MaxPool3d(2, 2);
            using var unpool = MaxUnpool3d(2, 2);

            var expShape = new long[] { 1, 1, 1, 1, 2 };

            var (output, indices) = pool.forward_with_indices(input);
            Assert.Equal(expShape, output.shape);
            Assert.Equal(expShape, indices.shape);

            var result = unpool.forward(output, indices);
            Tensor expected = torch.tensor(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 16 }).reshape(1, 1, 2, 2, 4);
            Assert.Equal(input.shape, result.shape);
            Assert.Equal(expected, result);
        }

        [Fact]
        public void TestMaxUnpool3D_2()
        {
            Tensor input = torch.tensor(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 }).reshape(1, 1, 2, 2, 5);

            using var pool = MaxPool3d(2, 2);
            using var unpool = MaxUnpool3d(2, 2);

            var expShape = new long[] { 1, 1, 1, 1, 2 };

            var (output, indices) = pool.forward_with_indices(input);
            Assert.Equal(expShape, output.shape);
            Assert.Equal(expShape, indices.shape);

            var result = unpool.forward(output, indices, output_size: input.shape);
            Tensor expected = torch.tensor(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 19, 0 }).reshape(1, 1, 2, 2, 5);
            Assert.Equal(input.shape, result.shape);
            Assert.Equal(expected, result);
        }

        [Fact]
        public void TestAvgPool1D_1()
        {
            Tensor ones = torch.ones(new long[] { 16, 3, 4 });
            using (var pool = AvgPool1d(2)) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 3, 2 }, pooled.shape);
                Assert.Equal(1, pooled[0, 0, 0].ToSingle());
                Assert.Equal(1, pooled[0, 1, 0].ToSingle());
            }
        }

        [Fact]
        public void TestAvgPool1D_2()
        {
            Tensor ones = torch.ones(new long[] { 16, 3, 4 });
            using (var pool = AvgPool1d(2, 1)) {
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
            Tensor ones = torch.ones(new long[] { 16, 4, 4 });
            using (var pool = AvgPool2d(new long[] { 2, 2 })) {
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
            Tensor ones = torch.ones(new long[] { 16, 4, 4 });
            using (var pool = AvgPool2d(new long[] { 2, 2 }, new long[] { 1, 1 })) {
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

            ones = torch.ones(new long[] { 16, 4, 4, 4 });
            using (var pool = AvgPool2d(new long[] { 2, 2 }, new long[] { 1, 1 })) {
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
            Tensor ones = torch.ones(new long[] { 16, 4, 4, 8 });
            using (var pool = AvgPool3d(new long[] { 2, 2, 2 })) {
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
            Tensor ones = torch.ones(new long[] { 16, 4, 4, 8 });
            using (var pool = AvgPool3d(new long[] { 2, 2, 2 }, new long[] { 1, 1, 1 })) {
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

            ones = torch.ones(new long[] { 16, 3, 4, 4, 8 });
            using (var pool = AvgPool3d(new long[] { 2, 2, 2 }, new long[] { 1, 1, 1 })) {
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

        static float sqrt2 = (float)Math.Sqrt(2.0); // Can't use MathF because of .NET FX 4.7

        [Fact]
        public void TestLPPool1D_1()
        {
            Tensor ones = torch.ones(new long[] { 16, 3, 4 });
            using (var pool = LPPool1d(2, 2)) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 3, 2 }, pooled.shape);
                Assert.Equal(sqrt2, pooled[0, 0, 0].ToSingle());
                Assert.Equal(sqrt2, pooled[0, 1, 0].ToSingle());
            }
        }

        [Fact]
        public void TestLPPool1D_2()
        {
            Tensor ones = torch.ones(new long[] { 16, 3, 4 });
            using (var pool = LPPool1d(2, 2, 1)) {
                var pooled = pool.forward(ones);

                Assert.Equal(new long[] { 16, 3, 3 }, pooled.shape);
                Assert.Equal(sqrt2, pooled[0, 0, 0].ToSingle());
                Assert.Equal(sqrt2, pooled[0, 1, 0].ToSingle());
                Assert.Equal(sqrt2, pooled[0, 2, 0].ToSingle());
            }
        }

        [Fact]
        public void TestLPPool2D_1()
        {
            Tensor ones = torch.ones(new long[] { 16, 4, 4 });
            using (var pool = LPPool2d(2, new long[] { 2, 2 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 2, 2 }, pooled.shape);
                Assert.Equal(2, pooled[0, 0, 0].ToSingle());
                Assert.Equal(2, pooled[0, 0, 1].ToSingle());
                Assert.Equal(2, pooled[0, 1, 0].ToSingle());
                Assert.Equal(2, pooled[0, 1, 1].ToSingle());
            }
        }

        [Fact]
        public void TestLPPool2D_2()
        {
            Tensor ones = torch.ones(new long[] { 16, 4, 4 });
            using (var pool = LPPool2d(2, new long[] { 2, 2 }, new long[] { 1, 1 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 3, 3 }, pooled.shape);
                Assert.Equal(2, pooled[0, 0, 0].ToSingle());
                Assert.Equal(2, pooled[0, 0, 1].ToSingle());
                Assert.Equal(2, pooled[0, 0, 2].ToSingle());
                Assert.Equal(2, pooled[0, 1, 0].ToSingle());
                Assert.Equal(2, pooled[0, 1, 1].ToSingle());
                Assert.Equal(2, pooled[0, 1, 2].ToSingle());
                Assert.Equal(2, pooled[0, 2, 0].ToSingle());
                Assert.Equal(2, pooled[0, 2, 1].ToSingle());
                Assert.Equal(2, pooled[0, 2, 2].ToSingle());
            }

            ones = torch.ones(new long[] { 16, 4, 4, 4 });
            using (var pool = LPPool2d(2, new long[] { 2, 2 }, new long[] { 1, 1 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(new long[] { 16, 4, 3, 3 }, pooled.shape);
                Assert.Equal(2, pooled[0, 0, 0, 0].ToSingle());
                Assert.Equal(2, pooled[0, 0, 0, 1].ToSingle());
                Assert.Equal(2, pooled[0, 0, 0, 2].ToSingle());
                Assert.Equal(2, pooled[0, 0, 1, 0].ToSingle());
                Assert.Equal(2, pooled[0, 0, 1, 1].ToSingle());
                Assert.Equal(2, pooled[0, 0, 1, 2].ToSingle());
                Assert.Equal(2, pooled[0, 0, 2, 0].ToSingle());
                Assert.Equal(2, pooled[0, 0, 2, 1].ToSingle());
                Assert.Equal(2, pooled[0, 0, 2, 2].ToSingle());
            }
        }
        #endregion

        #region Normalization
        [Fact]
        public void TestBatchNorm1D()
        {
            {
                var ones = torch.ones(new long[] { 16, 3, 28 });
                using (var pool = BatchNorm1d(3)) {

                    var sd = pool.state_dict();
                    Assert.Equal(5, sd.Count);
                    var np = pool.named_parameters();
                    Assert.Equal(2, np.Count());
                    var nb = pool.named_buffers();
                    Assert.Equal(3, nb.Count());

                    var pooled = pool.forward(ones);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2 })));
                }
            }
            {
                var ones = torch.ones(new long[] { 1, 3, 28 });
                using (var pool = BatchNorm1d(3)) {
                    var pooled = pool.forward(ones);
                    Assert.Equal(ones.shape, pooled.shape);
                }
            }
            {
                var ones = torch.ones(new long[] { 16, 28 });
                using (var pool = BatchNorm1d(28)) {
                    var pooled = pool.forward(ones);
                    Assert.Equal(ones.shape, pooled.shape);
                }
            }
        }

        [Fact]
        public void TestBatchNorm1DWeightAndBias()
        {
            var ones = torch.ones(new long[] { 16, 3, 28 });

            using (var norm = BatchNorm1d(3, track_running_stats: false)) {
                var pooled = norm.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);

                var w = norm.weight;
                var b = norm.bias;

                Assert.NotNull(w);
                Assert.NotNull(b);

                Assert.Null(norm.running_mean);
                Assert.Null(norm.running_var);

                Assert.Equal(new long[] { 3 }, w.shape);
                Assert.Equal(new long[] { 3 }, b.shape);
            }
        }

        [Fact]
        public void TestBatchNorm1DRunningStats()
        {
            var ones = torch.ones(new long[] { 16, 3, 28 });

            using (var norm = BatchNorm1d(3, track_running_stats: true)) {
                var pooled = norm.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);

                var m = norm.running_mean;
                var v = norm.running_var;

                Assert.NotNull(m);
                Assert.NotNull(v);

                if (m is not null) Assert.Equal(new long[] { 3 }, m.shape);
                if (v is not null) Assert.Equal(new long[] { 3 }, v.shape);
            }
        }

        [Fact]
        public void TestBatchNorm2D()
        {
            {
                var ones = torch.ones(new long[] { 16, 3, 28, 28 });
                using (var pool = BatchNorm2d(3)) {
                    var pooled = pool.forward(ones);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16, 2, 2 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2, 2 })));
                }
            }
            {
                var ones = torch.ones(new long[] { 1, 3, 28, 28 });
                using (var pool = BatchNorm2d(3)) {
                    var pooled = pool.forward(ones);
                    Assert.Equal(ones.shape, pooled.shape);
                }
            }
        }

        [Fact]
        public void TestBatchNorm2dWeightAndBias()
        {
            var ones = torch.ones(new long[] { 16, 3, 28, 28 });

            using (var norm = BatchNorm2d(3, track_running_stats: false)) {
                var pooled = norm.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);

                var w = norm.weight;
                var b = norm.bias;

                Assert.NotNull(w);
                Assert.NotNull(b);

                Assert.Null(norm.running_mean);
                Assert.Null(norm.running_var);

                Assert.Equal(new long[] { 3 }, w.shape);
                Assert.Equal(new long[] { 3 }, b.shape);
            }
        }

        [Fact]
        public void TestBatchNorm2dRunningStats()
        {
            var ones = torch.ones(new long[] { 16, 3, 28, 28 });

            using (var norm = BatchNorm2d(3, track_running_stats: true)) {
                var pooled = norm.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);

                var m = norm.running_mean;
                var v = norm.running_var;

                Assert.NotNull(m);
                Assert.NotNull(v);

                if (m is not null) Assert.Equal(new long[] { 3 }, m.shape);
                if (v is not null) Assert.Equal(new long[] { 3 }, v.shape);
            }
        }

        [Fact]
        public void TestBatchNorm3D()
        {
            var ones = torch.ones(new long[] { 16, 3, 12, 28, 28 });
            using (var pool = BatchNorm3d(3)) {
                var pooled = pool.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);
                Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16, 2, 2, 2 })));
                Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2, 2, 2 })));
            }
        }

        [Fact]
        public void TestBatchNorm3dWeightAndBias()
        {
            var ones = torch.ones(new long[] { 16, 3, 12, 28, 28 });

            using (var norm = BatchNorm3d(3, track_running_stats: false)) {
                var pooled = norm.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);

                var w = norm.weight;
                var b = norm.bias;

                Assert.NotNull(w);
                Assert.NotNull(b);

                Assert.Null(norm.running_mean);
                Assert.Null(norm.running_var);

                Assert.Equal(new long[] { 3 }, w.shape);
                Assert.Equal(new long[] { 3 }, b.shape);
            }
        }

        [Fact]
        public void TestBatchNorm3dRunningStats()
        {
            var ones = torch.ones(new long[] { 16, 3, 12, 28, 28 });

            using (var norm = BatchNorm3d(3, track_running_stats: true)) {
                var pooled = norm.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);

                var m = norm.running_mean;
                var v = norm.running_var;

                Assert.NotNull(m);
                Assert.NotNull(v);

                if (m is not null) Assert.Equal(new long[] { 3 }, m.shape);
                if (v is not null) Assert.Equal(new long[] { 3 }, v.shape);
            }
        }

        [Fact]
        public void TestInstanceNorm1D()
        {
            var ones = torch.ones(new long[] { 16, 3, 28 });
            {
                using (var pool = InstanceNorm1d(3)) {
                    var pooled = pool.forward(ones);
                    Assert.Null(pool.weight);
                    Assert.Null(pool.bias);
                    Assert.Null(pool.running_mean);
                    Assert.Null(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2 })));
                }
            }
            {
                using (var pool = InstanceNorm1d(3, affine: true)) {
                    var pooled = pool.forward(ones);
                    Assert.NotNull(pool.weight);
                    Assert.NotNull(pool.bias);
                    Assert.Null(pool.running_mean);
                    Assert.Null(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2 })));
                }
            }
            {
                using (var pool = InstanceNorm1d(3, track_running_stats: true)) {
                    var pooled = pool.forward(ones);
                    Assert.Null(pool.weight);
                    Assert.Null(pool.bias);
                    Assert.NotNull(pool.running_mean);
                    Assert.NotNull(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2 })));
                }
            }
            {
                using (var pool = InstanceNorm1d(3, affine: true, track_running_stats: true)) {
                    var pooled = pool.forward(ones);
                    Assert.NotNull(pool.weight);
                    Assert.NotNull(pool.bias);
                    Assert.NotNull(pool.running_mean);
                    Assert.NotNull(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2 })));
                }
            }
        }

        [Fact]
        public void TestInstanceNorm2D()
        {
            var ones = torch.ones(new long[] { 16, 3, 28, 28 });
            {
                using (var pool = InstanceNorm2d(3)) {
                    var pooled = pool.forward(ones);
                    Assert.Null(pool.weight);
                    Assert.Null(pool.bias);
                    Assert.Null(pool.running_mean);
                    Assert.Null(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16, 2, 2 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2, 2 })));
                }
            }
            {
                using (var pool = InstanceNorm2d(3, affine: true)) {
                    var pooled = pool.forward(ones);
                    Assert.NotNull(pool.weight);
                    Assert.NotNull(pool.bias);
                    Assert.Null(pool.running_mean);
                    Assert.Null(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16, 2, 2 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2, 2 })));
                }
            }
            {
                using (var pool = InstanceNorm2d(3, track_running_stats: true)) {
                    var pooled = pool.forward(ones);
                    Assert.Null(pool.weight);
                    Assert.Null(pool.bias);
                    Assert.NotNull(pool.running_mean);
                    Assert.NotNull(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16, 2, 2 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2, 2 })));
                }
            }
            {
                using (var pool = InstanceNorm2d(3, affine: true, track_running_stats: true)) {
                    var pooled = pool.forward(ones);
                    Assert.NotNull(pool.weight);
                    Assert.NotNull(pool.bias);
                    Assert.NotNull(pool.running_mean);
                    Assert.NotNull(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16, 2, 2 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2, 2 })));
                }
            }
        }

        [Fact]
        public void TestInstanceNorm3D()
        {
            var ones = torch.ones(new long[] { 16, 3, 12, 28, 28 });
            {
                using (var pool = InstanceNorm3d(3)) {
                    var pooled = pool.forward(ones);
                    Assert.Null(pool.weight);
                    Assert.Null(pool.bias);
                    Assert.Null(pool.running_mean);
                    Assert.Null(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16, 2, 2, 2 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2, 2, 2 })));
                }
            }
            {
                using (var pool = InstanceNorm3d(3, affine: true)) {
                    var pooled = pool.forward(ones);
                    Assert.NotNull(pool.weight);
                    Assert.NotNull(pool.bias);
                    Assert.Null(pool.running_mean);
                    Assert.Null(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16, 2, 2, 2 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2, 2, 2 })));
                }
            }
            {
                using (var pool = InstanceNorm3d(3, track_running_stats: true)) {
                    var pooled = pool.forward(ones);
                    Assert.Null(pool.weight);
                    Assert.Null(pool.bias);
                    Assert.NotNull(pool.running_mean);
                    Assert.NotNull(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16, 2, 2, 2 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2, 2, 2 })));
                }
            }
            {
                using (var pool = InstanceNorm3d(3, affine: true, track_running_stats: true)) {
                    var pooled = pool.forward(ones);
                    Assert.NotNull(pool.weight);
                    Assert.NotNull(pool.bias);
                    Assert.NotNull(pool.running_mean);
                    Assert.NotNull(pool.running_var);
                    Assert.Equal(ones.shape, pooled.shape);
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 16, 2, 2, 2 })));
                    Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2, 2, 2, 2, 2 })));
                }
            }
        }

        [Fact]
        public void TestLayerNorm()
        {
            var ones = torch.ones(new long[] { 16, 3, 12, 28, 28 });
            using (var pool = LayerNorm(new long[] { 12, 28, 28 })) {
                var pooled = pool.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);
            }
        }

        [Fact]
        public void TestLocalResponseNorm()
        {
            var ones = torch.ones(new long[] { 16, 3, 12, 28, 28 });
            using (var pool = LocalResponseNorm(2)) {
                var pooled = pool.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);
                Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2 })));
            }
        }

        [Fact]
        public void TestGroupNorm()
        {
            var ones = torch.ones(new long[] { 20, 6, 10, 10 });
            using (var pool = GroupNorm(3, 6)) {
                var pooled = pool.forward(ones);
                Assert.Equal(ones.shape, pooled.shape);
                Assert.Throws<ArgumentException>(() => pool.forward(torch.ones(new long[] { 2, 2 })));
            }
        }

        private Tensor NormalizeTensor(Tensor x, long[] dim, double eps = 1e-5)
        {
            var x_mean = torch.mean(x, dimensions: dim, keepdim: true);
            var x_var = torch.var(x, unbiased: false, dimensions: dim, keepdim: true);
            return NormalizeTensor(x, x_mean, x_var, eps);
        }

        private Tensor NormalizeTensor(Tensor x, Tensor x_mean, Tensor x_var, double eps = 1e-5)
        {
            return (x - x_mean) / torch.sqrt(eps + x_var);
        }

        [Fact]
        public void TestBatchNormFunc()
        {
            var x = torch.randn(3, 2, 4);
            var running_mean = torch.randn(2);
            var running_var = torch.square(torch.randn(2));
            var y = torch.nn.functional.batch_norm(x, running_mean, running_var);
            var z = NormalizeTensor(x, torch.unsqueeze(running_mean, 1), torch.unsqueeze(running_var, 1));
            Assert.InRange(torch.mean(torch.square(z - y)).item<float>(), 0, 1e-5);

            var weight = torch.randn(2);
            var bias = torch.randn(2);
            y = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias);
            z = torch.unsqueeze(weight, 1) * z + torch.unsqueeze(bias, 1);
            Assert.InRange(torch.mean(torch.square(z - y)).item<float>(), 0, 1e-5);

            y = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training: true);
            Assert.Equal(x.shape, y.shape);
        }

        [Fact]
        public void TestGroupNormFunc()
        {
            var x = torch.randn(3, 12, 5);
            var y = torch.nn.functional.group_norm(x, 4);
            y = y[TensorIndex.Colon, TensorIndex.Slice(3, 6)];
            var z = NormalizeTensor(x[TensorIndex.Colon, TensorIndex.Slice(3, 6)], new long[] { 1, 2 });
            Assert.InRange(torch.mean(torch.square(z - y)).item<float>(), 0, 1e-5);

            var weight = torch.randn(12);
            var bias = torch.randn(12);
            y = torch.nn.functional.group_norm(x, 4, weight, bias);
            y = y[TensorIndex.Colon, TensorIndex.Slice(3, 6)];
            z = weight[TensorIndex.Slice(3, 6), TensorIndex.None] * z + bias[TensorIndex.Slice(3, 6), TensorIndex.None];
            Assert.InRange(torch.mean(torch.square(z - y)).item<float>(), 0, 1e-5);
        }

        [Fact]
        public void TestInstanceNormFunc()
        {
            var x = torch.randn(3, 2, 5);
            var y = torch.nn.functional.instance_norm(x);
            var z = NormalizeTensor(x, new long[] { 2 });
            Assert.InRange(torch.mean(torch.square(z - y)).item<float>(), 0, 1e-5);

            var running_mean = torch.randn(2);
            var running_var = torch.square(torch.randn(2));
            y = torch.nn.functional.instance_norm(x, running_mean, running_var);
            Assert.InRange(torch.mean(torch.square(z - y)).item<float>(), 0, 1e-5);

            var weight = torch.randn(2);
            var bias = torch.randn(2);
            y = torch.nn.functional.instance_norm(x, running_mean, running_var, weight, bias);
            z = torch.unsqueeze(weight, 1) * z + torch.unsqueeze(bias, 1);
            Assert.InRange(torch.mean(torch.square(z - y)).item<float>(), 0, 1e-5);
        }

        [Fact]
        public void TestLayerNormFunc()
        {
            var x = torch.randn(3, 5, 12);
            var y = torch.nn.functional.layer_norm(x, new long[] { 12 });
            var z = NormalizeTensor(x, new long[] { 2 });
            Assert.InRange(torch.mean(torch.square(z - y)).item<float>(), 0, 1e-5);

            var weight = torch.randn(12);
            var bias = torch.randn(12);
            y = torch.nn.functional.layer_norm(x, new long[] { 12 }, weight, bias);
            z = weight * z + bias;
            Assert.InRange(torch.mean(torch.square(z - y)).item<float>(), 0, 1e-5);
        }

        [Fact]
        public void TestLocalResponseNormFunc()
        {
            var x = torch.randn(3, 6, 4);
            var y = torch.nn.functional.local_response_norm(x, 5, alpha: 0.5);
            y = y[TensorIndex.Colon, 3];
            var z = x[TensorIndex.Colon, 3] * torch.pow(torch.square(x[TensorIndex.Colon, TensorIndex.Slice(1, 6)]).sum(dim: 1) * 0.5 / 5 + 1, torch.tensor(-0.75f));
            Assert.InRange(torch.mean(torch.square(z - y)).item<float>(), 0, 1e-5);
        }
        #endregion

        #region Embedding, Encoding, Transformer
        [Fact]
        public void TestEmbeddingDefaults()
        {
            var ones = torch.ones(new long[] { 16 }, torch.int32);
            using (var emb = Embedding(1000, 12)) {
                var output = emb.forward(ones);
                Assert.Equal(new long[] { 16, 12 }, output.shape);
            }
        }

        [Fact]
        public void TestEmbeddingWithMaxNorm()
        {
            var ones = torch.ones(new long[] { 16 }, torch.int32);
            using (var emb = Embedding(1000, 128, max_norm: 1.5)) {
                var output = emb.forward(ones);
                Assert.Equal(new long[] { 16, 128 }, output.shape);
            }
        }

        [Fact]
        public void TestEmbeddingSetWeights()
        {
            var ones = torch.ones(new long[] { 16 }, torch.int32);
            using (var emb = Embedding(1000, 12)) {
                var weights = torch.randn(new long[] { 1000, 12 });

                emb.weight = weights.clone().AsParameter();

                Assert.Equal(emb.weight.shape.Length, weights.shape.Length);
                Assert.Equal(emb.weight.shape[0], weights.shape[0]);
                Assert.Equal(emb.weight.shape[1], weights.shape[1]);
            }
        }

        [Fact]
        public void TestEmbeddingFromPretrained()
        {
            var ones = torch.ones(new long[] { 16 }, torch.int32);
            var weights = torch.randn(new long[] { 1000, 12 });

            using (var emb = Embedding_from_pretrained(weights)) {
                Assert.Equal(emb.weight!.shape.Length, weights.shape.Length);
                Assert.Equal(emb.weight!.shape[0], weights.shape[0]);
                Assert.Equal(emb.weight!.shape[1], weights.shape[1]);
            }
        }

        [Fact]
        public void TestEmbeddingBagDefaults()
        {
            var ones = torch.ones(new long[] { 16, 12 }, torch.int64);
            using (var emb = EmbeddingBag(1000, 12)) {
                var output = emb.forward(ones);
                Assert.Equal(new long[] { 16, 12 }, output.shape);
            }
        }

        [Fact]
        public void TestEmbeddingBagWithMaxNormAndSum()
        {
            var ones = torch.ones(new long[] { 16, 12 }, torch.int64);
            using (var emb = EmbeddingBag(1000, 128, max_norm: 1.5, mode: EmbeddingBagMode.Sum)) {
                var output = emb.forward(ones);
                Assert.Equal(new long[] { 16, 128 }, output.shape);
            }
        }

        [Fact]
        public void TestEmbeddingBagWithOffsets()
        {
            var ones = torch.ones(new long[] { 16 }, torch.int32);
            var offsets = torch.tensor(new int[] { 0, 8 });
            using (var emb = EmbeddingBag(1000, 128, max_norm: 1.5, mode: EmbeddingBagMode.Sum)) {
                var output = emb.forward(ones, offsets);
                Assert.Equal(new long[] { offsets.shape[0], 128 }, output.shape);
            }
        }

        [Fact]
        public void TestEmbeddingBagSetWeights()
        {
            var ones = torch.ones(new long[] { 16 }, torch.int32);
            using (var emb = EmbeddingBag(1000, 12)) {
                var weights = torch.randn(new long[] { 1000, 12 });
                emb.weight = weights.clone().AsParameter();

                Assert.Equal(emb.weight.shape.Length, weights.shape.Length);
                Assert.Equal(emb.weight.shape[0], weights.shape[0]);
                Assert.Equal(emb.weight.shape[1], weights.shape[1]);
            }
        }

        [Fact]
        public void TestEmbeddingBagFromPretrained()
        {
            var ones = torch.ones(new long[] { 16 }, torch.int32);
            var weights = torch.randn(new long[] { 1000, 12 });

            using (var emb = EmbeddingBag_from_pretrained(weights)) {
                Assert.Equal(emb.weight!.shape.Length, weights.shape.Length);
                Assert.Equal(emb.weight!.shape[0], weights.shape[0]);
                Assert.Equal(emb.weight!.shape[1], weights.shape[1]);
            }
        }

        [Fact]
        public void TestOneHotEncoding1()
        {
            var ones = torch.tensor(new long[] { 1, 2, 0, 0, 3, 4, 2, 2 });
            var env = one_hot(ones, 5);
            var values = env.data<long>().ToArray();
            Assert.Equal(ones.shape[0], env.shape[0]);
            Assert.Equal(5, env.shape[1]);
        }

        [Fact]
        public void TestOneHotEncoding2()
        {
            var ones = torch.tensor(new long[] { 1, 2, 0, 5, 3, 4, 2, 2 });
            var env = one_hot(ones);
            var values = env.data<long>().ToArray();
            Assert.Equal(ones.shape[0], env.shape[0]);
            Assert.Equal(6, env.shape[1]);
        }

        [Fact]
        public void TestTransformer()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var transformer_model = Transformer(d_model: 64, nhead: 2, num_encoder_layers: 2, dim_feedforward: 128)) {
                var src = torch.rand(new long[] { 10, 16, 64 });
                var tgt = torch.rand(new long[] { 20, 16, 64 });
                var output = transformer_model.forward(src, tgt);
                Assert.Equal(tgt.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerWithMasks()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var transformer_model = Transformer(d_model: 64, nhead: 2, num_encoder_layers: 2, dim_feedforward: 128)) {
                var src = torch.rand(new long[] { 10, 16, 64 });
                var tgt = torch.rand(new long[] { 20, 16, 64 });
                var src_mask = torch.rand(new long[] { 10, 10 });
                var tgt_mask = torch.rand(new long[] { 20, 20 });
                var output = transformer_model.forward(src, tgt, src_mask: src_mask, tgt_mask: tgt_mask);
                Assert.Equal(tgt.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerEncoderLayer()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var encoder_layer = TransformerEncoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128)) {
                var src = torch.rand(new long[] { 10, 16, 64 });
                var output = encoder_layer.forward(src);
                Assert.Equal(src.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerEncoderLayerWithMasks()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var encoder_layer = TransformerEncoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128)) {
                var src = torch.rand(new long[] { 10, 16, 64 });
                var src_mask = torch.rand(new long[] { 10, 10 });
                var output = encoder_layer.forward(src, src_mask: src_mask);
                Assert.Equal(src.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerEncoder()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var encoder_layer = TransformerEncoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128))
            using (var encoder = TransformerEncoder(encoder_layer, 1)) {
                var src = torch.rand(new long[] { 10, 16, 64 });
                var output = encoder.forward(src);
                Assert.Equal(src.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerEncoderWithMasks()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var encoder_layer = TransformerEncoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128))
            using (var encoder = TransformerEncoder(encoder_layer, 1)) {
                var src = torch.rand(new long[] { 10, 16, 64 });
                var src_mask = torch.rand(new long[] { 10, 10 });
                var output = encoder.forward(src, src_mask: src_mask);
                Assert.Equal(src.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerDecoderLayer()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var decoder_layer = TransformerDecoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128)) {
                var tgt = torch.rand(new long[] { 20, 16, 64 });
                var memory = torch.rand(new long[] { 10, 16, 64 });
                var output = decoder_layer.forward(tgt, memory);
                Assert.Equal(tgt.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerDecoderLayerWithMasks()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var decoder_layer = TransformerDecoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128)) {
                var tgt = torch.rand(new long[] { 20, 16, 64 });
                var memory = torch.rand(new long[] { 10, 16, 64 });
                var tgt_mask = torch.rand(new long[] { 20, 20 });
                var output = decoder_layer.forward(tgt, memory, tgt_mask: tgt_mask);
                Assert.Equal(tgt.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerDecoder()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var decoder_layer = TransformerDecoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128))
            using (var decoder = TransformerDecoder(decoder_layer, 1)) {
                var tgt = torch.rand(new long[] { 20, 16, 64 });
                var memory = torch.rand(new long[] { 10, 16, 64 });
                var output = decoder.forward(tgt, memory);
                Assert.Equal(tgt.shape, output.shape);
            }
        }

        [Fact]
        public void TestTransformerDecoderWithMasks()
        {
            // Transformers are very memory-intensive. It is useful to avoid using the defaults here.
            using (var decoder_layer = TransformerDecoderLayer(d_model: 64, nhead: 2, dim_feedforward: 128))
            using (var decoder = TransformerDecoder(decoder_layer, 1)) {
                var tgt = torch.rand(new long[] { 20, 16, 64 });
                var memory = torch.rand(new long[] { 10, 16, 64 });
                var tgt_mask = torch.rand(new long[] { 20, 20 });
                var output = decoder.forward(tgt, memory, tgt_mask: tgt_mask);
                Assert.Equal(tgt.shape, output.shape);
            }
        }

        [Fact]
        public void TestMultiheadAttention()
        {
            var num_heads = 1;
            var qembed_dim = 2L; // Must be divisible by the number of heads
            var kembed_dim = 2L;
            var vembed_dim = 2L;
            var src_seq_len = 3L;
            var tgt_seq_len = 3L;
            var batch_size = 1L;
            var dropout = 0.0; //  # This is not supported
            var bias = false;
            var add_bias_kv = false;
            var add_zero_attn = false;

            var q_data = new float[]{  0.3367f, 0.1288f,
                                       0.2345f,  0.2303f,
                                       -1.1229f, -0.1863f};

            var k_data = new float[] { 2.2082f, -0.6380f,
                                       0.4617f,  0.2674f,
                                       0.5349f,  0.8094f};

            var v_data = new float[] {1.1103f, -1.6898f,
                                      -0.9890f,  0.9580f,
                                       1.3221f,  0.8172f};

            var attn_data = new float[] {0.342628956f, 0.3370244f, 0.3203467f,
                                         0.336390018f, 0.333694249f, 0.329915673f,
                                         0.296296269f, 0.314919323f, 0.388784438f};



            using (var mha = MultiheadAttention(qembed_dim, num_heads, dropout: dropout, bias: bias, add_bias_kv: add_bias_kv, add_zero_attn: add_zero_attn, kdim: kembed_dim, vdim: vembed_dim))
            using (var Q = torch.tensor(q_data, tgt_seq_len, batch_size, qembed_dim))
            using (var K = torch.tensor(k_data, src_seq_len, batch_size, kembed_dim))
            using (var V = torch.tensor(v_data, src_seq_len, batch_size, vembed_dim))
            using (var Attn = torch.tensor(attn_data, batch_size, src_seq_len, src_seq_len)) {

                mha.eval();
                Assert.False(mha.training);

                var (att_out, att_wts) = mha.forward(Q, K, V);
                var t = att_wts.allclose(Attn, rtol: 0.5, atol: 0.5);
                Assert.True(t);
            }
        }

        #endregion

        #region Dropout
        [Fact]
        public void TestDropout()
        {
            var drop = Dropout(0.75);
            var data = torch.rand(new long[] { 12, 23, 24 });
            var output = drop.forward(data);
            Assert.Equal(data.shape, output.shape);

            var dataVal = data.data<float>().ToArray();
            var outVal = output.data<float>().ToArray();
            Assert.NotEqual(outVal, dataVal);
        }

        [Fact]
        public void TestDropoutInPlace()
        {
            var drop = Dropout(0.75, inplace: true);
            var data = torch.rand(new long[] { 12, 23, 24 });
            var output = drop.forward(data);
            Assert.Equal(data.shape, output.shape);

            var dataVal = data.data<float>().ToArray();
            var outVal = output.data<float>().ToArray();
            Assert.Equal(outVal, dataVal);
        }

        [Fact]
        public void TestDropout1d()
        {
            var drop = Dropout1d(0.75);
            var data = torch.rand(new long[] { 12, 23, 24 });
            var output = drop.forward(data);
            Assert.Equal(data.shape, output.shape);

            var dataVal = data.data<float>().ToArray();
            var outVal = output.data<float>().ToArray();
            Assert.NotEqual(dataVal, outVal);
        }

        [Fact]
        public void TestDropout1dInPlace()
        {
            var drop = Dropout1d(0.75, inplace: true);
            var data = torch.rand(new long[] { 12, 23, 24 });
            var output = drop.forward(data);
            Assert.Equal(data.shape, output.shape);

            var dataVal = data.data<float>().ToArray();
            var outVal = output.data<float>().ToArray();
            Assert.Equal(outVal, dataVal);
        }

        [Fact]
        public void TestDropout2d()
        {
            var drop = Dropout2d(0.75);
            var data = torch.rand(new long[] { 12, 23, 24, 5 });
            var output = drop.forward(data);
            Assert.Equal(data.shape, output.shape);

            var dataVal = data.data<float>().ToArray();
            var outVal = output.data<float>().ToArray();
            Assert.NotEqual(outVal, dataVal);
        }

        [Fact]
        public void TestDropout2dInPlace()
        {
            var drop = Dropout2d(0.75, inplace: true);
            var data = torch.rand(new long[] { 12, 23, 24, 5 });
            var output = drop.forward(data);
            Assert.Equal(data.shape, output.shape);

            var dataVal = data.data<float>().ToArray();
            var outVal = output.data<float>().ToArray();
            Assert.Equal(outVal, dataVal);
        }

        [Fact]
        public void TestDropout3d()
        {
            var drop = Dropout3d(0.75);
            var data = torch.rand(new long[] { 12, 23, 24, 5, 6 });
            var output = drop.forward(data);
            Assert.Equal(data.shape, output.shape);

            var dataVal = data.data<float>().ToArray();
            var outVal = output.data<float>().ToArray();
            Assert.NotEqual(outVal, dataVal);
        }

        [Fact]
        public void TestDropout3dInPlace()
        {
            var drop = Dropout3d(0.75, inplace: true);
            var data = torch.rand(new long[] { 12, 23, 24, 5, 6 });
            var output = drop.forward(data);
            Assert.Equal(data.shape, output.shape);

            var dataVal = data.data<float>().ToArray();
            var outVal = output.data<float>().ToArray();
            Assert.Equal(outVal, dataVal);
        }

        [Fact]
        public void TestAlphaDropout()
        {
            var drop = AlphaDropout(0.75);
            var data = torch.rand(new long[] { 12, 23, 24 });
            var output = drop.forward(data);
            Assert.Equal(data.shape, output.shape);

            var dataVal = data.data<float>().ToArray();
            var outVal = output.data<float>().ToArray();
            Assert.NotEqual(outVal, dataVal);
        }
        #endregion

#if DEBUG
        [Fact(Skip = "Not working on Mac and Ubuntu (note: may now be working, we need to recheck)")]
        public void TestErrorHandling()
        {
            using (Tensor input = torch.tensor(new float[] { 0.5f, 1.5f }))
            using (Tensor target = torch.tensor(new float[] { 1f, 2f, 3f })) {
                Assert.Throws<ExternalException>(() => torch.nn.PoissonNLLLoss().forward(input, target));
            }
        }
#endif

        [Fact]
        public void TestFlatten()
        {
            var data = torch.rand(new long[] { 32, 3, 4, 5, 6 });

            using (var flat = Flatten()) {
                var output = flat.forward(data);
                Assert.Equal(new long[] { 32, 360 }, output.shape);
            }

            using (var flat = Flatten(startDim: 2)) {
                var output = flat.forward(data);
                Assert.Equal(new long[] { 32, 3, 120 }, output.shape);
            }

            using (var flat = Flatten(startDim: 0)) {
                var output = flat.forward(data);
                Assert.Equal(new long[] { 32 * 360 }, output.shape);
            }
        }

        [Fact]
        public void TestUnflatten()
        {
            var input = torch.rand(new long[] { 2, 50 });

            var uf = Unflatten(1, new long[] { 2, 5, 5 });
            var res = uf.forward(input);

            Assert.Equal(4, res.Dimensions);
            Assert.Equal(new long[] { 2, 2, 5, 5 }, res.shape);
        }

        [Fact]
        public void TestCosineSimilarity()
        {
            using (Tensor input1 = torch.rand(new long[] { 5, 12 }))
            using (Tensor input2 = torch.randint(12, new long[] { 5, 12 }, torch.int64))

            using (var module = CosineSimilarity()) {
                var output = module.forward(input1, input2);
                Assert.Equal(input1.shape[0], output.shape[0]);
            }
        }

        [Fact]
        public void TestPairwiseDistance()
        {
            using (Tensor input1 = torch.rand(new long[] { 5, 12 }))
            using (Tensor input2 = torch.randint(12, new long[] { 5, 12 }, torch.int64))

            using (var module = PairwiseDistance(keep_dim: true)) {
                var output = module.forward(input1, input2);
                Assert.Equal(input1.shape[0], output.shape[0]);
                Assert.Equal(1, output.shape[1]);
            }
        }

        [Fact]
        public void TestZeroPad2d()
        {
            var data = torch.rand(new long[] { 32, 3, 4, 4 });

            using (var pad = ZeroPad2d(3)) {
                var output = pad.forward(data);
                Assert.Equal(new long[] { 32, 3, 10, 10 }, output.shape);
                Assert.Equal(0.0, output[0, 0, 0, 0].ToDouble());
            }
        }

        [Fact]
        public void TestReflectionPad1d()
        {
            var data = torch.rand(new long[] { 32, 3, 4 });

            using (var pad = ReflectionPad1d(3)) {
                var output = pad.forward(data);
                Assert.Equal(new long[] { 32, 3, 10 }, output.shape);
                var values = output.data<float>().ToArray();
                Assert.Equal(values[6], values[0]);
                Assert.Equal(values[5], values[1]);
                Assert.Equal(values[4], values[2]);
                Assert.Equal(values[5], values[7]);
                Assert.Equal(values[4], values[8]);
                Assert.Equal(values[3], values[9]);
            }
        }

        [Fact]
        public void TestReflectionPad2d()
        {
            var data = torch.rand(new long[] { 32, 3, 4, 4 });

            using (var pad = ReflectionPad2d(3)) {
                var output = pad.forward(data);
                Assert.Equal(new long[] { 32, 3, 10, 10 }, output.shape);
                var values = output.data<float>().ToArray();
                Assert.Equal(values[6], values[0]);
                Assert.Equal(values[5], values[1]);
                Assert.Equal(values[4], values[2]);
            }
        }

        [Fact]
        public void TestReflectionPad3d()
        {
            var data = torch.rand(new long[] { 32, 3, 4, 4, 4 });

            using (var pad = ReflectionPad3d(3)) {
                var output = pad.forward(data);
                Assert.Equal(new long[] { 32, 3, 10, 10, 10 }, output.shape);
                var values = output.data<float>().ToArray();
                Assert.Equal(values[6], values[0]);
                Assert.Equal(values[5], values[1]);
                Assert.Equal(values[4], values[2]);
            }
        }

        [Fact]
        public void TestReplicationPad1d()
        {
            var data = torch.rand(new long[] { 32, 3, 4 });

            using (var pad = ReplicationPad1d(3)) {
                var output = pad.forward(data);
                Assert.Equal(new long[] { 32, 3, 10 }, output.shape);
                var values = output.data<float>().ToArray();
                Assert.Equal(values[3], values[0]);
                Assert.Equal(values[3], values[1]);
                Assert.Equal(values[3], values[3]);
                Assert.Equal(values[6], values[7]);
                Assert.Equal(values[6], values[8]);
                Assert.Equal(values[6], values[9]);
            }
        }

        [Fact]
        public void TestReplicationPad2d()
        {
            var data = torch.rand(new long[] { 32, 3, 4, 4 });

            using (var pad = ReplicationPad2d(3)) {
                var output = pad.forward(data);
                Assert.Equal(new long[] { 32, 3, 10, 10 }, output.shape);
                var values = output.data<float>().ToArray();
                Assert.Equal(values[3], values[0]);
                Assert.Equal(values[3], values[1]);
                Assert.Equal(values[3], values[3]);
            }
        }

        [Fact]
        public void TestReplicationPad3d()
        {
            var data = torch.rand(new long[] { 32, 3, 4, 4, 4 });

            using (var pad = ReplicationPad3d(3)) {
                var output = pad.forward(data);
                Assert.Equal(new long[] { 32, 3, 10, 10, 10 }, output.shape);
                var values = output.data<float>().ToArray();
                Assert.Equal(values[3], values[0]);
                Assert.Equal(values[3], values[1]);
                Assert.Equal(values[3], values[3]);
            }
        }

        [Fact]
        public void TestConstantPad1d()
        {
            var data = torch.rand(new long[] { 32, 3, 4 }, torch.float64);

            using (var pad = ConstantPad1d(3, Math.PI)) {
                var output = pad.forward(data);
                Assert.Equal(new long[] { 32, 3, 10 }, output.shape);
                Assert.Equal(Math.PI, output[0, 0, 0].ToDouble());
            }
        }

        [Fact]
        public void TestConstantPad2d()
        {
            var data = torch.rand(new long[] { 32, 3, 4, 4 }, torch.float64);

            using (var pad = ConstantPad2d(3, Math.PI)) {
                var output = pad.forward(data);
                Assert.Equal(new long[] { 32, 3, 10, 10 }, output.shape);
                Assert.Equal(Math.PI, output[0, 0, 0, 0].ToDouble());
            }
        }

        [Fact]
        public void TestConstantPad3d()
        {
            var data = torch.rand(new long[] { 32, 3, 4, 4, 4 }, torch.float64);

            using (var pad = ConstantPad3d(3, Math.PI)) {
                var output = pad.forward(data);
                Assert.Equal(new long[] { 32, 3, 10, 10, 10 }, output.shape);
                Assert.Equal(Math.PI, output[0, 0, 0, 0, 0].ToDouble());
            }
        }

        [Fact]
        public void TestRNN1()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 1, 3, 20 }))
            using (var rnn = RNN(10, 20)) {
                var (output, hN) = rnn.forward(input, h0);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }

        }

        [Fact]
        public void TestRNN2()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 2, 3, 20 }))
            using (var rnn = RNN(10, 20, 2)) {
                var (output, hN) = rnn.forward(input, h0);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }

        }

        [Fact]
        public void TestRNN3()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 2, 3, 20 }))
            using (var rnn = RNN(10, 20, 2)) {
                var (output, hN) = rnn.forward(input);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }

        }

        [Fact]
        public void TestRNN4()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 1, 3, 20 }))
            using (var rnn = RNN(10, 20)) {
                rnn.flatten_parameters();
                var (output, hN) = rnn.forward(input, h0);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }
        }

        [Fact]
        public void TestRNN5()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   lengths = torch.tensor(new long[] { 5, 5, 5 }))
            using (var rnn = RNN(10, 20, 2)) {
                rnn.flatten_parameters();
                var (output, hN) = rnn.forward(input);
                var packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths);
                var (packed_output, packed_hN) = rnn.forward(packed_input);
                float mse;
                mse = torch.mean(torch.square(hN - packed_hN)).item<float>();
                Assert.InRange(mse, 0, 0.1f);
                var (unpacked_output, unpacked_output_lengths) = torch.nn.utils.rnn.pad_packed_sequence(packed_output);
                mse = torch.mean(torch.square(output - unpacked_output)).item<float>();
                Assert.InRange(mse, 0, 0.1f);
            }
        }

        [Fact]
        public void TestRNNCell1()
        {
            var seq = 5;
            using (Tensor input = torch.randn(new long[] { seq, 3, 10 }),
                   h0 = torch.randn(new long[] { 3, 20 }))
            using (var rnn = RNNCell(10, 20)) {
                var hN = rnn.forward(input[0], h0);
                Assert.Equal(h0.shape, hN.shape);
                for (int i = 1; i < seq; ++i) {
                    hN = rnn.forward(input[i], hN);
                    Assert.Equal(h0.shape, hN.shape);
                }
            }
        }

        [Fact]
        public void TestRNNCell2()
        {
            using (Tensor input = torch.randn(new long[] { 3, 10 }),
                   h0 = torch.randn(new long[] { 3, 20 }))
            using (var rnn = RNNCell(10, 20, NonLinearities.ReLU)) {
                var hN = rnn.forward(input, h0);
                Assert.Equal(h0.shape, hN.shape);
            }
        }

        [Fact]
        public void TestLRNNEditWeightsAndBias()
        {
            var lin = RNN(10, 20, 2, NonLinearities.ReLU);
            var bias = torch.randn(new long[] { 100 });
            var weights = torch.randn(new long[] { 100, 1000 });

            var w1 = lin.get_weight_ih(1);

            Assert.Equal(2, w1!.shape.Length);
            Assert.Equal(20, w1!.shape[0]);
            Assert.Equal(20, w1!.shape[1]);

            var w2 = lin.parameters().ToArray()[0];

            Assert.Equal(2, w2.shape.Length);
            Assert.Equal(20, w2.shape[0]);
            Assert.Equal(10, w2.shape[1]);
        }

        [Fact]
        public void TestLRNNCellEditWeightsAndBias()
        {
            var lin = RNNCell(10, 20, NonLinearities.ReLU);
            var bias = torch.randn(new long[] { 100 });
            var weights = torch.randn(new long[] { 100, 1000 });

            lin.bias_ih = bias.clone().AsParameter();
            lin.weight_ih = weights.clone().AsParameter();

            var w1 = lin.weight_ih;
            var b1 = lin.bias_ih;

            Assert.Equal(w1.shape.Length, weights.shape.Length);
            Assert.Equal(w1.shape[0], weights.shape[0]);
            Assert.Equal(w1.shape[1], weights.shape[1]);

            for (int i = 0; i < 100; i++) {
                Assert.Equal(b1.data<float>()[i], bias.data<float>()[i]);
            }

            var w2 = lin.parameters().ToArray()[0];
            var b2 = lin.parameters().ToArray()[2];

            Assert.Equal(weights.shape.Length, w2.shape.Length);
            Assert.Equal(weights.shape[0], w2.shape[0]);
            Assert.Equal(weights.shape[1], w2.shape[1]);

            for (int i = 0; i < 100; i++) {
                Assert.Equal(b2.data<float>()[i], bias.data<float>()[i]);
            }
        }

        [Fact]
        public void TestGRU1()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 1, 3, 20 }))
            using (var gru = GRU(10, 20)) {
                var (output, hN) = gru.forward(input, h0);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }

        }

        [Fact]
        public void TestGRU2()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 2, 3, 20 }))
            using (var gru = GRU(10, 20, 2)) {
                var (output, hN) = gru.forward(input, h0);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }

        }

        [Fact]
        public void TestGRU3()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 2, 3, 20 }))
            using (var gru = GRU(10, 20, 2)) {
                var (output, hN) = gru.forward(input);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }

        }

        [Fact]
        public void TestGRU4()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 2, 3, 20 }))
            using (var gru = GRU(10, 20, 2)) {
                gru.flatten_parameters();
                var (output, hN) = gru.forward(input);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }

        }

        [Fact]
        public void TestGRU5()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }).to(torch.float64),
                   h0 = torch.randn(new long[] { 2, 3, 20 }).to(torch.float64))
            using (var gru = GRU(10, 20, 2)) {
                gru.to(torch.float64);
                gru.flatten_parameters();
                var (output, hN) = gru.forward(input);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }
        }

        [Fact]
        public void TestGRU6()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   lengths = torch.tensor(new long[] { 5, 5, 5 }))
            using (var rnn = GRU(10, 20, 2)) {
                rnn.flatten_parameters();
                var (output, hN) = rnn.forward(input);
                var packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths);
                var (packed_output, packed_hN) = rnn.forward(packed_input);
                float mse;
                mse = torch.mean(torch.square(hN - packed_hN)).item<float>();
                Assert.InRange(mse, 0, 0.1f);
                var (unpacked_output, unpacked_output_lengths) = torch.nn.utils.rnn.pad_packed_sequence(packed_output);
                mse = torch.mean(torch.square(output - unpacked_output)).item<float>();
                Assert.InRange(mse, 0, 0.1f);
            }
        }

        [Fact]
        public void TestGRUCell1()
        {
            var seq = 5;
            using (Tensor input = torch.randn(new long[] { seq, 3, 10 }),
                   h0 = torch.randn(new long[] { 3, 20 }))
            using (var rnn = GRUCell(10, 20)) {
                var hN = rnn.forward(input[0], h0);
                Assert.Equal(h0.shape, hN.shape);
                for (int i = 1; i < seq; ++i) {
                    hN = rnn.forward(input[i], hN);
                    Assert.Equal(h0.shape, hN.shape);
                }
            }
        }

        [Fact]
        public void TestGRUCell2()
        {
            using (Tensor input = torch.randn(new long[] { 3, 10 }),
                   h0 = torch.randn(new long[] { 3, 20 }))
            using (var rnn = GRUCell(10, 20, bias: false)) {
                var hN = rnn.forward(input, h0);
                Assert.Equal(h0.shape, hN.shape);
            }
        }

        [Fact]
        public void TestLSTM1()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 1, 3, 20 }),
                   c0 = torch.randn(new long[] { 1, 3, 20 }))
            using (var rnn = LSTM(10, 20)) {
                var (output, hN, cN) = rnn.forward(input, (h0, c0));
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(c0.shape, cN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }

        }

        [Fact]
        public void TestLSTM2()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 2, 3, 20 }),
                   c0 = torch.randn(new long[] { 2, 3, 20 }))
            using (var rnn = LSTM(10, 20, 2)) {
                var (output, hN, cN) = rnn.forward(input, (h0, c0));
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(c0.shape, cN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }

        }

        [Fact]
        public void TestLSTM3()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 2, 3, 20 }),
                   c0 = torch.randn(new long[] { 2, 3, 20 }))
            using (var rnn = LSTM(10, 20, 2)) {
                var (output, hN, cN) = rnn.forward(input);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(c0.shape, cN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }

        }

        [Fact]
        public void TestLSTM4()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   h0 = torch.randn(new long[] { 2, 3, 20 }),
                   c0 = torch.randn(new long[] { 2, 3, 20 }))
            using (var rnn = LSTM(10, 20, 2)) {
                rnn.flatten_parameters();
                var (output, hN, cN) = rnn.forward(input);
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(c0.shape, cN.shape);
                Assert.Equal(new long[] { input.shape[0], input.shape[1], 20 }, output.shape);
            }

        }

        [Fact]
        public void TestLSTM5()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   lengths = torch.tensor(new long[] { 5, 5, 5 }),
                   h0 = torch.randn(new long[] { 2, 3, 20 }),
                   c0 = torch.randn(new long[] { 2, 3, 20 }))
            using (var rnn = LSTM(10, 20, 2)) {
                rnn.flatten_parameters();
                var (output, hN, cN) = rnn.forward(input, (h0, c0));
                var packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths);
                var (packed_output, packed_hN, packed_cN) = rnn.forward(packed_input, (h0, c0));
                float mse;
                mse = torch.mean(torch.square(hN - packed_hN)).item<float>();
                Assert.InRange(mse, 0, 0.1f);
                mse = torch.mean(torch.square(cN - packed_cN)).item<float>();
                Assert.InRange(mse, 0, 0.1f);
                var (unpacked_output, unpacked_output_lengths) = torch.nn.utils.rnn.pad_packed_sequence(packed_output);
                mse = torch.mean(torch.square(output - unpacked_output)).item<float>();
                Assert.InRange(mse, 0, 0.1f);
            }
        }

        [Fact]
        public void TestLSTM6()
        {
            using (Tensor input = torch.randn(new long[] { 5, 3, 10 }),
                   lengths = torch.tensor(new long[] { 5, 5, 5 }))
            using (var rnn = LSTM(10, 20, 2)) {
                rnn.flatten_parameters();
                var (output, hN, cN) = rnn.forward(input);
                var packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths);
                var (packed_output, packed_hN, packed_cN) = rnn.forward(packed_input);
                float mse;
                mse = torch.mean(torch.square(hN - packed_hN)).item<float>();
                Assert.InRange(mse, 0, 0.1f);
                mse = torch.mean(torch.square(cN - packed_cN)).item<float>();
                Assert.InRange(mse, 0, 0.1f);
                var (unpacked_output, unpacked_output_lengths) = torch.nn.utils.rnn.pad_packed_sequence(packed_output);
                mse = torch.mean(torch.square(output - unpacked_output)).item<float>();
                Assert.InRange(mse, 0, 0.1f);
            }
        }

        [Fact]
        public void TestLSTMCell1()
        {
            var seq = 5;
            using (Tensor input = torch.randn(new long[] { seq, 3, 10 }),
                   h0 = torch.randn(new long[] { 3, 20 }),
                   c0 = torch.randn(new long[] { 3, 20 }))
            using (var rnn = LSTMCell(10, 20)) {
                var (hN, cN) = rnn.forward(input[0], (h0, c0));
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(c0.shape, cN.shape);
                for (int i = 1; i < seq; ++i) {
                    (hN, cN) = rnn.forward(input[i], (hN, cN));
                    Assert.Equal(h0.shape, hN.shape);
                    Assert.Equal(c0.shape, cN.shape);
                }
            }

        }

        [Fact]
        public void TestLSTMCell2()
        {
            using (Tensor input = torch.randn(new long[] { 3, 10 }),
                   h0 = torch.randn(new long[] { 3, 20 }),
                   c0 = torch.randn(new long[] { 3, 20 }))
            using (var rnn = LSTMCell(10, 20, bias: false)) {
                var (hN, cN) = rnn.forward(input, (h0, c0));
                Assert.Equal(h0.shape, hN.shape);
                Assert.Equal(c0.shape, cN.shape);
            }

        }

        [Fact]
        public void TestPixelShuffle()
        {
            using (Tensor input = torch.randn(new long[] { 8, 9, 4, 4 }))
            using (var layer = PixelShuffle(3)) {
                var res = layer.forward(input);
                Assert.Equal(new long[] { 8, 1, 12, 12 }, res.shape);
            }
        }

        [Fact]
        public void TestPixelUnshuffle()
        {
            using (Tensor input = torch.randn(new long[] { 8, 1, 12, 12 }))
            using (var layer = PixelUnshuffle(3)) {
                var res = layer.forward(input);
                Assert.Equal(new long[] { 8, 9, 4, 4 }, res.shape);
            }
        }

        [Fact]
        public void TestVisionPad()
        {
            using (Tensor p4d = torch.randn(new long[] { 3, 3, 4, 2 })) {
                using (var res = torchvision.transforms.functional.pad(p4d, new long[] { 1, 1 }, 0.0, PaddingModes.Constant)) {
                    Assert.Equal(new long[] { 3, 3, 6, 4 }, res.shape);
                }
                using (var res = torchvision.transforms.functional.pad(p4d, new long[] { 1, 1, 2, 2 }, 0.0, PaddingModes.Constant)) {
                    Assert.Equal(new long[] { 3, 3, 7, 5 }, res.shape);
                }
            }
        }

        [Fact]
        public void TestNNPad()
        {
            {
                var source = torch.arange(9).reshape(3, 3);
                var result = torch.nn.functional.pad(input: source, pad: new long[] { 1, 2 }, mode: PaddingModes.Constant, value: 0);
                Assert.Equal(new long[] { 3, 6 }, result.shape);

                var str = result.ToString(TensorStringStyle.Numpy, newLine: "\n");
                Assert.Equal("[[0 0 1 2 0 0]\n [0 3 4 5 0 0]\n [0 6 7 8 0 0]]", str);
            }
            using (Tensor p4d = torch.randn(new long[] { 3, 3, 4, 2 })) {
                using (var res = pad(p4d, new long[] { 1, 1, 2, 3 }, PaddingModes.Constant, 0.0)) {
                    Assert.Equal(new long[] { 3, 3, 9, 4 }, res.shape);
                }
            }
        }


        [Fact]
        public void TestInterpolateDefaults()
        {
            using (Tensor input = torch.arange(1, 5, float32).view(1, 1, 2, 2))
            using (var res = interpolate(input, scale_factor: new double[] { 2, 2 })) {
                Assert.Equal(new long[] { 1, 1, 4, 4 }, res.shape);
            }
        }

        [Fact]
        public void TestInterpolateNearest()
        {
            using (Tensor input = torch.arange(1, 5, float32).view(1, 1, 2, 2))
            using (var res = interpolate(input, scale_factor: new double[] { 2, 2 }, mode: InterpolationMode.Nearest)) {
                Assert.Equal(new long[] { 1, 1, 4, 4 }, res.shape);
            }
        }

        [Fact]
        public void TestInterpolateBilinear2D()
        {
            using (Tensor input = torch.arange(1, 5, float32).view(1, 1, 2, 2))
            using (var res = interpolate(input, scale_factor: new double[] { 2, 2 }, mode: InterpolationMode.Bilinear)) {
                Assert.Equal(new long[] { 1, 1, 4, 4 }, res.shape);
            }
        }


        [Fact]
        public void TestInterpolateArea()
        {
            using (Tensor input = torch.arange(1, 5, float32).view(1, 1, 2, 2))
            using (var res = interpolate(input, scale_factor: new double[] { 2, 2 }, mode: InterpolationMode.Area)) {
                Assert.Equal(new long[] { 1, 1, 4, 4 }, res.shape);
            }
        }

        [Fact]
        public void TestInterpolateTrilinear()
        {
            using (Tensor input = torch.arange(1, 9, 1, float32).view(1, 1, 2, 2, 2))
            using (var res = interpolate(input, scale_factor: new double[] { 2, 2, 2 }, mode: InterpolationMode.Trilinear)) {
                Assert.Equal(new long[] { 1, 1, 4, 4, 4 }, res.shape);
            }
        }

        [Fact]
        public void TestUpsampleNearest()
        {
            using (Tensor input = torch.arange(1, 5, float32).view(1, 1, 2, 2))
            using (var layer = Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest)) {
                var res = layer.forward(input);
                Assert.Equal(new long[] { 1, 1, 4, 4 }, res.shape);
            }
        }

        [Fact]
        public void TestUpsampleLinear()
        {
            using (Tensor input = torch.arange(1, 5, float32).view(1, 1, 4))
            using (var layer = Upsample(scale_factor: new double[] { 2 }, mode: UpsampleMode.Linear)) {
                var res = layer.forward(input);
                Assert.Equal(new long[] { 1, 1, 8 }, res.shape);
            }
        }

        [Fact]
        public void TestUpsampleBilinear()
        {
            using (Tensor input = torch.arange(1, 5, float32).view(1, 1, 2, 2))
            using (var layer = Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Bilinear)) {
                var res = layer.forward(input);
                Assert.Equal(new long[] { 1, 1, 4, 4 }, res.shape);
            }
        }

        [Fact]
        public void TestUpsampleBilinearAC()
        {
            using (Tensor input = torch.arange(1, 5, float32).view(1, 1, 2, 2))
            using (var layer = Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Bilinear, alignCorners: true)) {
                var res = layer.forward(input);
                Assert.Equal(new long[] { 1, 1, 4, 4 }, res.shape);
            }
        }

        [Fact]
        public void TestUpsampleBicubic()
        {
            using (Tensor input = torch.arange(1, 5, float32).view(1, 1, 2, 2))
            using (var layer = Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Bicubic)) {
                var res = layer.forward(input);
                Assert.Equal(new long[] { 1, 1, 4, 4 }, res.shape);
            }
        }

        [Fact]
        public void TestUpsampleBicubicAC()
        {
            using (Tensor input = torch.arange(1, 5, float32).view(1, 1, 2, 2))
            using (var layer = Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Bicubic, alignCorners: true)) {
                var res = layer.forward(input);
                Assert.Equal(new long[] { 1, 1, 4, 4 }, res.shape);
            }
        }

        [Fact]
        public void TestUpsampleTrilinear()
        {
            using (Tensor input = torch.arange(1, 9, float32).view(1, 1, 2, 2, 2))
            using (var layer = Upsample(scale_factor: new double[] { 2, 2, 2 }, mode: UpsampleMode.Trilinear)) {
                var res = layer.forward(input);
                Assert.Equal(new long[] { 1, 1, 4, 4, 4 }, res.shape);
            }
        }

        [Fact]
        public void TestUpsampleTrilinearAC()
        {
            using (Tensor input = torch.arange(1, 9, float32).view(1, 1, 2, 2, 2))
            using (var layer = Upsample(scale_factor: new double[] { 2, 2, 2 }, mode: UpsampleMode.Trilinear, alignCorners: true)) {
                var res = layer.forward(input);
                Assert.Equal(new long[] { 1, 1, 4, 4, 4 }, res.shape);
            }
        }
    }
}