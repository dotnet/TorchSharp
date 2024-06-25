using System.Linq;
using System.Collections.Generic;
using Xunit;

using static TorchSharp.torch;
using System;
using System.Threading;

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestAutogradFunction
    {

        private void TestCustomLinearFunction(Device device, bool requires_grad)
        {
            var x = torch.randn(new long[] { 2, 3 }, device: device).requires_grad_(requires_grad);
            var weight = torch.randn(new long[] { 4, 3 }, device: device).requires_grad_(requires_grad);
            var y = LinearFunction.apply(x, weight);

            if (!requires_grad) return;

            y.sum().backward();

            Assert.NotNull(x.grad);
            Assert.NotNull(weight.grad);
        }


        private void TestCustomTwoInputLinearFunction(Device device, bool requires_grad)
        {
            var x1 = torch.randn(new long[] { 2, 3 }, device: device).requires_grad_(requires_grad);
            var x2 = torch.randn(new long[] { 5, 3 }, device: device).requires_grad_(requires_grad);
            var weight = torch.randn(new long[] { 4, 3 }, device: device).requires_grad_(requires_grad);
            var y = TwoInputLinearFunction.apply(x1, x2, weight);

            if (!requires_grad) return;

            (y[0].sum() + y[1].sum()).backward();

            Assert.NotNull(x1.grad);
            Assert.NotNull(x2.grad);
            Assert.NotNull(weight.grad);
        }

        private void TestCustomTwoInputOneGradientLinearFunction(Device device, bool requires_grad)
        {
            var x1 = torch.randn(new long[] { 2, 3 }, device: device).requires_grad_(requires_grad);
            var x2 = torch.randn(new long[] { 5, 3 }, device: device).requires_grad_(requires_grad);
            var weight = torch.randn(new long[] { 4, 3 }, device: device).requires_grad_(requires_grad);
            var y = TwoInputOneGradientLinearFunction.apply(x1, x2, weight);

            if (!requires_grad) return;

            (y[0].sum() + y[1].sum()).backward();

            Assert.NotNull(x1.grad);
            Assert.NotNull(x2.grad);
            Assert.Null(weight.grad);
        }

        private float TrainXOR(Device device)
        {
            Generator gen = new torch.Generator();
            var weight1 = torch.nn.Parameter(torch.randn(new long[] { 2, 2 }, generator: gen).to(device));
            var weight2 = torch.nn.Parameter(torch.randn(new long[] { 1, 2 }, generator: gen).to(device));
            var optim = torch.optim.SGD(new[] { weight1, weight2 }, 0.01);

            float lastLoss = 0;
            for (int epoch = 0; epoch < 5000; epoch++) {
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        using var dd = torch.NewDisposeScope();
                        optim.zero_grad();

                        var input = torch.tensor(new float[] { i, j }, device: device).unsqueeze(0);
                        var output = LinearFunction.apply(input, weight1);
                        output = LinearFunction.apply(torch.nn.functional.tanh(input), weight2);
                        var loss = (output - (i ^ j)).pow(2);
                        loss.backward();
                        optim.step();
                        lastLoss = loss.item<float>();
                    }
                }
            }

            return lastLoss;
        }


        [Fact]
        public void TestCustomLinearFunction_CPU()
        {
            TestCustomLinearFunction(torch.CPU, true);
        }

        [Fact]
        public void TestCustomLinearFunction_CPU_NoRequiresGrad()
        {
            TestCustomLinearFunction(torch.CPU, false);
        }

        [Fact]
        public void TestCustomLinearFunction_CUDA()
        {
            if (torch.cuda.is_available()) {
                TestCustomLinearFunction(torch.CUDA, true);
            }
        }

        [Fact]
        public void TestCustomLinearFunction_CUDA_NoRequiresGrad()
        {
            if (torch.cuda.is_available()) {
                TestCustomLinearFunction(torch.CUDA, false);
            }
        }

        [Fact]
        public void TestCustomTwoInputLinearFunction_CPU()
        {
            TestCustomTwoInputLinearFunction(torch.CPU, true);
        }

        [Fact]
        public void TestCustomTwoInputLinearFunction_CPU_NoRequiresGrad()
        {
            TestCustomTwoInputLinearFunction(torch.CPU, false);
        }

        [Fact]
        public void TestCustomTwoInputLinearFunction_CUDA()
        {
            if (torch.cuda.is_available()) {
                TestCustomTwoInputLinearFunction(torch.CUDA, true);
            }
        }

        [Fact]
        public void TestCustomTwoInputLinearFunction_CUDA_NoRequiresGrad()
        {
            if (torch.cuda.is_available()) {
                TestCustomTwoInputLinearFunction(torch.CUDA, false);
            }
        }

        [Fact]
        public void TestCustomTwoInputOneGradientLinearFunction_CPU()
        {
            TestCustomTwoInputOneGradientLinearFunction(torch.CPU, true);
        }

        [Fact]
        public void TestCustomTwoInputOneGradientLinearFunction_CUDA()
        {
            if (torch.cuda.is_available()) {
                TestCustomTwoInputOneGradientLinearFunction(torch.CUDA, true);
            }
        }

        [Fact]
        public void TestCustomLinearXORLearn_CPU()
        {
            float loss = TrainXOR(torch.CPU);
            LossIsClose(0.4513f, loss);
        }

        [Fact]
        public void TestCustomLinearXORLearn_CUDA()
        {
            if (torch.cuda.is_available()) {
                float loss = TrainXOR(torch.CUDA);
                LossIsClose(0.4513f, loss);
            }
        }

        [Fact]
        public void TestCustomLinearXORLearn_Parallel_CPU()
        {
            Enumerable.Range(0, 20).AsParallel().ForAll(i => {
                float loss = TrainXOR(torch.CPU);
                LossIsClose(0.4513f, loss);
            });
        }

        [Fact]
        public void TestCustomLinearFunction_Parallel_CPU_NoRequiresGrad()
        {
            Enumerable.Range(0, 50).AsParallel().ForAll(i => {
                TestCustomLinearFunction(torch.CPU, false);
            });
        }

        [Fact]
        private void TestCustomLinearFunctionWithGC()
        {
            var x = torch.randn(new long[] { 2, 3 }).requires_grad_();
            var weight = torch.randn(new long[] { 4, 3 }).requires_grad_();
            var y = LinearFunction.apply(x, weight);

            // Try to force objects to be cleaned up by giving them a timer + explicitly calling the GC
            Thread.Sleep(2000);
            GC.Collect();

            y.sum().backward();

            Assert.NotNull(x.grad);
            Assert.NotNull(weight.grad);
        }

        [Fact]
        private void TestBackwardWithPartialGradInput()
        {
            var x = torch.randn(new long[] { 2, 3 }).requires_grad_();
            var y = MulConstantFunction.apply(x, 2.0);
            y.sum().backward();

            Assert.NotNull(x.grad);
        }

        class MulConstantFunction : torch.autograd.SingleTensorFunction<MulConstantFunction>
        {
            public override string Name => nameof(MulConstantFunction);

            public override List<Tensor> backward(autograd.AutogradContext ctx, Tensor grad_output)
            {
                return new() { grad_output * (double)ctx.get_data("constant"), null };
            }

            public override Tensor forward(autograd.AutogradContext ctx, params object[] vars)
            {
                var tensor = (Tensor)vars[0];
                double constant = (double)vars[1];

                ctx.save_data("constant", constant);

                return tensor * constant;
            }
        }
        class TwoInputOneGradientLinearFunction : torch.autograd.MultiTensorFunction<TwoInputOneGradientLinearFunction>
        {
            public override string Name => nameof(TwoInputOneGradientLinearFunction);

            public override List<Tensor> backward(autograd.AutogradContext ctx, List<Tensor> grad_outputs)
            {
                var saved = ctx.get_saved_variables();
                var input1 = saved[0];
                var input2 = saved[1];
                var weight = saved[2];

                var grad_output1 = grad_outputs[0];
                var grad_output2 = grad_outputs[1];
                var grad_input1 = grad_output1.mm(weight);
                var grad_input2 = grad_output2.mm(weight);
                var grad_weight = grad_output1.t().mm(input1) + grad_output2.t().mm(input2);

                return new List<Tensor>() { grad_input1, grad_input2, null };

            }

            public override List<Tensor> forward(autograd.AutogradContext ctx, params object[] vars)
            {
                var input1 = (Tensor)vars[0];
                var input2 = (Tensor)vars[1];
                var weight = (Tensor)vars[2];

                ctx.save_for_backward(new() { input1, input2, weight });

                var output1 = input1.mm(weight.t());
                var output2 = input2.mm(weight.t());

                return new() { output1, output2 };
            }
        }
        class TwoInputLinearFunction : torch.autograd.MultiTensorFunction<TwoInputLinearFunction>
        {
            public override string Name => nameof(TwoInputLinearFunction);

            public override List<Tensor> backward(autograd.AutogradContext ctx, List<Tensor> grad_outputs)
            {
                var saved = ctx.get_saved_variables();
                var input1 = saved[0];
                var input2 = saved[1];
                var weight = saved[2];

                var grad_output1 = grad_outputs[0];
                var grad_output2 = grad_outputs[1];
                var grad_input1 = grad_output1.mm(weight);
                var grad_input2 = grad_output2.mm(weight);
                var grad_weight = grad_output1.t().mm(input1) + grad_output2.t().mm(input2);

                return new List<Tensor>() { grad_input1, grad_input2, grad_weight };

            }

            public override List<Tensor> forward(autograd.AutogradContext ctx, params object[] vars)
            {
                var input1 = (Tensor)vars[0];
                var input2 = (Tensor)vars[1];
                var weight = (Tensor)vars[2];
                
                ctx.save_for_backward(new() { input1, input2, weight });

                var output1 = input1.mm(weight.t());
                var output2 = input2.mm(weight.t());
                
                return new() { output1, output2 };
            }
        }

        class LinearFunction : torch.autograd.SingleTensorFunction<LinearFunction>
        {
            public override string Name => nameof(LinearFunction);

            public override List<Tensor> backward(autograd.AutogradContext ctx, Tensor grad_output)
            {
                var saved = ctx.get_saved_variables();
                var input = saved[0];
                var weight = saved[1];
                var bias = saved.Count == 2 ? null : saved[2];

                var grad_input = grad_output.mm(weight);
                var grad_weight = grad_output.t().mm(input);
                var grad_bias = bias is null || bias.IsInvalid ? null : grad_output.sum(0);

                return new List<Tensor>() { grad_input, grad_weight, grad_bias };
            }

            public override Tensor forward(autograd.AutogradContext ctx, params object[] vars)
            {
                var input = (Tensor)vars[0];
                var weight = (Tensor)vars[1];
                var bias = vars.Length == 2 ? null : (Tensor)vars[2];

                ctx.save_for_backward(new() { input, weight, bias });

                var output = input.mm(weight.t());
                if (bias is not null && !bias.IsInvalid)
                    output += bias.unsqueeze(0).expand_as(output);

                return output;
            }
        }

        private void LossIsClose(float expected, float actual, float tolerance = 0.001f)
        {
            // The error tolerance should be relative, not absolute.
            tolerance *= actual;
            Assert.True(MathF.Abs(actual - expected) <= tolerance, $"Expected {expected}, but got {actual}");
        }
    }
}