using System.Linq;
using System.Collections.Generic;
using Xunit;

using static TorchSharp.torch;
using System;
using System.Threading;

namespace TorchSharp
{
    public class TestAutogradFunction
    {

        private void TestCustomLinearFunction(Device device)
        {
            var x = torch.randn([2, 3], device: device).requires_grad_();
            var weight = torch.randn([4, 3], device: device).requires_grad_();
            var y = LinearFunction.apply(x, weight);
            y.sum().backward();

            Assert.NotNull(x.grad());
            Assert.NotNull(weight.grad());
        }


        private void TestCustomTwoInputLinearFunction(Device device)
        {
            var x1 = torch.randn([2, 3], device: device).requires_grad_();
            var x2 = torch.randn([5, 3], device: device).requires_grad_();
            var weight = torch.randn([4, 3], device: device).requires_grad_();
            var y = TwoInputLinearFunction.apply(x1, x2, weight);
            (y[0].sum() + y[1].sum()).backward();

            Assert.NotNull(x1.grad());
            Assert.NotNull(x2.grad());
            Assert.NotNull(weight.grad());
        }

        private float TrainXOR(Device device)
        {
            Generator gen = new torch.Generator();
            var weight1 = torch.nn.Parameter(torch.randn([2, 2], generator: gen).to(device));
            var weight2 = torch.nn.Parameter(torch.randn([1, 2], generator: gen).to(device));
            var optim = torch.optim.SGD([weight1, weight2], 0.01);

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
            TestCustomLinearFunction(torch.CPU);
        }

        [Fact]
        public void TestCustomLinearFunction_CUDA()
        {
            if (torch.cuda.is_available()) {
                TestCustomLinearFunction(torch.CUDA);
            }
        }

        [Fact]
        public void TestCustomTwoInputLinearFunction_CPU()
        {
            TestCustomTwoInputLinearFunction(torch.CPU);
        }

        [Fact]
        public void TestCustomTwoInputLinearFunction_CUDA()
        {
            if (torch.cuda.is_available()) {
                TestCustomTwoInputLinearFunction(torch.CUDA);
            }
        }

        [Fact]
        public void TestCustomLinearXORLearn_CPU()
        {
            float loss = TrainXOR(torch.CPU);
            LossIsClose(loss, 0.4513f);
        }

        [Fact]
        public void TestCustomLinearXORLearn_CUDA()
        {
            if (torch.cuda.is_available()) {
                float loss = TrainXOR(torch.CUDA);
                LossIsClose(loss, 0.4513f);
            }
        }

        [Fact]
        public void TestCustomLinearXORLearn_Parallel_CPU()
        {
            Enumerable.Range(0, 20).AsParallel().ForAll(i => {
                float loss = TrainXOR(torch.CPU);
                LossIsClose(loss, 0.4513f);
            });
        }

        [Fact]
        private void TestCustomLinearFunctionWithGC()
        {
            var x = torch.randn([2, 3]).requires_grad_();
            var weight = torch.randn([4, 3]).requires_grad_();
            var y = LinearFunction.apply(x, weight);

            // Try to force objects to be cleaned up by giving them a timer + explicitly calling the GC
            Thread.Sleep(2000);
            GC.Collect();

            y.sum().backward();

            Assert.NotNull(x.grad());
            Assert.NotNull(weight.grad());
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
                
                return [grad_input1, grad_input2, grad_weight];

            }

            public override List<Tensor> forward(autograd.AutogradContext ctx, params object[] vars)
            {
                var input1 = (Tensor)vars[0];
                var input2 = (Tensor)vars[1];
                var weight = (Tensor)vars[2];
                
                ctx.save_for_backward([input1, input2, weight]);

                var output1 = input1.mm(weight.t());
                var output2 = input2.mm(weight.t());
                
                return new() { output1, output2 };
            }
        }

        class LinearFunction : torch.autograd.SingleTensorFunction<LinearFunction>
        {
            public override string Name => nameof(LinearFunction);

            public override List<Tensor> backward(autograd.AutogradContext ctx, List<Tensor> grad_outputs)
            {
                var saved = ctx.get_saved_variables();
                var input = saved[0];
                var weight = saved[1];
                var bias = saved.Count == 2 ? null : saved[2];

                var grad_output = grad_outputs[0];
                var grad_input = grad_output.mm(weight);
                var grad_weight = grad_output.t().mm(input);
                var grad_bias = bias is null || bias.IsInvalid ? null : grad_output.sum(0);

                return [grad_input, grad_weight, grad_bias];
            }

            public override Tensor forward(autograd.AutogradContext ctx, params object[] vars)
            {
                var input = (Tensor)vars[0];
                var weight = (Tensor)vars[1];
                var bias = vars.Length == 2 ? null : (Tensor)vars[2];

                ctx.save_for_backward([input, weight, bias]);

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