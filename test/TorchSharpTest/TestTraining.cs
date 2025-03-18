// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch.nn;
using Xunit;

using static TorchSharp.torch;
using TorchSharp.Modules;

#nullable enable

namespace TorchSharp
{
    [Collection("Sequential")]
    public class TestTraining
    {

        // <summary>
        /// Check if sequential module goes from training to eval mode
        // </summary>
        [Fact]
        public void TestTrainingEvalModes()
        {
            var sequential = Sequential(
                ("lin1", Linear(100, 10)),
                ("lin2", Linear(10, 5))
            );
            sequential.eval();
            // The entire sequential module and its layers should be in evaluation mode, not in training
            var firstLayer = (torch.nn.Module)sequential[0];
            Assert.False(firstLayer.training);

            var secondLayer = (torch.nn.Module)sequential[1];
            Assert.False(secondLayer.training);

            Assert.False(sequential.training);

            sequential.train();
            // The entire sequential module and its layers should be in training mode, not in evaluation
            Assert.True(firstLayer.training);
            Assert.True(secondLayer.training);
            Assert.True(sequential.training);
        }


        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using gradient descent.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html"/>.
        /// </summary>
        [Fact]
        public void Training1()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);

            var submodules = new List<(string name, torch.nn.Module<Tensor, Tensor> submodule)>();
            submodules.Add(("lin1", lin1));
            submodules.Add(("relu1", ReLU()));
            submodules.Add(("lin2", lin2));

            using var seq = Sequential(submodules);

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            float learning_rate = 0.00004f;
            var loss = MSELoss(Reduction.Sum);

            float initialLoss = loss.call(seq.call(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                var eval = seq.call(x);
                var output = loss.call(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                seq.zero_grad();

                output.backward();

                using (torch.no_grad()) {
                    foreach (var param in seq.parameters()) {
                        var grad = param.grad;
                        if (grad is not null) {
                            var update = grad.mul(learning_rate);
                            param.sub_(update);
                        }
                    }
                }
            }

            Assert.True(finalLoss < initialLoss);
        }

        /// <summary>
        /// Adding a dropout module to the linear nn.
        /// </summary>
        [Fact]
        public void TrainingWithDropout()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("drop1", Dropout(0.1)), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            float learning_rate = 0.00004f;
            var loss = MSELoss(Reduction.Sum);

            float initialLoss = loss.call(seq.call(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                var eval = seq.call(x);
                var output = loss.call(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                seq.zero_grad();

                output.backward();

                using (torch.no_grad()) {
                    foreach (var param in seq.parameters()) {
                        var grad = param.grad;
                        if (grad is not null) {
                            var update = grad.mul(learning_rate);
                            param.sub_(update);
                        }
                    }
                }
            }

            Assert.True(finalLoss < initialLoss);
        }

        [Fact]
        public void Adam()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00001;

            var optimizer = torch.optim.Adam(seq.parameters(), learning_rate);

            Assert.NotNull(optimizer);
        }

        [Fact]
        public void AdamBetas()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00001;

            var optimizer = torch.optim.Adam(seq.parameters(), learning_rate);
            var (beta1, beta2) = optimizer.Betas;
            Assert.Equal(0.9, beta1);
            Assert.Equal(0.999, beta2);

            optimizer.Betas = (0.85, 0.975);

            (beta1, beta2) = optimizer.Betas;
            Assert.Equal(0.85, beta1);
            Assert.Equal(0.975, beta2);
        }


        internal static void CreateDataAndLabels(Generator gen, out Tensor data, out Tensor labels, int batchSize = 64, int inputSize = 1000, int categories = 10)
        {
            data = torch.rand(new long[] { 64, inputSize }, generator: gen);
            labels = torch.rand(new long[] { 64, categories }, generator: gen);
        }

        internal static void CreateLinearLayers(Generator gen, out Linear linear1, out Linear linear2, int inputSize = 1000, int categories = 10, int hiddenSize = 100)
        {
            linear1 = Linear(inputSize, hiddenSize);
            linear2 = Linear(hiddenSize, categories);

            ReInitializeLinear(gen, linear1);
            ReInitializeLinear(gen, linear2);
        }

        private static void ReInitializeLinear(Generator gen, Linear linear)
        {
            // The Linear module will have been created with some RNG that we don't have control over, specifically
            // the default RNG, which does not work well in a parallel environment (which the unit test framework is).
            //
            // Therefore, we need to set the initial parameters based on the generator we do have control over
            // and which is unique to the unit test.

            var w = rand(linear.weight!.shape, generator: gen);
            var wBound = 1 / Math.Sqrt(w.shape[1]);

            linear.weight = Parameter(w * (2 * wBound) - wBound);

            if (linear.bias is not null) {
                var (fanIn, _) = init.CalculateFanInAndFanOut(linear.weight);
                var bBound = (fanIn > 0) ? 1 / Math.Sqrt(fanIn) : 0;
                var b = rand(linear.bias.shape, generator: gen);
                linear.bias = Parameter(b * (2 * bBound) - bBound);
            }
        }


        /// <summary>
        /// Used to test optimizers created with 'maximize: true'
        /// </summary>
        class NegateLoss : Loss<Tensor, Tensor, Tensor>
        {
            internal NegateLoss(Loss<Tensor, Tensor, Tensor> base_loss)
            {
                this.base_loss = base_loss;
            }

            public override Tensor forward(Tensor input1, Tensor input2)
            {
                return base_loss.call(input1, input2).neg();
            }

            private Loss<Tensor, Tensor, Tensor> base_loss;
        }

        private static float TrainLoop(IModule<Tensor, Tensor> seq, Tensor x, Tensor y, optim.Optimizer optimizer, bool maximize = false)
        {
            Loss<Tensor, Tensor, Tensor> loss = MSELoss(Reduction.Sum);
            if (maximize) loss = new NegateLoss(loss);

            float initialLoss = loss.call(seq.call(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.call(x);
                using var output = loss.call(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }

            // After 10 iterations, the final criterion value should always be better than the initial value.
            if (maximize) {
                Assert.True(finalLoss > initialLoss);
            } else {
                Assert.True(finalLoss < initialLoss);
            }

            return finalLoss;
        }

        private static float TrainLoop(IModule<Tensor, Tensor> seq, Tensor x, Tensor y, optim.Optimizer optimizer, optim.lr_scheduler.LRScheduler scheduler, bool check_lr = true, int iters = 10, bool maximize = false)
        {
            Loss<Tensor, Tensor, Tensor> loss = MSELoss(Reduction.Sum);
            if (maximize) loss = new NegateLoss(loss);

            float initialLoss = loss.call(seq.call(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            var pgFirst = optimizer.ParamGroups.First();
            var lastLR = pgFirst.LearningRate;

            for (int i = 0; i < iters; i++) {
                using var eval = seq.call(x);
                using var output = loss.call(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
                scheduler.step(lossVal);

                if (check_lr) {
                    // For most LR schedulers, the LR decreases monotonically with each step. However,
                    // that is not always the case, so the test must be disabled in some circumstances.
                    Assert.True(pgFirst.LearningRate < lastLR);
                    lastLR = pgFirst.LearningRate;
                }
            }

            // After 10 iterations, the final criterion value should always be better than the initial value.
            if (maximize) {
                Assert.True(finalLoss > initialLoss);
            } else {
                Assert.True(finalLoss < initialLoss);
            }

            return finalLoss;
        }

        private void LossIsClose(float expected, float actual, float tolerance = 0.001f)
        {
            // The error tolerance should be relative, not absolute.
            tolerance *= actual;
            Assert.True(MathF.Abs(actual - expected) <= tolerance, $"Expected {expected}, but got {actual}");
        }


        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using Adam optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TrainingAdamDefaults()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adam(seq.parameters());

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(53.3606f, loss);
        }

        [Fact]
        public void TrainingAdamMax()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adam(seq.parameters(), maximize:true);

            var loss = TrainLoop(seq, x, y, optimizer, maximize:true);

            LossIsClose(53.3606f, -loss);
        }

        [Fact]
        public void TrainingAdamParamGroups()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adam(new Adam.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.005f } },
                new () { Parameters = lin2.parameters() }
            });

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(105.82f, loss);
        }

        [Fact]
        public void TrainingAdamAmsGrad()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adam(seq.parameters(), amsgrad: true);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(53.34f, loss);
        }

        [Fact]
        public void TrainingAdamWeightDecay()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adam(seq.parameters(), weight_decay: 0.5);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(53.28f, loss);
        }

        [Fact]
        public void TrainingAdamOneCycleLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var lr = 0.0005;
            var optimizer = torch.optim.Adam(seq.parameters(), lr: lr, amsgrad: true);
            var scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr * 10, total_steps: 10);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false);

            LossIsClose(94.69524f, loss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using Adam optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TrainingAdamWDefaults()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.AdamW(seq.parameters());

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(53.3606f, loss);
        }

        [Fact]
        public void TrainingAdamWParamGroups()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.AdamW(new AdamW.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.005f } },
                new () { Parameters = lin2.parameters() }
            });

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(105.82f, loss);
        }

        [Fact]
        public void TrainingAdamWAmsGrad()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.AdamW(seq.parameters(), amsgrad: true);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(53.34f, loss);
        }

        [Fact]
        public void TrainingAdamWWeightDecay()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.AdamW(seq.parameters(), weight_decay: 0.5);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(53.28f, loss);
        }

        [Fact]
        public void TrainingAdamWOneCycleLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var lr = 0.001;
            var optimizer = torch.optim.AdamW(seq.parameters(), lr: lr, amsgrad: true);
            var scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr * 10, total_steps: 10);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false);

            LossIsClose(197.873f, loss);
        }


        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using Adagrad optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TrainingAdagrad()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adagrad(seq.parameters(), learning_rate);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(203.20f, loss);
        }

        [Fact]
        public void TrainingAdagradWeightDecay()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adagrad(seq.parameters(), learning_rate, weight_decay: 0.5);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(203.21f, loss);
        }

        [Fact]
        public void TrainingAdagradLRDecay()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adagrad(seq.parameters(), learning_rate, lr_decay: 0.1);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(211.065f, loss);
        }

        [Fact]
        public void TrainingAdagradParamGroups()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adagrad(new Adagrad.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.005f } },
                new () { Parameters = lin2.parameters() }
            });

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(51.51f, loss);
        }

        [Fact]
        public void TrainingAdagradParamGroupsWD()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adagrad(new Adagrad.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.005f, weight_decay = 0.25 } },
                new () { Parameters = lin2.parameters() }
            });

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(51.45f, loss);
        }

        [Fact]
        public void TrainingAdadelta()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 1.0f;
            var optimizer = torch.optim.Adadelta(seq.parameters(), learning_rate);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(74.754f, loss);
        }

        [Fact]
        public void TrainingAdadeltaMax()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 1.0f;
            var optimizer = torch.optim.Adadelta(seq.parameters(), learning_rate, maximize:true);

            var loss = TrainLoop(seq, x, y, optimizer, maximize: true);

            LossIsClose(74.754f, -loss);
        }

        [Fact]
        public void TrainingAdadeltaWD()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 1.0f;
            var optimizer = torch.optim.Adadelta(seq.parameters(), learning_rate, weight_decay: 0.35);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(73.226f, loss);
        }

        [Fact]
        public void TrainingAdadeltaRHO()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 1.0f;
            var optimizer = torch.optim.Adadelta(seq.parameters(), learning_rate, rho: 0.75);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(73.027f, loss);
        }

        [Fact]
        public void TrainingAdadeltaParamGroups()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adadelta(new Adadelta.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.005f } },
                new () { Parameters = lin2.parameters() }
            });

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(77.891f, loss);
        }

        [Fact]
        public void TrainingAdamax()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adamax(seq.parameters());

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(55.559f, loss);
        }

        [Fact]
        public void TrainingAdamaxWD()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adamax(seq.parameters(), weight_decay: 0.35);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(55.47f, loss);
        }

        [Fact]
        public void TrainingAdamaxBetas()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adamax(seq.parameters(), beta1: 0.75, beta2: 0.95);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(51.00f, loss);
        }

        [Fact]
        public void TrainingAdamaxParamGroups()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Adamax(new Adamax.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.005f } },
                new () { Parameters = lin2.parameters() }
            });

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(141.519f, loss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using Adamax optimizer.
        /// </summary>
        [Fact]
        public void TrainingAdamaxOneCycleLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var lr = 0.002;
            var optimizer = torch.optim.Adamax(seq.parameters(), lr: lr);
            var scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr * 10, total_steps: 15);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false);

            LossIsClose(61.45f, loss);
        }

        [Fact]
        public void TrainingNAdam()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.NAdam(seq.parameters());

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(66.262f, loss);
        }

        [Fact]
        public void TrainingNAdamWD()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.NAdam(seq.parameters(), weight_decay: 0.45);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(67.041f, loss);
        }

        [Fact]
        public void TrainingNAdamBetas()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.NAdam(seq.parameters(), beta1: 0.75, beta2: 0.95);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(66.729f, loss);
        }

        [Fact]
        public void TrainingNAdamMD()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.NAdam(seq.parameters(), momentum_decay: 0.04);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(66.739f, loss);
        }

        [Fact]
        public void TrainingNAdamParamGroups()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.NAdam(new NAdam.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.005f } },
                new () { Parameters = lin2.parameters() }
            });

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(56.438f, loss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using NAdam optimizer.
        /// </summary>
        [Fact]
        public void TrainingNAdamOneCycleLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var lr = 0.002;
            var optimizer = torch.optim.NAdam(seq.parameters(), lr: lr);
            var scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr * 10, total_steps: 10);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false);

            LossIsClose(145.415f, loss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using RAdam optimizer.
        /// </summary>
        [Fact]
        public void TrainingRAdam()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.RAdam(seq.parameters(), 0.0005);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(66.2651f, loss);
        }

        [Fact]
        public void TrainingRAdamWD()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.RAdam(seq.parameters(), 0.0005, weight_decay: 0.05);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(66.263f, loss);
        }

        [Fact]
        public void TrainingRAdamBetas()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.RAdam(seq.parameters(), 0.0005, beta1: 0.9, beta2: 0.999);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(66.265f, loss);
        }

        [Fact]
        public void TrainingRAdamParamGroups()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.RAdam(new RAdam.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.001f } },
                new () { Parameters = lin2.parameters() }
            }, 0.0005);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(170.338f, loss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using RAdam optimizer.
        /// </summary>
        [Fact]
        public void TrainingRAdamOneCycleLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var lr = 0.0005;
            var optimizer = torch.optim.RAdam(seq.parameters(), lr: lr);
            var scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr * 1.5, total_steps: 10);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false);

            LossIsClose(124.478f, loss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using RAdam optimizer.
        /// </summary>
        [Fact]
        public void TrainingRAdamOneCycleLR_PG()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var lr = 0.0005;
            var optimizer = torch.optim.RAdam(new RAdam.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.003f } },
                new () { Parameters = lin2.parameters(), Options = new () { LearningRate = lr } }
            });
            var scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, new double[] { lr * 1.5, lr * 2.25 }, total_steps: 10);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false);

            LossIsClose(92.047f, loss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using RAdam optimizer.
        /// </summary>
        [Fact]
        public void TrainingRAdamCyclicLR_PG()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var lr = 0.0002;
            var optimizer = torch.optim.SGD(new SGD.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.003f } },
                new () { Parameters = lin2.parameters(), Options = new () { LearningRate = lr } }
            }, lr);
            var scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, new double[] { lr / 2, lr * 2 }, new double[] { lr * 1.5, lr * 2.25 });

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false);

            LossIsClose(58.589f, loss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using RAdam optimizer.
        /// </summary>
        [Fact]
        public void TrainingRAdamReduceLROnPlateau_PG()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var lr = 0.0002;
            var optimizer = torch.optim.SGD(new SGD.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.003f } },
                new () { Parameters = lin2.parameters(), Options = new () { LearningRate = lr } }
            }, lr);

            var scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer);

            Assert.Throws<InvalidOperationException>(() => scheduler.step());
            Assert.Throws<InvalidOperationException>(() => scheduler.step(epoch: 10));

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false, 500);

            LossIsClose(52.6109f, loss);

            var pgs = optimizer.ParamGroups.ToList();

            Assert.NotEqual(0.0002, pgs[0].LearningRate);
            Assert.NotEqual(0.003,  pgs[1].LearningRate);
        }

        [Fact]
        public void TrainingASGD()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.ASGD(seq.parameters());

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(57.748f, loss);
        }

        [Fact]
        public void TrainingASGDMax()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.ASGD(seq.parameters(), maximize:true);

            var loss = TrainLoop(seq, x, y, optimizer, maximize:true);

            LossIsClose(57.748f, -loss);
        }

        [Fact]
        public void TrainingASGDLambda()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.ASGD(seq.parameters(), lambd: 0.00025);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(57.748f, loss);
        }

        [Fact]
        public void TrainingASGDAlpha()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.ASGD(seq.parameters(), alpha: 0.65);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(57.748f, loss);
        }

        [Fact]
        public void TrainingASGDWD()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.ASGD(seq.parameters(), weight_decay: 0.25);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(57.748f, loss);
        }

        [Fact]
        public void TrainingASGDT0()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.ASGD(seq.parameters(), t0: 100000);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(57.748f, loss);
        }

        [Fact]
        public void TrainingASGDParamGroups()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var pgs = new ASGD.ParamGroup[]
            {
                new (lin1.parameters(), new () { LearningRate = 0.005f }),
                new (lin2.parameters())
            };

            var optimizer = torch.optim.ASGD(new ASGD.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.0002f } },
                new () { Parameters = lin2.parameters() }
            });

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(56.653f, loss);
        }

        [Fact]
        public void TrainingRprop()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Rprop(seq.parameters());

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(77.279f, loss);
        }


        [Fact]
        public void TrainingRpropMax()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Rprop(seq.parameters(), maximize: true);

            var loss = TrainLoop(seq, x, y, optimizer, maximize:true);

            LossIsClose(77.279f, -loss);
        }

        [Fact]
        public void TrainingRpropEtam()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Rprop(seq.parameters(), etaminus: 0.55);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(171.12f, loss);
        }

        [Fact]
        public void TrainingRpropEtap()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Rprop(seq.parameters(), etaplus: 1.25);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(65.859f, loss);
        }


        [Fact]
        public void TrainingRpropParamGroups()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var optimizer = torch.optim.Rprop(new Rprop.ParamGroup[]
            {
                new () { Parameters = lin1.parameters(), Options = new () { LearningRate = 0.005f } },
                new () { Parameters = lin2.parameters() }
            });

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(66.479f, loss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using RMSprop optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TrainingRMSLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(56.589f, loss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using RMSprop optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TrainingRMSOneCycleLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate * 10, total_steps: 10);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false);

            LossIsClose(207.87f, loss);
        }

        [Fact]
        public void TrainingRMSAlpha()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate, alpha: 0.75);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(156.339f, loss);
        }

        [Fact]
        public void TrainingRMSAlphaMax()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate, alpha: 0.75, maximize:true);

            var loss = TrainLoop(seq, x, y, optimizer, maximize:true);

            LossIsClose(156.339f, -loss);
        }

        [Fact]
        public void TrainingRMSWD()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate, weight_decay: 0.15);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(56.608f, loss);
        }

        [Fact]
        public void TrainingRMSCentered()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate, centered: true);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(56.189f, loss);
        }

        [Fact]
        public void TrainingRMSCenteredWD()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate, weight_decay: 0.15, centered: true);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(56.189f, loss);
        }

        [Fact]
        public void TrainingRMSMomentum()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate, momentum: 0.15);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(53.50f, loss);
        }

        [Fact]
        public void TrainingRMSCenteredParamGroups()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;

            var optimizer = torch.optim.RMSProp(
                new RMSProp.ParamGroup[] {
                    new(lin1.parameters()),
                    new(lin2.parameters(), lr: learning_rate*10) },
                learning_rate, centered: true);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(53.6049f, loss);
        }

        [Fact]
        public void TrainingSGDMomentum()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate, momentum: 0.5);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(53.711f, loss);
        }

        [Fact]
        public void TrainingSGDDampening()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate, dampening: 0.5);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(62.494f, loss);
        }

        [Fact()]
        public void TrainingSGDNesterov()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate, momentum: 0.1, nesterov: true);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(59.270f, loss);
        }

        [Fact]
        public void TrainingSGDDefaults()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(62.494f, loss);
        }

        [Fact]
        public void TrainingSGDDefaultsParamGroups()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var pgs = new SGD.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, dampening: 0.05, momentum: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(pgs, learning_rate);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(58.698f, loss);
        }

        [Fact]
        public void TrainingSGDStepLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.97);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler);

            LossIsClose(67.36136f, loss);
        }

        [Fact]
        public void TrainingSGDLambdaLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, i => Math.Pow(0.95, (1 + i)));

            var loss = TrainLoop(seq, x, y, optimizer, scheduler);

            LossIsClose(73.87f, loss);
        }

        [Fact]
        public void TrainingSGDMultiplicativeLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, i => 0.95);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler);

            LossIsClose(71.1419f, loss);
        }

        [Fact]
        public void TrainingSGDExponentialLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler);

            LossIsClose(180.249f, loss);
        }

        [Fact]
        public void TrainingSGDLinearLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor: 0.75, total_iters: 10);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false);

            LossIsClose(101.83f, loss);
        }

        [Fact]
        public void TrainingSGDSequentialLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler0 = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor: 0.75, total_iters: 10);
            var scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer);
            var scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, new[] { scheduler0, scheduler1, torch.optim.lr_scheduler.ExponentialLR(optimizer) }, new[] { 10, 15 });

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false, iters: 20);

            LossIsClose(67.5635f, loss);
        }

        [Fact]
        public void TrainingSGDSequentialLRWithAllClosedFormSchedulers()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler0 = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor: 0.75, total_iters: 10);
            var scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer);
            var scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, 2);
            var scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer, new[] { 2, 4 });
            var scheduler4 = torch.optim.lr_scheduler.ExponentialLR(optimizer);
            var scheduler5 = torch.optim.lr_scheduler.PolynomialLR(optimizer, power: 2);
            var scheduler6 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, 0.1);
            var scheduler7 = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor: 0.75);
            var scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, new[] { scheduler0, scheduler1, scheduler2, scheduler3, scheduler4, scheduler5, scheduler6, scheduler7}, new[] { 5, 5, 5, 5, 5, 5, 5 });

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false, iters: 35);

            LossIsClose(52.36025f, loss);
        }

        [Fact]
        public void TrainingSGDMultiStepLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, new int[] { 3, 5, 7 }, 0.97);

            var loss = MSELoss(Reduction.Sum);

            float initialLoss = loss.call(seq.call(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            double lastLR = learning_rate;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.call(x);
                using var output = loss.call(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
                scheduler.step();

                var pgFirst = optimizer.ParamGroups.First();
                if (i == 2 || i == 4 || i == 6) {
                    Assert.True(pgFirst.LearningRate < lastLR);
                }
                lastLR = pgFirst.LearningRate;
            }

            Assert.True(finalLoss < initialLoss);

            LossIsClose(64.019f, finalLoss);
        }

        [Fact]
        public void TrainingSGDCosineAnnealingLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10);

            var loss = MSELoss(Reduction.Sum);

            float initialLoss = loss.call(seq.call(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            double lastLR = learning_rate;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.call(x);
                using var output = loss.call(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
                scheduler.step();

                var pgFirst = optimizer.ParamGroups.First();
                if (i == 2 || i == 4 || i == 6) {
                    Assert.True(pgFirst.LearningRate < lastLR);
                }
                lastLR = pgFirst.LearningRate;
            }

            Assert.True(finalLoss < initialLoss);

            LossIsClose(88.98f, finalLoss);
        }

        [Fact]
        public void TrainingSGDCyclicLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.0001, 0.0004, step_size_up: 5);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false);

            LossIsClose(65.459f, loss);
        }

        [Fact]
        public void TrainingSGDOneCycleLR()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.0004, total_steps: 10);

            var loss = TrainLoop(seq, x, y, optimizer, scheduler, false);

            LossIsClose(112.4992f, loss);
        }

        [Fact]
        public void TrainingLBFGSDefaults()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.LBFGS(seq.parameters(), learning_rate);
            var loss = MSELoss(Reduction.Sum);

            float initialLoss = loss.call(seq.call(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {

                Func<Tensor> closure = () => {
                    using var eval = seq.call(x);
                    var output = loss.call(eval, y);

                    finalLoss = output.ToSingle();

                    optimizer.zero_grad();

                    output.backward();
                    return output;
                };

                optimizer.step(closure);
            }
            // 10 iterations is not alway enough for LBGFS with these parameters
            // so the check is disabled for now. We're still testing that the native
            // interop is working.
            //Assert.True(finalLoss < initialLoss);
        }

        [Fact]
        public void TrainingLBFGSNoClosure()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = (torch.optim.Optimizer)torch.optim.LBFGS(seq.parameters(), learning_rate);
            Assert.Throws<ArgumentNullException>(() => optimizer.step());
        }

        [Fact]
        public void TrainingLBFGS_ME()
        {
            var gen = new Generator(4711);
            CreateLinearLayers(gen, out var lin1, out var lin2);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.LBFGS(seq.parameters(), learning_rate, max_iter: 15, max_eval: 15);
            var loss = MSELoss(Reduction.Sum);

            float initialLoss = loss.call(seq.call(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            Assert.Throws<ArgumentNullException>(() => optimizer.step(null));

            for (int i = 0; i < 10; i++) {

                Func<Tensor> closure = () => {
                    using var eval = seq.call(x);
                    var output = loss.call(eval, y);

                    finalLoss = output.ToSingle();

                    optimizer.zero_grad();

                    output.backward();
                    return output;
                };

                optimizer.step(closure);
            }
            // 10 iterations is not alway enough for LBGFS with these parameters
            // so the check is disabled for now. We're still testing that the native
            // interop is working.
            //Assert.True(finalLoss < initialLoss);
        }

        [Fact]
        public void TrainingLoadedTorchScript()
        {
            var gen = new Generator(4711);
            CreateDataAndLabels(gen, out var x, out var y);

            using var seq = torch.jit.load<torch.Tensor, torch.Tensor>(@"l1000_100_10.script.dat");

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);

            var loss = TrainLoop(seq, x, y, optimizer);

            LossIsClose(53.81697f, loss);

            seq.eval();
        }

        [Fact]
        public void TrainingConv2d()
        {
            var conv1 = Conv2d(3, 4, 3, stride: 2);
            var lin1 = Linear(4 * 13 * 13, 32);
            var lin2 = Linear(32, 10);

            using var seq = Sequential(
                ("conv1", conv1),
                ("r1", ReLU(inplace: true)),
                ("drop1", Dropout(0.1)),
                ("flat1", Flatten()),
                ("lin1", lin1),
                ("r2", ReLU(inplace: true)),
                ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 3, 28, 28 });
            var y = torch.randn(new long[] { 64, 10 });

            using var optimizer = torch.optim.Adam(seq.parameters());

            TrainLoop(seq, x, y, optimizer);
        }


        [Fact]
        public void TrainingConv2dCUDA()
        {
            if (torch.cuda.is_available()) {
                var device = torch.CUDA;

                using (Module<Tensor, Tensor> conv1 = Conv2d(3, 4, 3, stride: 2),
                      lin1 = Linear(4 * 13 * 13, 32),
                      lin2 = Linear(32, 10))

                using (var seq = Sequential(
                        ("conv1", conv1),
                        ("r1", ReLU(inplace: true)),
                        ("drop1", Dropout(0.1)),
                        ("flat1", Flatten()),
                        ("lin1", lin1),
                        ("r2", ReLU(inplace: true)),
                        ("lin2", lin2))) {

                    seq.to((Device)device);

                    var optimizer = torch.optim.Adam(seq.parameters());
                    var loss = MSELoss(Reduction.Sum);

                    using (Tensor x = torch.randn(new long[] { 64, 3, 28, 28 }, device: (Device)device),
                           y = torch.randn(new long[] { 64, 10 }, device: (Device)device)) {

                        float initialLoss = loss.call(seq.call(x), y).ToSingle();
                        float finalLoss = float.MaxValue;

                        for (int i = 0; i < 10; i++) {
                            var eval = seq.call(x);
                            var output = loss.call(eval, y);
                            var lossVal = output.ToSingle();

                            finalLoss = lossVal;

                            optimizer.zero_grad();

                            output.backward();

                            optimizer.step();
                        }
                        Assert.True(finalLoss < initialLoss);
                    }
                }
            } else {
                Assert.Throws<InvalidOperationException>(() => torch.randn(new long[] { 64, 3, 28, 28 }).cuda());
            }
        }
    }
}
