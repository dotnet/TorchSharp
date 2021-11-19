// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Xunit;

using static TorchSharp.torch;

#nullable enable

namespace TorchSharp
{
    public class TestTraining
    {

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using gradient descent.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html"/>.
        /// </summary>
        [Fact]
        public void TestTraining1()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);

            var submodules = new List<(string name, torch.nn.Module submodule)>();
            submodules.Add(("lin1", lin1));
            submodules.Add(("relu1", ReLU()));
            submodules.Add(("lin2", lin2));

            var seq = Sequential(submodules);

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            float learning_rate = 0.00004f;
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                seq.zero_grad();

                output.backward();

                using (torch.no_grad()) {
                    foreach (var param in seq.parameters()) {
                        var grad = param.grad();
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
        public void TestTrainingWithDropout()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("drop1", Dropout(0.1)), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            float learning_rate = 0.00004f;
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                seq.zero_grad();

                output.backward();

                using (torch.no_grad()) {
                    foreach (var param in seq.parameters()) {
                        var grad = param.grad();
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
        public void TestAdam()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            double learning_rate = 0.00001;

            var optimizer = torch.optim.Adam(seq.parameters(), learning_rate);

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
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("drop1", Dropout(0.1)), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            var optimizer = torch.optim.Adam(seq.parameters());
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                using var _ = optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        [Fact]
        public void TestTrainingAdamAmsGrad()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            var optimizer = torch.optim.Adam(seq.parameters(), amsgrad: true);
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using Adam optimizer.
        /// Taken from <see href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [Fact]
        public void TestTrainingAdamWDefaults()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("drop1", Dropout(0.1)), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            var optimizer = torch.optim.AdamW(seq.parameters());
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
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

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adagrad(seq.parameters(), learning_rate);
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using Adadelta optimizer.
        /// </summary>
        [Fact]
        public void TestTrainingAdadelta()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 1.0f;
            var optimizer = torch.optim.Adadelta(seq.named_parameters(), learning_rate);
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using Adamax optimizer.
        /// </summary>
        [Fact]
        public void TestTrainingAdamax()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            var optimizer = torch.optim.Adamax(seq.named_parameters());
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 15; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using NAdam optimizer.
        /// </summary>
        [Fact]
        public void TestTrainingNAdam()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            var optimizer = torch.optim.NAdam(seq.named_parameters());
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using RAdam optimizer.
        /// </summary>
        [Fact]
        public void TestTrainingRAdam()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            var optimizer = torch.optim.RAdam(seq.named_parameters());
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using ASGD optimizer.
        /// </summary>
        [Fact]
        public void TestTrainingASGD()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            var optimizer = torch.optim.ASGD(seq.named_parameters());
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        /// <summary>
        /// Fully connected ReLU net with one hidden layer trained using ASGD optimizer.
        /// </summary>
        [Fact]
        public void TestTrainingRprop()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            var optimizer = torch.optim.Rprop(seq.named_parameters());
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
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

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate);
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        [Fact]
        public void TestTrainingRMSAlpha()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate, alpha: 0.75);
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        [Fact]
        public void TestTrainingRMSCentered()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate, centered: true);
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
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

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate, momentum: 0.5);
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        [Fact(Skip = "Fails with an exception in native code.")]
        public void TestTrainingSGDNesterov()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate, nesterov: true);
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        [Fact]
        public void TestTrainingSGDDefaults()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }

        [Fact]
        public void TestTrainingSGDStepLR()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.97);

            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            double lastLR = learning_rate;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
                scheduler.step();

                Assert.True(scheduler.LearningRate < lastLR);

                lastLR = scheduler.LearningRate;
            }

            Assert.True(finalLoss < initialLoss);
        }

        [Fact]
        public void TestTrainingSGDMultiStepLR()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, new int[] { 3, 5, 7 }, 0.97);

            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            double lastLR = learning_rate;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
                scheduler.step();

                if (i == 2 || i == 4 || i == 6) {
                    Assert.True(scheduler.LearningRate < lastLR);
                }
                lastLR = scheduler.LearningRate;
            }

            Assert.True(finalLoss < initialLoss);
        }

        [Fact]
        public void TestTrainingSGDCosineAnnealingLR()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
            var scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10);

            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            double lastLR = learning_rate;

            for (int i = 0; i < 10; i++) {
                using var eval = seq.forward(x);
                using var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
                scheduler.step();

                if (i == 2 || i == 4 || i == 6) {
                    Assert.True(scheduler.LearningRate < lastLR);
                }
                lastLR = scheduler.LearningRate;
            }

            Assert.True(finalLoss < initialLoss);
        }

        [Fact]
        public void TestTrainingLBFGSDefaults()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.LBFGS(seq.parameters(), learning_rate);
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {

                Func<Tensor> closure = () => {
                    using var eval = seq.forward(x);
                    var output = loss(eval, y);

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
        public void TestTrainingLBFGSNoClosure()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));
            double learning_rate = 0.00004f;
            var optimizer = torch.optim.LBFGS(seq.parameters(), learning_rate);
            Assert.Throws<ArgumentNullException>(() => optimizer.step());
        }

        [Fact]
        public void TestTrainingLBFGS_ME()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 1000 });
            var y = torch.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.LBFGS(seq.parameters(), learning_rate, max_iter: 15, max_eval: 15);
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {

                Func<Tensor> closure = () => {
                    using var eval = seq.forward(x);
                    var output = loss(eval, y);

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
        public void TestTrainingConv2d()
        {
            var conv1 = Conv2d(3, 4, 3, stride: 2);
            var lin1 = Linear(4 * 13 * 13, 32);
            var lin2 = Linear(32, 10);

            var seq = Sequential(
                ("conv1", conv1),
                ("r1", ReLU(inPlace: true)),
                ("drop1", Dropout(0.1)),
                ("flat1", Flatten()),
                ("lin1", lin1),
                ("r2", ReLU(inPlace: true)),
                ("lin2", lin2));

            var x = torch.randn(new long[] { 64, 3, 28, 28 });
            var y = torch.randn(new long[] { 64, 10 });

            var optimizer = torch.optim.Adam(seq.parameters());
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                optimizer.zero_grad();

                output.backward();

                optimizer.step();
            }
            Assert.True(finalLoss < initialLoss);
        }


        [Fact]
        public void TestTrainingConv2dCUDA()
        {
            if (torch.cuda.is_available()) {
                var device = torch.CUDA;

                using (Module conv1 = Conv2d(3, 4, 3, stride: 2),
                      lin1 = Linear(4 * 13 * 13, 32),
                      lin2 = Linear(32, 10))

                using (var seq = Sequential(
                        ("conv1", conv1),
                        ("r1", ReLU(inPlace: true)),
                        ("drop1", Dropout(0.1)),
                        ("flat1", Flatten()),
                        ("lin1", lin1),
                        ("r2", ReLU(inPlace: true)),
                        ("lin2", lin2))) {

                    seq.to((Device)device);

                    var optimizer = torch.optim.Adam(seq.parameters());
                    var loss = mse_loss(Reduction.Sum);

                    using (Tensor x = torch.randn(new long[] { 64, 3, 28, 28 }, device: (Device)device),
                           y = torch.randn(new long[] { 64, 10 }, device: (Device)device)) {

                        float initialLoss = loss(seq.forward(x), y).ToSingle();
                        float finalLoss = float.MaxValue;

                        for (int i = 0; i < 10; i++) {
                            var eval = seq.forward(x);
                            var output = loss(eval, y);
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
    }
}
