// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.IO;
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
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            float learning_rate = 0.00004f;
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                seq.ZeroGrad();

                output.backward();

                using (var noGrad = new AutoGradMode(false)) {
                    foreach (var param in seq.parameters()) {
                        var grad = param.grad();
                        var update = grad.mul(learning_rate);
                        param.sub_(update);
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

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            float learning_rate = 0.00004f;
            var loss = mse_loss(Reduction.Sum);

            float initialLoss = loss(seq.forward(x), y).ToSingle();
            float finalLoss = float.MaxValue;

            for (int i = 0; i < 10; i++) {
                var eval = seq.forward(x);
                var output = loss(eval, y);
                var lossVal = output.ToSingle();

                finalLoss = lossVal;

                seq.ZeroGrad();

                output.backward();

                using (var noGrad = new AutoGradMode(false)) {
                    foreach (var param in seq.parameters()) {
                        var grad = param.grad();
                        var update = grad.mul(learning_rate);
                        param.sub_(update);
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
        public void TestTrainingAdamAmsGrad()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

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

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

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

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

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
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate);
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
        public void TestTrainingRMSAlpha()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate, alpha: 0.75);
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
        public void TestTrainingRMSCentered()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(seq.parameters(), learning_rate, centered: true);
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
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate, momentum: 0.5);
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

        [Fact(Skip = "Fails with an exception in native code.")]
        public void TestTrainingSGDNesterov()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate, nesterov: true);
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
        public void TestTrainingSGDDefaults()
        {
            var lin1 = Linear(1000, 100);
            var lin2 = Linear(100, 10);
            var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("lin2", lin2));

            var x = Float32Tensor.randn(new long[] { 64, 1000 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(seq.parameters(), learning_rate);
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

            var x = Float32Tensor.randn(new long[] { 64, 3, 28, 28 });
            var y = Float32Tensor.randn(new long[] { 64, 10 });

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

                    using (Tensor x = Float32Tensor.randn(new long[] { 64, 3, 28, 28 }, device: (Device)device),
                           y = Float32Tensor.randn(new long[] { 64, 10 }, device: (Device)device)) {

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
                Assert.Throws<InvalidOperationException>(() => Float32Tensor.randn(new long[] { 64, 3, 28, 28 }).cuda());
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
