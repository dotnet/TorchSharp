using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using TorchSharp.JIT;
using TorchSharp.Tensor;

namespace TorchSharp.Test
{
    [TestClass]
    public class TorchSharp
    {
        [TestMethod]
        public void CreateFloatTensorOnes()
        {
            var ones = FloatTensor.Ones(new long[] { 2, 2 });
            Assert.IsNotNull(ones);
        }

        [TestMethod]
        public void CreateFloatTensorCheckMemory()
        {
            ITorchTensor<float> ones = null;

            for (int i = 0; i < 10; i++)
            {
                using (var tmp = FloatTensor.Ones(new long[] { 1000, 1000, 1000 }))
                {
                    ones = tmp;
                    Assert.IsNotNull(ones);
                }
            }
        }

        [TestMethod]
        public void CreateFloatTensorOnesCheckData()
        {
            var ones = FloatTensor.Ones(new long[] { 2, 2 });
            var data = ones.Data;

            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(1.0, data[i]);
            }
        }

        [TestMethod]
        public void CreateFloatTensorZerosCheckData()
        {
            var zeros = FloatTensor.Zeros(new long[] { 2, 2 });
            var data = zeros.Data;

            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(0, data[i]);
            }
        }

        [TestMethod]
        public void CreateIntTensorOnesCheckData()
        {
            var ones = IntTensor.Ones(new long[] { 2, 2 });
            var data = ones.Data;

            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(data[i], 1);
            }
        }

        [TestMethod]
        public void CreateFloatTensorCheckDevice()
        {
            var ones = FloatTensor.Ones(new long[] { 2, 2 });
            var device = ones.Device;

            Assert.AreEqual(ones.Device, "cpu");
        }

        [TestMethod]
        public void CreateFloatTensorFromData()
        {
            var data = new float[1000];
            data[100] = 1;

            using (var tensor = FloatTensor.From(data, new long[] { 100, 10 }))
            {
                Assert.AreEqual(tensor.Data[100], 1);
            }
        }

        [TestMethod]
        public void CreateFloatTensorFromDataCheckDispose()
        {
            var data = new float[1000];
            data[100] = 1;

            using (var tensor = FloatTensor.From(data, new long[] { 100, 10 }))
            {
                Assert.AreEqual(tensor.Data[100], 1);
            }

            Assert.AreEqual(data[100], 1);
        }

        [TestMethod]
        public void CreateFloatTensorFromData2()
        {
            CreateFloatTensorFromData2Generic<float>();
        }

        private static void CreateFloatTensorFromData2Generic<T>()
        {
            var data = new T[1000];

            using (var tensor = data.ToTorchTensor(new long[] { 10, 100 }))
            {
                Assert.AreEqual(tensor.Data[100], default(T));
            }
        }

        [TestMethod]
        public void CreateFloatTensorFromScalar()
        {
            float scalar = 333.0f;

            using (var tensor = FloatTensor.From(scalar))
            {
                Assert.AreEqual(tensor.DataItem, 333);
            }
        }

        [TestMethod]
        public void CreateFloatTensorFromScalar2()
        {
            float scalar = 333.0f;

            using (var tensor = scalar.ToTorchTensor())
            {
                Assert.AreEqual(tensor.DataItem, 333);
            }
        }

        [TestMethod]
        public void InitUniform()
        {
            using (var tensor = FloatTensor.Zeros(new long[] { 2, 2 }))
            {
                NN.Init.Uniform(tensor);

                Assert.IsNotNull(tensor);
            } 
        }

        [TestMethod]
        public void TestSparse()
        {
            using (var i = LongTensor.From(new long[] { 0, 1, 1, 2, 0, 2 }, new long[] { 2, 3 }))
            using (var v = FloatTensor.From(new float[] { 3, 4, 5 }, new long[] { 3 }))
            {
                var sparse = FloatTensor.Sparse(i, v, new long[] { 2, 3 });

                Assert.IsNotNull(sparse);
                Assert.IsTrue(sparse.IsSparse);
            }
        }

        [TestMethod]
        public void CopyCpuToCuda()
        {
            var cpu = FloatTensor.Ones(new long[] { 2, 2 });
            Assert.AreEqual(cpu.Device, "cpu");

            if (Torch.IsCudaAvailable())
            {
                var cuda = cpu.Cuda();
                Assert.AreEqual(cuda.Device, "cuda");

                // Copy back to CPU to inspect the elements
                cpu = cuda.Cpu();
                var data = cpu.Data;
                for (int i = 0; i < 4; i++)
                {
                    Assert.AreEqual(data[i], 1);
                }
            }
            else
            {
                Assert.ThrowsException<InvalidOperationException>(cpu.Cuda);
            }

        }

        [TestMethod]
        public void CopyCudaToCpu()
        {
            if (Torch.IsCudaAvailable())
            {
                var cuda = FloatTensor.Ones(new long[] { 2, 2 }, "cuda");
                Assert.AreEqual(cuda.Device, "cuda");

                var cpu = cuda.Cpu();
                Assert.AreEqual(cpu.Device, "cpu");

                var data = cpu.Data;
                for (int i = 0; i < 4; i++)
                {
                    Assert.AreEqual(data[i], 1);
                }
            }
            else
            {
                Assert.ThrowsException<InvalidOperationException>(() => { FloatTensor.Ones(new long[] { 2, 2 }, "cuda"); });
            }
        }

        [TestMethod]
        public void ScoreModel()
        {
            var ones = FloatTensor.Ones(new long[] { 1, 3, 224, 224 });
            Assert.IsNotNull(ones);

            var module = JIT.Module.Load(@"..\..\..\Resources\model.pt");
            Assert.IsNotNull(module);

            var result = module.Forward(ones);
            Assert.IsNotNull(result);
        }

        [TestMethod]
        public void LoadModelCheckInput()
        {
            var module = JIT.Module.Load(@"E:\Source\Repos\libtorch\model.pt");
            Assert.IsNotNull(module);

            var num = module.GetNumberOfInputs();

            for (int i = 0; i < num; i++)
            {
                var type = module.GetInputType(i);

                Assert.IsNotNull(type as DynamicType);
            }
        }

        [TestMethod]
        public void LoadModelCheckOutput()
        {
            var module = JIT.Module.Load(@"E:\Source\Repos\libtorch\model.pt");
            Assert.IsNotNull(module);

            var num = module.GetNumberOfOutputs();

            for (int i = 0; i < num; i++)
            {
                var type = module.GetOutputType(i);

                Assert.IsNotNull(type as DynamicType);
            }
        }

        [TestMethod]
        public void ScoreModelCheckOutput()
        {
            var module = JIT.Module.Load(@"E:\Source\Repos\libtorch\model.pt");
            Assert.IsNotNull(module);

            var num = module.GetNumberOfOutputs();

            for (int i = 0; i < num; i++)
            {
                var type = module.GetOutputType(i);

                Assert.IsNotNull(type as DynamicType);
            }
        }

        [TestMethod]
        public void CreateLinear()
        {
            var lin = NN.Module.Linear(1000, 100);
            Assert.IsNotNull(lin);
            var modules = lin.GetName();
        }

        [TestMethod]
        public void TestGetBiasInLinear()
        {
            var lin = NN.Module.Linear(1000, 100);
            Assert.IsFalse(lin.WithBias);
            Assert.ThrowsException<ArgumentNullException>(() => lin.Bias);
        }

        [TestMethod]
        public void TestSetGetBiasInLinear()
        {
            var lin = NN.Module.Linear(1000, 100, true);
            Assert.IsNotNull(lin.Bias);

            var bias = FloatTensor.Ones(new long[] { 1000 });

            lin.Bias = bias;

            Assert.AreEqual(lin.Bias.NumberOfElements, bias.NumberOfElements);
        }

        [TestMethod]
        public void CreateRelu()
        {
            var rel = NN.Module.Relu();
            Assert.IsNotNull(rel);
            var modules = rel.GetName();
        }

        [TestMethod]
        public void CreateSequence()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);
            var modules = seq.GetModules();
        }

        [TestMethod]
        public void EvalSequence()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0", requiresGrad: true);
            var eval = seq.Forward(x);
            Assert.IsNotNull(eval);
        }

        [TestMethod]
        public void EvalLossSequence()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            var eval = seq.Forward(x);
            var loss = NN.LossFunction.MSE(eval, y, NN.Reduction.Sum);

            var result = loss.DataItem;
            Assert.IsNotNull(result);
        }

        [TestMethod]
        public void TestPoissonNLLLoss()
        {
            using (FloatTensor input = FloatTensor.From(new float[] { 0.5f, 1.5f, 2.5f }))
            using (FloatTensor target = FloatTensor.From(new float[] { 1f, 2f, 3f }))
            {
                var componentWiseLoss = ((FloatTensor)input.Exp()) - target * input;
                Assert.IsTrue(componentWiseLoss.Equal(NN.LossFunction.PoissonNLL(input, target, reduction: NN.Reduction.None)));
                Assert.IsTrue(componentWiseLoss.Sum().Equal(NN.LossFunction.PoissonNLL(input, target, reduction: NN.Reduction.Sum)));
                Assert.IsTrue(componentWiseLoss.Mean().Equal(NN.LossFunction.PoissonNLL(input, target, reduction: NN.Reduction.Mean)));
            }
        }

        [TestMethod]
        public void TestZeroGrad()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            seq.ZeroGrad();
        }

        [TestMethod]
        public void TestBackward()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            var eval = seq.Forward(x);
            var loss = NN.LossFunction.MSE(eval, y, NN.Reduction.None);

            seq.ZeroGrad();

            loss.Backward();
        }

        [TestMethod]
        public void TestGettingParameters()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            var eval = seq.Forward(x);
            var loss = NN.LossFunction.MSE(eval, y, NN.Reduction.None);

            seq.ZeroGrad();

            loss.Backward();

            foreach (var parm in seq.Parameters())
            {
                Assert.IsNotNull(parm);
            }
        }

        [TestMethod]
        public void TestGrad()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            var eval = seq.Forward(x);
            var loss = NN.LossFunction.MSE(eval, y, NN.Reduction.None);

            seq.ZeroGrad();

            loss.Backward();

            foreach (var parm in seq.Parameters())
            {
                var grad = parm.Grad();
                Assert.IsNotNull(grad);
            }
        }

        [TestMethod]
        public void TestAutoGradMode()
        {
            var x = FloatTensor.RandomN(new long[] { 2, 3 }, device: "cpu:0", requiresGrad: true);
            using (var mode = new AutoGradMode(false))
            {
                Assert.IsFalse(AutoGradMode.IsAutogradEnabled());
                var sum = x.Sum();
                sum.Backward();
                var grad = x.Grad();
                Assert.IsTrue(grad.Handle == IntPtr.Zero);
            }
            using (var mode = new AutoGradMode(true))
            {
                Assert.IsTrue(AutoGradMode.IsAutogradEnabled());
                var sum = x.Sum();
                sum.Backward();
                var grad = x.Grad();
                Assert.IsFalse(grad.Handle == IntPtr.Zero);
                var data = grad.Data;
                for (int i = 0; i < 2 * 3; i++)
                {
                    Assert.AreEqual(data[i], 1.0);
                }
            }
        }

        [TestMethod]
        public void TestSubInPlace()
        {
            var x = IntTensor.Ones(new long[] { 100, 100 });
            var y = IntTensor.Ones(new long[] { 100, 100 });

            x.SubInPlace(y);

            var xdata = x.Data;

            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    Assert.AreEqual(0, xdata[i + j]);
                }
            }
        }

        [TestMethod]
        public void TestMul()
        {
            var x = FloatTensor.Ones(new long[] { 100, 100 });

            var y = x.Mul(0.5f);

            var ydata = y.Data;
            var xdata = x.Data;

            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    Assert.AreEqual(ydata[i + j], xdata[i + j] * 0.5f);
                }
            }
        }

        /// <summary>
        /// Fully connected Relu net with one hidden layer trained using gradient descent.
        /// Taken from <see cref="https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html"/>.
        /// </summary>
        [TestMethod]
        public void TestTraining()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device: "cpu:0");
            var y = FloatTensor.RandomN(new long[] { 64, 10 }, device: "cpu:0");

            float learning_rate = 0.00004f;
            float prevLoss = float.MaxValue;

            for (int i = 0; i < 10; i++)
            {
                var eval = seq.Forward(x);
                var loss = NN.LossFunction.MSE(eval, y, NN.Reduction.Sum);
                var lossVal = loss.DataItem;

                Assert.IsTrue(lossVal < prevLoss);
                prevLoss = lossVal;

                seq.ZeroGrad();

                loss.Backward();

                using (var noGrad = new AutoGradMode(false))
                {
                    foreach (var param in seq.Parameters())
                    {
                        var grad = param.Grad();
                        var update = grad.Mul(learning_rate);
                        param.SubInPlace(update);
                    }
                }
            }
        }

        [TestMethod]
        public void TestAdam()
        {
            var lin1 = NN.Module.Linear(1000, 100);
            var lin2 = NN.Module.Linear(100, 10);
            var seq = NN.Module.Sequential(lin1, NN.Module.Relu(), lin2);

            double learning_rate = 0.00001;

            var optimizer = NN.Optimizer.Adam(seq.Parameters(), learning_rate);

            Assert.IsNotNull(optimizer);
        }

        /// <summary>
        /// Fully connected Relu net with one hidden layer trained using Adam optimizer.
        /// Taken from <see cref="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim"/>.
        /// </summary>
        [TestMethod]
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

            for (int i = 0; i < 10; i++)
            {
                var eval = seq.Forward(x);
                var loss = NN.LossFunction.MSE(eval, y, NN.Reduction.Sum);
                var lossVal = loss.DataItem;

                Assert.IsTrue(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.ZeroGrad();

                loss.Backward();

                optimizer.Step();
            }
        }

        [TestMethod]
        public void TestMNISTLoader()
        {
            using (var train = Data.Loader.MNIST(@"E:/Source/Repos/LibTorchSharp/MNIST", 32))
            {
                Assert.IsNotNull(train);

                var size = train.Size();
                int i = 0;

                Assert.IsNotNull(size);

                foreach (var (data, target) in train)
                {
                    i++;

                    CollectionAssert.AreEqual(data.Shape, new long[] { 32, 1, 28, 28 });
                    CollectionAssert.AreEqual(target.Shape, new long[] { 32 });

                    data.Dispose();
                    target.Dispose();
                }

                Assert.AreEqual(size, i * 32);
            }
        }

        [TestMethod]
        public void TestMNISTLoaderWithEpochs()
        {
            using (var train = Data.Loader.MNIST(@"E:/Source/Repos/LibTorchSharp/MNIST", 32))
            {
                var size = train.Size();
                var epochs = 10;

                Assert.IsNotNull(train);
                Assert.IsNotNull(size);

                int i = 0;

                for (int e = 0; e < epochs; e++)
                {
                    foreach (var (data, target) in train)
                    {
                        i++;

                        CollectionAssert.AreEqual(data.Shape, new long[] { 32, 1, 28, 28 });
                        CollectionAssert.AreEqual(target.Shape, new long[] { 32 });

                        data.Dispose();
                        target.Dispose();
                    }
                }

                Assert.AreEqual(size * epochs, i * 32);
            }
        }
    }
}