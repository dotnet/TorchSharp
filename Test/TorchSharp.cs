using Microsoft.VisualStudio.TestTools.UnitTesting;
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
        public void CreateFloatTensorOnesCheckData()
        {
            unsafe
            {
                var ones = FloatTensor.Ones(new long[] { 2, 2 });
                var data = ones.Data;

                for (int i = 0; i < 4; i++)
                {
                    Assert.AreEqual(data[i], 1.0);
                }
            }
        }

        [TestMethod]
        public void CreateIntTensorOnesCheckData()
        {
            unsafe
            {
                var ones = IntTensor.Ones(new long[] { 2, 2 });
                var data = ones.Data;

                for (int i = 0; i < 4; i++)
                {
                    Assert.AreEqual(data[i], 1);
                }
            }
        }

        [TestMethod]
        public void ScoreModel()
        {
            var ones = FloatTensor.Ones(new long[] { 1, 3, 224, 224 });
            Assert.IsNotNull(ones);

            var module = JIT.Module.Load(@"E:\Source\Repos\libtorch\model.pt");
            Assert.IsNotNull(module);

            var result = module.Forward(ones);
            Assert.IsNotNull(result);
        }

        [TestMethod]
        public void CreateLinear()
        {
            var lin = NN.Module.Linear(1000, 100);
            Assert.IsNotNull(lin);
            var modules = lin.GetName();
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
            var loss = NN.LossFunction.MSE(eval, y, NN.Reduction.None);

            var result = loss.Item;
            Assert.IsNotNull(result);
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
        public void TestSubInPlace()
        {
            var x = IntTensor.Ones(new long[] { 100, 100 });
            var y = IntTensor.Ones(new long[] { 100, 100 });

            var z = x.SubInPlace(y);

            var zdata = z.Data;
            var xdata = x.Data;

            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    Assert.AreEqual(zdata[i + j], xdata[i + j]);
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
                var loss = NN.LossFunction.MSE(eval, y, NN.Reduction.None);
                var lossVal = loss.Item;

                Assert.IsTrue(lossVal < prevLoss);
                prevLoss = lossVal;

                seq.ZeroGrad();

                loss.Backward();

                // using(var noGrad = NN.NoGrad())
                // The operators Mul and SubInPlace have no_grad=true by default
                foreach (var param in seq.Parameters())
                {
                    var grad = param.Grad();
                    var update = grad.Mul(learning_rate);
                    param.SubInPlace(update);
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

            var optimizer = NN.Optimizer.Adam(seq, learning_rate);

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
            var optimizer = NN.Optimizer.Adam(seq, learning_rate);

            for (int i = 0; i < 10; i++)
            {
                var eval = seq.Forward(x);
                var loss = NN.LossFunction.MSE(eval, y, NN.Reduction.Sum);
                var lossVal = loss.Item;

                Assert.IsTrue(lossVal < prevLoss);
                prevLoss = lossVal;

                optimizer.ZeroGrad();

                loss.Backward();

                optimizer.Step();
            }
        }
    }
}