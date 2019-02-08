using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TorchSharp.Test
{
    [TestClass]
    public class TorchSharp
    {
        [TestMethod]
        public void CreateFloatTensorOnes()
        {
            var ones = FloatTensor.Ones( new long[]{ 2, 2} );
            Assert.IsNotNull(ones);
        }

        [TestMethod]
        public void CreateFloatTensorOnesAccessData()
        {
            unsafe
            {
                var ones = FloatTensor.Ones(new long[] { 2, 2 });
                var data = new System.Span<float>(ones.Data.ToPointer(), 4);

                for (int i = 0; i < 4; i++)
                {
                    Assert.AreEqual(data[i], 1.0);
                }
            }
        }

        [TestMethod]
        public void CreateIntTensorOnesAccessData()
        {
            unsafe
            {
                var ones = IntTensor.Ones(new long[] { 2, 2 });
                var data = new System.Span<int>(ones.Data.ToPointer(), 4);

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

            var module = JIT.Module.LoadModule(@"E:\Source\Repos\libtorch\model.pt");
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

            var x = FloatTensor.RandomN(new long[] { 64, 1000 }, device : "cpu:0", requiresGrad : true);
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
            var loss = LossFunction.MSELoss(eval, y, Reduction.Sum);

            var result = loss.Item();
            Assert.IsNotNull(result);
        }
    }
}